import sys
import os
import xml.etree.ElementTree as ET
import torch
import logging
import warnings
from tqdm import tqdm
import torch.optim as optim
from datetime import datetime
import numpy as np
import torch.nn.functional as F
import multiprocessing
import random
import importlib.util
from typing import Type
from PIL import Image

from dataset.landslide_dataset import LandslideDataset


EPS = 1e-8

def worker_init_fn(worker_id):
    """Set independent random seeds for each worker process"""
    base_seed = getattr(worker_init_fn, 'base_seed', 114514)
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)

def setup_device():
    """Set up training device (GPU preferred)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_count = torch.cuda.device_count()
        logging.info(f"Training with GPU, number of devices: {device_count}")
    else:
        device = torch.device("cpu")
        device_count = 1
        logging.info("No GPU detected, training with CPU")
    return device, device_count

def _parse_single_xml(xml_file):
    """Parse a single XML file and return parameter dictionary"""
    if not os.path.exists(xml_file):
        raise FileNotFoundError(f"Configuration file not found: {xml_file}")
    
    root = ET.parse(xml_file).getroot()
    config = {}
    
    for param in root.findall('param'):
        name = param.find('name').text
        value = param.find('value').text
        
        if param.find('type').text == 'int':
            config[name] = int(value)
        elif param.find('type').text == 'float':
            config[name] = float(value)
        elif param.find('type').text == 'bool':
            config[name] = value.lower() == 'true'
        elif param.find('type').text == 'list':
            config[name] = eval(value)
        else:
            config[name] = value if value != 'None' else None
    
    return config

def parse_config(xml_file, parsed_files=None):
    """
    Parse XML configuration file with support for base configuration inheritance
    xml_file: Path to current configuration file
    parsed_files: Set of parsed files to detect circular references
    """
    if parsed_files is None:
        parsed_files = set()
    
    xml_abs_path = os.path.abspath(xml_file)
    if xml_abs_path in parsed_files:
        raise RuntimeError(f"Circular reference in configuration files: {xml_abs_path}")
    parsed_files.add(xml_abs_path)
    
    current_config = _parse_single_xml(xml_file)
    
    base_param = current_config.pop('base_param', None)
    if base_param:
        current_dir = os.path.dirname(xml_abs_path)
        base_xml_path = os.path.join(current_dir, base_param)
        
        base_config = parse_config(base_xml_path, parsed_files)
        
        base_config.update(current_config)
        return base_config
    else:
        current_config.setdefault('preprocess_npz', True)
        current_config.setdefault('cache_dir', os.path.join(current_config.get('data_root', ''), 'npz_cache'))
        current_config.setdefault('cache_max_size', 50)
        current_config.setdefault('lr', 1e-4)
        current_config.setdefault('weight_decay', 1e-5)
        current_config.setdefault('pred_threshold', 0.5)
        current_config.setdefault('test_pred_save_dir', os.path.join(current_config.get('checkpoint_dir', ''), 'test_predictions'))
        return current_config

def setup_logger(log_dir):
    """Initialize logging system"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logger = logging.getLogger('landslide_training')
    if logger.hasHandlers():
        logger.handlers.clear()
    
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    return logger

def setup_random_seeds(seed):
    """Initialize all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logging.info(f"All random seeds initialized with seed value: {seed}")
    worker_init_fn.base_seed = seed

def create_dataloaders(config):
    """Create data loaders (adjusted to match sample_names attribute of LandslideDataset)"""
    setup_random_seeds(config['seed'])
    
    full_dataset = LandslideDataset(
        root_dir=config['data_root'],
        mode='full',
        exclude_features=config['exclude_features'],
        seed=config['seed'],
        use_png_labels=config['use_png_labels'],
        use_data_cache=config.get('use_data_cache', True),
        preprocess_npz=config['preprocess_npz'],
        cache_dir=config['cache_dir'],
    )
    full_dataset.cache_max_size = config['cache_max_size']
    
    sample_features, sample_labels = full_dataset[0]
    valid_label_mask = (sample_labels != -100)
    if valid_label_mask.sum() == 0:
        logging.warning("Warning: Dataset contains samples with no valid labels, which may cause training abnormalities")
    if (sample_labels[valid_label_mask] == 1).sum() == 0:
        logging.warning("Warning: No landslide labels (class 1) found in dataset, which may cause metric calculation abnormalities")
    
    feature_info = full_dataset.get_feature_info()
    feature_types = full_dataset.feature_types
    logger = logging.getLogger('landslide_training')
    logger.info("Feature information:")
    for feature, bands in feature_info.items():
        logger.info(f"  {feature}: {bands} bands")
    
    assert 'before' in feature_info and 'after' in feature_info, "Missing pre-disaster/post-disaster features"
    assert feature_info['before'] == feature_info['after'], "Mismatched band count between pre-disaster and post-disaster features"
    
    total_samples = len(full_dataset)
    train_size = int(config['train_ratio'] * total_samples)
    val_size = int(config['val_ratio'] * total_samples)
    test_size = total_samples - train_size - val_size
    
    generator = torch.Generator().manual_seed(config['seed'])
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=generator
    )
    
    train_dataset.sample_names = [full_dataset.sample_names[idx] for idx in train_dataset.indices]
    val_dataset.sample_names = [full_dataset.sample_names[idx] for idx in val_dataset.indices]
    test_dataset.sample_names = [full_dataset.sample_names[idx] for idx in test_dataset.indices]
    
    train_dataset.dataset.mode = 'train'
    val_dataset.dataset.mode = 'val'
    test_dataset.dataset.mode = 'test'
    
    logger.info("\nDataset split:")
    logger.info(f"  Training set: {len(train_dataset)} samples")
    logger.info(f"  Validation set: {len(val_dataset)} samples")
    logger.info(f"  Test set: {len(test_dataset)} samples")
    
    dataloader_kwargs = {
        'batch_size': config['batch_size'],
        'pin_memory': True,
        'persistent_workers': True,
        'num_workers': min(config['num_workers'], multiprocessing.cpu_count() // 2),
        'worker_init_fn': worker_init_fn
    }
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,** dataloader_kwargs
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        **dataloader_kwargs
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,** dataloader_kwargs
    )
    
    return train_loader, val_loader, test_loader, feature_info, feature_types, full_dataset

def dynamic_import_model(model_path, model_name):
    """Dynamically import model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    module_name = os.path.splitext(os.path.basename(model_path))[0]
    
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    if spec is None:
        raise ImportError(f"Failed to create module spec from path {model_path}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    if not hasattr(module, model_name):
        raise AttributeError(f"Model class {model_name} not found in module {model_path}")
    
    return getattr(module, model_name)

def calculate_iou(pred, target, mask, num_classes=2, device=None):
    """Calculate IoU"""
    if device is None:
        device = pred.device
    
    ious = torch.zeros(num_classes, device=device)
    for cls in range(num_classes):
        pred_mask = (pred == cls).float() * mask
        target_mask = (target == cls).float() * mask
        
        intersection = (pred_mask * target_mask).sum(dim=[1, 2]).sum() + EPS
        union = (pred_mask + target_mask - pred_mask * target_mask).sum(dim=[1, 2]).sum() + EPS
        
        iou = intersection / union
        ious[cls] = iou
    
    return ious[1].item(), (ious.mean()).item()

def batch_metrics(pred, target, mask, num_classes=2, device=None):
    """Calculate batch metrics"""
    if device is None:
        device = pred.device
    
    pred_pos = (pred == 1).float() * mask
    target_pos = (target == 1).float() * mask
    pred_neg = ((pred == 0).float() * mask)
    target_neg = ((target == 0).float() * mask)
    
    tp = (pred_pos * target_pos).sum() + EPS
    tn = (pred_neg * target_neg).sum() + EPS
    fp = (pred_pos * target_neg).sum() + EPS
    fn = (pred_neg * target_pos).sum() + EPS
    
    return tp, tn, fp, fn

def aggregate_metrics(tp, tn, fp, fn):
    """Aggregate metrics"""
    def to_float(x):
        return x.item() if isinstance(x, torch.Tensor) else x
    
    tp_val = to_float(tp)
    fp_val = to_float(fp)
    fn_val = to_float(fn)
    
    precision = tp_val / (tp_val + fp_val) if (tp_val + fp_val) > EPS else 0.0
    recall = tp_val / (tp_val + fn_val) if (tp_val + fn_val) > EPS else 0.0
    f1 = 2 * (precision * recall) / (precision + recall + EPS)
    
    return precision, recall, f1

def split_features(features, feature_types, feature_info):
    """Split features"""
    current = 0
    x_before = x_after = None
    dynamic_inputs = []
    
    for feature in feature_types:
        bands = feature_info[feature]
        feature_tensor = features[:, current:current+bands, ...]
        if feature == 'before':
            x_before = feature_tensor
        elif feature == 'after':
            x_after = feature_tensor
        else:
            dynamic_inputs.append(feature_tensor)
        current += bands
    
    assert x_before is not None and x_after is not None, "Feature splitting failed"
    return x_before, x_after, dynamic_inputs

def evaluate_model(model, data_loader, criterion, device, feature_types, feature_info, full_dataset, 
                   phase='Validation', threshold=0.5):
    """Evaluate model"""
    full_dataset.clear_cache()
    
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_boundary = 0.0
    running_focal = 0.0
    running_tversky = 0.0
    
    total_tp = torch.tensor(0.0, device=device)
    total_tn = torch.tensor(0.0, device=device)
    total_fp = torch.tensor(0.0, device=device)
    total_fn = torch.tensor(0.0, device=device)
    
    total_slide_iou = 0.0
    total_mean_iou = 0.0
    iou_count = 0
    
    eval_start = datetime.now()
    
    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(tqdm(data_loader, desc=f"{phase} Evaluation")):
            features = features.to(device)
            labels = labels.to(device)
            
            x_before, x_after, dynamic_inputs = split_features(features, feature_types, feature_info)
            outputs = model(x_before, x_after, dynamic_inputs)
            
            loss_dict = criterion(outputs, labels)
            
            if torch.isnan(loss_dict['total']):
                logging.warning(f"Detected NaN loss! Batch index: {batch_idx}, skipping this batch")
                continue
            
            running_loss += loss_dict['total'].item()
            running_dice += loss_dict['dice'].item() if not torch.isnan(loss_dict['dice']) else 0.0
            running_boundary += loss_dict['boundary'].item() if not torch.isnan(loss_dict['boundary']) else 0.0
            running_focal += loss_dict['focal'].item() if not torch.isnan(loss_dict['focal']) else 0.0
            running_tversky += loss_dict['tversky'].item() if not torch.isnan(loss_dict['tversky']) else 0.0
            
            prob = F.softmax(outputs, dim=1)[:, 1]
            preds = (prob > threshold).long()
            
            mask = (labels != -100).float()
            valid_pixels = mask.sum()
            
            if valid_pixels < EPS:
                continue
            
            tp, tn, fp, fn = batch_metrics(preds, labels, mask, device=device)
            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn
            
            slide_iou, mean_iou = calculate_iou(preds, labels, mask, device=device)
            total_slide_iou += slide_iou
            total_mean_iou += mean_iou
            iou_count += 1
    
    avg_loss = running_loss / len(data_loader) if len(data_loader) > 0 else 0.0
    avg_dice = running_dice / len(data_loader) if len(data_loader) > 0 else 0.0
    avg_boundary = running_boundary / len(data_loader) if len(data_loader) > 0 else 0.0
    avg_focal = running_focal / len(data_loader) if len(data_loader) > 0 else 0.0
    avg_tversky = running_tversky / len(data_loader) if len(data_loader) > 0 else 0.0
    
    precision, recall, f1 = aggregate_metrics(total_tp, total_tn, total_fp, total_fn)
    
    slide_iou = total_slide_iou / iou_count if iou_count > 0 else 0.0
    mean_iou = total_mean_iou / iou_count if iou_count > 0 else 0.0
    
    eval_time = (datetime.now() - eval_start).total_seconds()
    result_str = (f"\n[{phase} Results] (Time elapsed: {eval_time:.2f}s, Threshold: {threshold})\n"
                  f"Loss details:\n"
                  f"  Total loss: {avg_loss:.4f}\n"
                  f"  Dice loss: {avg_dice:.4f}\n"
                  f"  Boundary loss: {avg_boundary:.4f}\n"
                  f"  Focal loss: {avg_focal:.4f}\n"
                  f"  Tversky loss: {avg_tversky:.4f}\n"
                  f"Landslide extraction metrics:\n"
                  f"  Precision: {precision:.4f} | Recall: {recall:.4f} | F1 score: {f1:.4f}\n"
                  f"  Slide IoU: {slide_iou:.4f} | Mean IoU: {mean_iou:.4f}")
    
    return result_str, {
        'loss': avg_loss, 'dice': avg_dice, 'boundary': avg_boundary,
        'focal': avg_focal, 'tversky': avg_tversky,
        'precision': precision, 'recall': recall, 'f1': f1,
        'slide_iou': slide_iou, 'mean_iou': mean_iou
    }

def save_test_predictions(model, test_loader, test_sample_names, config, device, feature_types, feature_info, full_dataset):
    """
    Generate and save test set prediction maps
    Inputs:
        - model: Trained best model
        - test_loader: Test set data loader
        - test_sample_names: List of test set sample names (one-to-one with prediction maps)
        - config: Configuration dictionary (contains save path, prediction threshold)
        - device: Computing device
        - feature_types/feature_info: Feature-related information
        - full_dataset: Complete dataset (used for cache cleaning)
    Output:
        - Save PNG format prediction maps to config['test_pred_save_dir']
    """
    pred_save_dir = config['test_pred_save_dir']
    os.makedirs(pred_save_dir, exist_ok=True)
    logger = logging.getLogger('landslide_training')
    logger.info(f"\nStarting generation of test set prediction maps, save path: {pred_save_dir}")
    
    full_dataset.clear_cache()
    model.eval()
    pred_idx = 0
    
    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(tqdm(test_loader, desc="Generating Test Predictions")):
            features = features.to(device)
            x_before, x_after, dynamic_inputs = split_features(features, feature_types, feature_info)
            outputs = model(x_before, x_after, dynamic_inputs)
            
            prob = F.softmax(outputs, dim=1)[:, 1]
            preds = (prob > config['pred_threshold']).long()
            
            for pred in preds:
                sample_name = test_sample_names[pred_idx]
                pred_save_path = os.path.join(pred_save_dir, f"{sample_name}_pred.png")
                
                pred_255 = (pred.cpu().numpy() * 255).astype(np.uint8)
                
                pred_img = Image.fromarray(pred_255)
                pred_img.save(pred_save_path)
                
                pred_idx += 1
    
    logger.info(f"Test set prediction maps generation completed! Generated {pred_idx} prediction maps in total")
    
