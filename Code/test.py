import sys
import os
import xml.etree.ElementTree as ET
import torch
import logging
import warnings
import numpy as np
import torch.nn.functional as F
import multiprocessing
from datetime import datetime
from PIL import Image
import importlib.util
import random
import time

from dataset.landslide_dataset import LandslideDataset
from loss.loss import LandslideLoss
from utils import *

warnings.filterwarnings("ignore", category=UserWarning)

def parse_xml_config(config_path):
    """Parse XML configuration file"""
    tree = ET.parse(config_path)
    root = tree.getroot()
    
    config = {}
    for param in root.findall('param'):
        name = param.find('name').text
        value = param.find('value').text
        
        type_str = param.find('type').text
        if type_str == 'int':
            config[name] = int(value)
        elif type_str == 'float':
            config[name] = float(value)
        elif type_str == 'bool':
            config[name] = value.lower() == 'true'
        elif type_str == 'list':
            if value.strip() == '[]':
                config[name] = []
            else:
                config[name] = [item.strip("'\" ") for item in value.strip('[]').split(',')]
        else:
            config[name] = value
    
    return config

def setup_logger(log_dir):
    """Set up logger"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logger = logging.getLogger('landslide_prediction')
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_file)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    return logger

def setup_device():
    """Set up computing device with CUDA configuration to avoid address alignment issues"""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        cuda_version = torch.version.cuda
        device = torch.device('cuda')
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        logging.info(f"Using GPU for prediction: {device_name}, Number of devices: {device_count}, CUDA version: {cuda_version}")
        
        try:
            major, minor = torch.cuda.get_device_capability(device)
            logging.info(f"GPU compute capability: {major}.{minor}")
            if (major == 7 and minor <= 5) or major < 7:
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                logging.info("Disabled TF32 to improve compatibility")
        except Exception as e:
            logging.warning(f"Failed to detect GPU compute capability: {str(e)}")
    else:
        device = torch.device('cpu')
        device_count = 1
        logging.info("Using CPU for prediction")
    return device, device_count

def setup_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def dynamic_import_model(model_path, model_name):
    """Dynamically import model class"""
    spec = importlib.util.spec_from_file_location(model_name, model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, model_name)

def count_parameters(model):
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def move_to_device(obj, device):
    """
    Recursively move objects or object lists to specified device
    Handles tensor, list, and tuple types
    """
    if isinstance(obj, torch.Tensor):
        if device.type == 'cuda' and obj.numel() > 0:
            return obj.contiguous().to(device)
        return obj.to(device)
    elif isinstance(obj, list):
        return [move_to_device(item, device) for item in obj]
    elif isinstance(tuple(obj), tuple):
        return tuple(move_to_device(item, device) for item in obj)
    else:
        return obj

def calculate_layer_flops(layer, input_data):
    """Calculate FLOPs for a single layer"""
    flops = 0
    
    if isinstance(layer, torch.nn.Conv2d):
        in_channels = input_data.shape[1]
        out_channels = layer.out_channels
        kernel_size = layer.kernel_size[0]
        output_shape = layer(input_data).shape[2:]
        
        elements = output_shape[0] * output_shape[1]
        flops_per_element = in_channels * kernel_size * kernel_size
        flops = out_channels * elements * flops_per_element
        
    elif hasattr(layer, 'depthwise') and hasattr(layer, 'pointwise'):
        depthwise = layer.depthwise
        in_channels = input_data.shape[1]
        kernel_size = depthwise.kernel_size[0]
        output_shape = depthwise(input_data).shape[2:]
        
        elements = output_shape[0] * output_shape[1]
        depthwise_flops = in_channels * elements * kernel_size * kernel_size
        
        pointwise = layer.pointwise
        out_channels = pointwise.out_channels
        pointwise_flops = out_channels * elements * in_channels
        
        flops = depthwise_flops + pointwise_flops
        
    elif isinstance(layer, torch.nn.Linear):
        in_features = input_data.view(-1).shape[0]
        out_features = layer.out_features
        flops = in_features * out_features
        
    elif isinstance(layer, torch.nn.BatchNorm2d):
        elements = input_data.numel()
        flops = 2 * elements
        
    elif isinstance(layer, (torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.Sigmoid, torch.nn.Softmax)):
        flops = input_data.numel()
        
    elif isinstance(layer, (torch.nn.MaxPool2d, torch.nn.AvgPool2d)):
        output = layer(input_data)
        kernel_size = layer.kernel_size if isinstance(layer.kernel_size, int) else layer.kernel_size[0]
        flops = output.numel() * (kernel_size ** 2)
        
    return flops

def calculate_model_flops(model, input_shape, device):
    """Recursively calculate FLOPs for the entire model"""
    model = model.to(device)
    model.eval()
    
    input_data = move_to_device(input_shape, device)
    
    layer_flops = {}
    total_flops = 0
    
    def hook_fn(module, input, output, name):
        nonlocal total_flops
        try:
            input_tensor = input[0] if isinstance(input, tuple) else input
            flops = calculate_layer_flops(module, input_tensor)
            layer_flops[name] = flops
            total_flops += flops
        except Exception as e:
            logging.debug(f"Failed to calculate FLOPs for layer {name}: {str(e)}")
    
    hooks = []
    for name, module in model.named_modules():
        if name:
            hook = module.register_forward_hook(
                lambda m, i, o, n=name: hook_fn(m, i, o, n)
            )
            hooks.append(hook)
    
    with torch.no_grad():
        model(*input_data)
    
    for hook in hooks:
        hook.remove()
    
    return total_flops, layer_flops

def calculate_model_complexity(model, input_shape, device):
    """Accurately calculate model complexity (FLOPs)"""
    try:
        total_flops, layer_flops = calculate_model_flops(model, input_shape, device)
        
        significant_layers = {k: v for k, v in layer_flops.items() if v > total_flops * 0.01}
        if significant_layers:
            logging.debug("FLOPs distribution of major layers:")
            for name, flops in sorted(significant_layers.items(), key=lambda x: x[1], reverse=True):
                logging.debug(f"  {name}: {flops/1e6:.2f} MFLOPs ({flops/total_flops*100:.1f}%)")
        
        model = model.to(device)
        model.eval()
        input_on_device = move_to_device(input_shape, device)
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                model(*input_on_device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        avg_time = (time.time() - start_time) / 100
        
        return total_flops, avg_time
    except Exception as e:
        logging.error(f"Error occurred while calculating model complexity: {str(e)}")
        raise

def create_prediction_dataloader(config):
    """Create dataset and dataloader for prediction (without NPZ caching)"""
    dataset = LandslideDataset(
        root_dir=config['data_root'],
        mode='full',
        exclude_features=config['exclude_features'],
        seed=config.get('seed', 42),
        use_data_cache=False,
        preprocess_npz=False,
        cache_dir=None
    )
    
    feature_info = dataset.get_feature_info()
    feature_types = dataset.feature_types
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', multiprocessing.cpu_count() // 2),
        pin_memory=True if torch.cuda.is_available() else False,
        pin_memory_device=str(torch.device('cuda')) if torch.cuda.is_available() else ''
    )
    
    return dataloader, feature_info, feature_types, dataset

def generate_visualization(preds, labels, threshold=0.5):
    """Generate visualization of prediction results"""
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    preds_binary = (preds > threshold).astype(np.uint8)
    h, w = preds_binary.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    tp_mask = (preds_binary == 1) & (labels == 1)
    vis[tp_mask] = [255, 255, 255]
    
    fp_mask = (preds_binary == 1) & (labels == 0)
    vis[fp_mask] = [52, 172, 254]
    
    fn_mask = (preds_binary == 0) & (labels == 1)
    vis[fn_mask] = [255, 0, 0]
    
    return vis

def load_parallel_model_weights(model, weights_path, logger):
    """
    Load model weights trained with nn.DataParallel, handling "module." prefix issue
    """
    saved_state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    
    new_state_dict = {}
    for key, value in saved_state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    model_state_dict = model.state_dict()
    compatible_state_dict = {}
    missing_keys = []
    unexpected_keys = []
    
    for key, value in new_state_dict.items():
        if key in model_state_dict:
            if model_state_dict[key].shape == value.shape:
                compatible_state_dict[key] = value
            else:
                logger.warning(f"Skipping mismatched parameter {key}: model expects {model_state_dict[key].shape}, weights provide {value.shape}")
                missing_keys.append(key)
        else:
            unexpected_keys.append(key)
    
    for key in model_state_dict:
        if key not in compatible_state_dict:
            missing_keys.append(key)
    
    if missing_keys:
        logger.warning(f"Detected {len(missing_keys)} missing parameter keys, default initialization will be used")
    if unexpected_keys:
        logger.warning(f"Detected {len(unexpected_keys)} unexpected parameter keys, which will be ignored")
    
    model.load_state_dict(compatible_state_dict, strict=False)
    return model

def predict(model, dataloader, config, device, feature_types, feature_info, dataset, logger):
    """Make predictions using the model and save results"""
    os.makedirs(config['output_dir'], exist_ok=True)
    logger.info(f"Prediction results will be saved to: {config['output_dir']}")
    
    device_count = torch.cuda.device_count()
    use_multi_gpu = device_count > 1 and config.get('use_multi_gpu', False)
    
    if use_multi_gpu:
        logger.info(f"Using {device_count} GPUs for prediction")
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.eval()
    
    total_start_time = time.time()
    total_samples = len(dataset)
    total_processing_time = 0
    
    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0
    
    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(dataloader):
            start_idx = batch_idx * dataloader.batch_size
            end_idx = start_idx + len(features)
            batch_sample_names = dataset.sample_names[start_idx:end_idx]
            
            logger.info(f"Processing batch {batch_idx + 1}/{len(dataloader)}, samples: {batch_sample_names[0]} to {batch_sample_names[-1]}")
            
            batch_start_time = time.time()
            
            features = features.contiguous().to(device, non_blocking=True)
            labels = labels.contiguous().to(device, non_blocking=True)
            
            x_before, x_after, dynamic_inputs = split_features(features, feature_types, feature_info)
            
            inference_start = time.time()
            try:
                outputs = model(x_before, x_after, dynamic_inputs)
            except RuntimeError as e:
                if "misaligned address" in str(e):
                    logger.error("Detected address alignment error, trying single GPU mode...")
                    model = model.module if isinstance(model, torch.nn.DataParallel) else model
                    outputs = model(x_before, x_after, dynamic_inputs)
                    use_multi_gpu = False
                else:
                    raise
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            inference_time = time.time() - inference_start
            total_processing_time += inference_time
            
            prob = F.softmax(outputs, dim=1)[:, 1]
            preds = (prob > config['pred_threshold']).long()
            mask = (labels != -100).float()
            
            for i in range(len(batch_sample_names)):
                sample_name = batch_sample_names[i]
                pred = preds[i]
                label = labels[i]
                msk = mask[i]
                
                valid_pixels = msk.sum()
                if valid_pixels > 0:
                    tp, tn, fp, fn = batch_metrics(pred.unsqueeze(0), label.unsqueeze(0), msk.unsqueeze(0), device=device)
                    total_tp += tp.item()
                    total_tn += tn.item()
                    total_fp += fp.item()
                    total_fn += fn.item()
                
                vis = generate_visualization(
                    pred.cpu().numpy(), 
                    label.cpu().numpy(),
                    config['pred_threshold']
                )
                
                output_path = os.path.join(config['output_dir'], f"{sample_name}.png")
                Image.fromarray(vis).save(output_path)
            
            batch_time = time.time() - batch_start_time
            batch_fps = len(features) / batch_time
            logger.info(f"Batch {batch_idx + 1} processed, time elapsed: {batch_time:.4f}s, FPS: {batch_fps:.2f}")
    
    total_time = time.time() - total_start_time
    overall_fps = total_samples / total_time
    inference_fps = total_samples / total_processing_time
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn) if (total_tp + total_tn + total_fp + total_fn) > 0 else 0
    
    logger.info("\n===== Prediction Performance Metrics =====")
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Total processing time: {total_time:.4f}s")
    logger.info(f"Total inference time: {total_processing_time:.4f}s")
    logger.info(f"Overall FPS (including IO): {overall_fps:.2f}")
    logger.info(f"Inference FPS (computation only): {inference_fps:.2f}")
    logger.info(f"Average sample processing time: {total_time/total_samples:.4f}s")
    
    logger.info("\n===== Prediction Accuracy Metrics =====")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"True Positives: {total_tp}")
    logger.info(f"True Negatives: {total_tn}")
    logger.info(f"False Positives: {total_fp}")
    logger.info(f"False Negatives: {total_fn}")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'fps': overall_fps,
        'inference_fps': inference_fps,
        'total_time': total_time
    }

def main(config_path):
    """Main function"""
    config = parse_xml_config(config_path)
    
    logger = setup_logger(config['log_dir'])
    logger.info("===== Landslide Prediction Program Started =====")
    logger.info("Configuration parameters:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    setup_random_seeds(config.get('seed', 42))
    
    device, device_count = setup_device()
    multiprocessing.set_start_method('spawn', force=True)
    
    dataloader, feature_info, feature_types, dataset = create_prediction_dataloader(config)
    logger.info(f"Dataset loaded, total {len(dataset)} samples")
    
    sample_features, _ = dataset[0]
    sample_features = sample_features.unsqueeze(0)
    x_before, x_after, dynamic_inputs = split_features(sample_features, feature_types, feature_info)
    input_shape = (x_before, x_after, dynamic_inputs)
    
    try:
        LandslideNet = dynamic_import_model(config['model_path'], config['model_name'])
        logger.info(f"Successfully imported model: {config['model_name']} from {config['model_path']}")
    except Exception as e:
        logger.error(f"Failed to import model: {str(e)}")
        raise
    
    dynamic_branch_channels = [
        feature_info[feature] for feature in feature_types if feature not in ['before', 'after']
    ]
    logger.info(f"Dynamic feature branch channels: {dynamic_branch_channels}")
    
    fixed_in_channels = feature_info['before']
    model = LandslideNet(
        fixed_in_channels=fixed_in_channels,
        dynamic_in_channels=dynamic_branch_channels
    )
    
    total_params, trainable_params = count_parameters(model)
    logger.info(f"Model parameters: Total={total_params:,} ({total_params/1e6:.2f}M), Trainable={trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    try:
        flops, avg_inference_time = calculate_model_complexity(model, input_shape, device)
        if flops >= 1e9:
            flops_str = f"{flops/1e9:.2f} GFLOPs"
        elif flops >= 1e6:
            flops_str = f"{flops/1e6:.2f} MFLOPs"
        else:
            flops_str = f"{flops/1e3:.2f} KFLOPs"
            
        logger.info(f"Model complexity: {flops_str}, Inference time per sample: {avg_inference_time:.6f}s")
    except Exception as e:
        logger.warning(f"Failed to calculate model complexity: {str(e)}")
        logger.info("Continuing with prediction process...")
    
    logger.info(f"Model initialization completed - Fixed branch channels: {fixed_in_channels}, Number of dynamic branches: {len(dynamic_branch_channels)}")
    
    if not os.path.exists(config['model_weights']):
        logger.error(f"Model weights file not found: {config['model_weights']}")
        raise FileNotFoundError(f"Model weights file not found: {config['model_weights']}")
    
    model = load_parallel_model_weights(model, config['model_weights'], logger)
    logger.info(f"Successfully loaded model weights: {config['model_weights']}")
    
    metrics = predict(
        model=model,
        dataloader=dataloader,
        config=config,
        device=device,
        feature_types=feature_types,
        feature_info=feature_info,
        dataset=dataset,
        logger=logger
    )
    
    logger.info("===== Prediction Program Completed =====")
    return metrics

if __name__ == '__main__':
    try:
        if len(sys.argv) < 2:
            raise RuntimeError("Please provide the configuration file path as an argument")
        config_path = sys.argv[1]
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        main(config_path)
        print('<prediction_status>0</prediction_status>')
        print('<prediction_log>Prediction successful</prediction_log>')
    except Exception as e:
        error_msg = str(e).replace('\n', ' ').replace('\t', ' ')
        print('<prediction_status>1</prediction_status>')
        print(f'<prediction_log>{error_msg}</prediction_log>')
        
        logger = logging.getLogger('landslide_prediction')
        if not logger.hasHandlers():
            logger.addHandler(logging.StreamHandler())
        logger.error(f"Prediction failed: {error_msg}")
        sys.exit(1)
        
