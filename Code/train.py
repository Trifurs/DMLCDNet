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

from dataset.landslide_dataset import LandslideDataset
from loss.loss import LandslideLoss
from utils import *

warnings.filterwarnings("ignore", category=UserWarning)

def train_model(model, train_loader, val_loader, test_loader, test_sample_names, config, device, 
                feature_types, feature_info, full_dataset, logger):
    """Train model (added test_sample_names parameter for generating prediction maps later)"""
    device_count = torch.cuda.device_count()
    if device_count > 1:
        logger.info(f"Using {device_count} GPUs for training")
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    
    criterion = LandslideLoss(
        dice_weight=config['dice_weight'],
        boundary_weight=config['boundary_weight'],
        focal_weight=config['focal_weight'],
        tversky_weight=config['tversky_weight']
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True, threshold_mode='abs'
    )
    
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    start_epoch = 1
    best_model_path = None
    if config['resume'] and os.path.exists(config['resume']):
        model.load_state_dict(torch.load(config['resume'], map_location=device))
        logger.info(f"Resuming training from checkpoint: {config['resume']}")
        start_epoch = int(os.path.basename(config['resume']).split('_')[-1].split('.')[0]) + 1
        best_model_path = config['resume']
    
    best_slide_iou = -1e9
    best_val_metrics = None
    no_improvement_counter = 0
    best_epoch = start_epoch
    early_stop_patience = config.get('early_stop_patience', 10)
    
    initial_model_path = os.path.join(config['checkpoint_dir'], f"initial_model_epoch_{start_epoch}.pth")
    torch.save(model.state_dict(), initial_model_path)
    logger.info(f"Saved initial model to {initial_model_path}")
    if best_model_path is None:
        best_model_path = initial_model_path
    
    logger.info("\n===== Starting Training =====")
    logger.info(f"Learning rate: {config['lr']}, Early stopping patience: {early_stop_patience}, Prediction threshold: {config['pred_threshold']}")
    
    for epoch in range(start_epoch, config['epochs'] + 1):
        model.train()
        epoch_start = datetime.now()
        running_total = 0.0
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
        
        data_load_time = 0.0
        compute_time = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['epochs']}")
        
        for batch_idx, (features, labels) in enumerate(progress_bar):
            batch_start = datetime.now()
            
            features = features.to(device)
            labels = labels.to(device)
            x_before, x_after, dynamic_inputs = split_features(features, feature_types, feature_info)
            
            compute_start = datetime.now()
            optimizer.zero_grad()
            outputs = model(x_before, x_after, dynamic_inputs)
            loss_dict = criterion(outputs, labels)
            
            if torch.isnan(loss_dict['total']):
                logger.warning(f"NaN loss in Epoch {epoch} Batch {batch_idx}, skipping backpropagation")
                compute_time += (datetime.now() - compute_start).total_seconds()
                data_load_time += (datetime.now() - batch_start).total_seconds()
                continue
            
            total_loss = loss_dict['total']
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            compute_time += (datetime.now() - compute_start).total_seconds()
            
            running_total += loss_dict['total'].item()
            running_dice += loss_dict['dice'].item() if not torch.isnan(loss_dict['dice']) else 0.0
            running_boundary += loss_dict['boundary'].item() if not torch.isnan(loss_dict['boundary']) else 0.0
            running_focal += loss_dict['focal'].item() if not torch.isnan(loss_dict['focal']) else 0.0
            running_tversky += loss_dict['tversky'].item() if not torch.isnan(loss_dict['tversky']) else 0.0
            
            prob = F.softmax(outputs, dim=1)[:, 1]
            preds = (prob > config['pred_threshold']).long()
            
            mask = (labels != -100).float()
            valid_pixels = mask.sum()
            
            if valid_pixels > EPS:
                tp, tn, fp, fn = batch_metrics(preds, labels, mask, device=device)
                total_tp += tp
                total_tn += tn
                total_fp += fp
                total_fn += fn
                
                slide_iou, mean_iou = calculate_iou(preds, labels, mask, device=device)
                total_slide_iou += slide_iou
                total_mean_iou += mean_iou
                iou_count += 1
            
            progress_bar.set_postfix({"Total Loss": f"{loss_dict['total'].item():.4f}"})
            data_load_time += (datetime.now() - batch_start).total_seconds()
        
        epoch_time = (datetime.now() - epoch_start).total_seconds()
        avg_total = running_total / len(train_loader) if len(train_loader) > 0 else 0.0
        avg_dice = running_dice / len(train_loader) if len(train_loader) > 0 else 0.0
        avg_boundary = running_boundary / len(train_loader) if len(train_loader) > 0 else 0.0
        avg_focal = running_focal / len(train_loader) if len(train_loader) > 0 else 0.0
        avg_tversky = running_tversky / len(train_loader) if len(train_loader) > 0 else 0.0
        
        precision, recall, f1 = aggregate_metrics(total_tp, total_tn, total_fp, total_fn)
        train_slide_iou = total_slide_iou / iou_count if iou_count > 0 else 0.0
        train_mean_iou = total_mean_iou / iou_count if iou_count > 0 else 0.0
        
        logger.info(f"\n[Epoch {epoch}/{config['epochs']}] (Total time: {epoch_time:.2f}s)")
        logger.info(f"  Time distribution: Data loading={data_load_time:.2f}s, Computation={compute_time:.2f}s")
        logger.info(f"  Training loss details:")
        logger.info(f"    Total loss: {avg_total:.4f} | Dice loss: {avg_dice:.4f}")
        logger.info(f"    Boundary loss: {avg_boundary:.4f} | Focal loss: {avg_focal:.4f} | Tversky loss: {avg_tversky:.4f}")
        logger.info(f"  Landslide extraction training metrics:")
        logger.info(f"    Precision: {precision:.4f} | Recall: {recall:.4f} | F1 score: {f1:.4f}")
        logger.info(f"    Slide IoU: {train_slide_iou:.4f} | Mean IoU: {train_mean_iou:.4f}")
        
        if epoch % config['val_interval'] == 0:
            val_result_str, val_metrics = evaluate_model(
                model, val_loader, criterion, device, feature_types, feature_info, full_dataset, 
                phase='Validation', threshold=config['pred_threshold']
            )
            logger.info(val_result_str)
            
            scheduler.step(val_metrics['slide_iou'])
            
            if val_metrics['slide_iou'] > best_slide_iou:
                best_slide_iou = val_metrics['slide_iou']
                best_val_metrics = val_metrics
                best_epoch = epoch
                no_improvement_counter = 0
                
                best_model_path = os.path.join(config['checkpoint_dir'], f"best_model_epoch_{epoch}.pth")
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"Saved best model to {best_model_path} (Slide IoU: {best_slide_iou:.4f})")
            else:
                no_improvement_counter += 1
                logger.info(f"No improvement in Slide IoU, count: {no_improvement_counter}/{early_stop_patience}")
                if no_improvement_counter >= early_stop_patience:
                    logger.info(f"No improvement for {early_stop_patience} consecutive epochs, triggering early stopping")
                    break
        
        if epoch % config['save_interval'] == 0:
            model_path = os.path.join(config['checkpoint_dir'], f"model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), model_path)
            logger.info(f"Periodically saved model to {model_path}")
    
    logger.info("\n===== Training Completed =====")
    logger.info(f"Best model at epoch {best_epoch} with Slide IoU: {best_slide_iou:.4f}")
    
    if not os.path.exists(best_model_path):
        logger.warning(f"Best model file not found, using initial model instead: {best_model_path}")
        best_model_path = initial_model_path
    
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    
    test_result_str, test_metrics = evaluate_model(
        model, test_loader, criterion, device, feature_types, feature_info, full_dataset, 
        phase='Test', threshold=config['pred_threshold']
    )
    logger.info("\n" + "="*50)
    logger.info("Final test set results:")
    logger.info(test_result_str)
    logger.info("="*50)
    
    save_test_predictions(
        model=model,
        test_loader=test_loader,
        test_sample_names=test_sample_names,
        config=config,
        device=device,
        feature_types=feature_types,
        feature_info=feature_info,
        full_dataset=full_dataset
    )
    
    final_model_path = os.path.join(config['checkpoint_dir'], "best_model_final.pth")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final best model saved to {final_model_path}")
    
    return test_metrics

def main(config_path):
    """Main function"""
    config = parse_config(config_path)
    
    logger = setup_logger(config['log_dir'])
    logger.info("===== Landslide Change Detection Model Training =====")
    logger.info("Final configuration parameters:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    setup_random_seeds(config['seed'])
    
    device, device_count = setup_device()
    multiprocessing.set_start_method('spawn', force=True)
    
    train_loader, val_loader, test_loader, feature_info, feature_types, full_dataset = create_dataloaders(config)
    test_sample_names = test_loader.dataset.sample_names
    
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
    logger.info(f"Model initialization completed - Fixed branch channels: {fixed_in_channels}, Number of dynamic branches: {len(dynamic_branch_channels)}")
    
    test_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        test_sample_names=test_sample_names,
        config=config,
        device=device,
        feature_types=feature_types,
        feature_info=feature_info,
        full_dataset=full_dataset,
        logger=logger
    )
    
    return test_metrics

if __name__ == '__main__':
    try:
        if len(sys.argv) < 2:
            raise RuntimeError("Please provide the configuration file path as an argument")
        config_path = sys.argv[1]
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        main(config_path)
        print('<training_status>0</training_status>')
        print('<training_log>Training successful</training_log>')
    except Exception as e:
        error_msg = str(e).replace('\n', ' ').replace('\t', ' ')
        print('<training_status>1</training_status>')
        print(f'<training_log>{error_msg}</training_log>')
        logger = logging.getLogger('landslide_training')
        if not logger.hasHandlers():
            logger.addHandler(logging.StreamHandler())
        logger.error(f"Training failed: {error_msg}")
      
