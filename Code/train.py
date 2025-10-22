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

# 导入自定义模块
from dataset.landslide_dataset import LandslideDataset
from loss.loss import LandslideLoss
from utils import *  # 包含新增的 save_test_predictions 函数

# 忽略不必要的警告
warnings.filterwarnings("ignore", category=UserWarning)

def train_model(model, train_loader, val_loader, test_loader, test_sample_names, config, device, 
                feature_types, feature_info, full_dataset, logger):
    """训练模型（新增 test_sample_names 参数，用于后续生成预测图）"""
    device_count = torch.cuda.device_count()
    if device_count > 1:
        logger.info(f"使用 {device_count} 个GPU进行训练")
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    
    # 初始化损失函数
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
    
    # 恢复训练支持
    start_epoch = 1
    best_model_path = None
    if config['resume'] and os.path.exists(config['resume']):
        model.load_state_dict(torch.load(config['resume'], map_location=device))
        logger.info(f"从 checkpoint 恢复训练: {config['resume']}")
        start_epoch = int(os.path.basename(config['resume']).split('_')[-1].split('.')[0]) + 1
        best_model_path = config['resume']
    
    # 训练参数初始化
    best_slide_iou = -1e9
    best_val_metrics = None
    no_improvement_counter = 0
    best_epoch = start_epoch
    early_stop_patience = config.get('early_stop_patience', 10)
    
    # 保存初始模型
    initial_model_path = os.path.join(config['checkpoint_dir'], f"initial_model_epoch_{start_epoch}.pth")
    torch.save(model.state_dict(), initial_model_path)
    logger.info(f"保存初始模型到 {initial_model_path}")
    if best_model_path is None:
        best_model_path = initial_model_path
    
    logger.info("\n===== 开始训练 =====")
    logger.info(f"学习率: {config['lr']}, 早停耐心系数: {early_stop_patience}, 预测阈值: {config['pred_threshold']}")
    
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
                logger.warning(f"Epoch {epoch} 批次 {batch_idx} 出现NaN损失，跳过反向传播")
                compute_time += (datetime.now() - compute_start).total_seconds()
                data_load_time += (datetime.now() - batch_start).total_seconds()
                continue
            
            total_loss = loss_dict['total']
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            compute_time += (datetime.now() - compute_start).total_seconds()
            
            # 累加损失
            running_total += loss_dict['total'].item()
            running_dice += loss_dict['dice'].item() if not torch.isnan(loss_dict['dice']) else 0.0
            running_boundary += loss_dict['boundary'].item() if not torch.isnan(loss_dict['boundary']) else 0.0
            running_focal += loss_dict['focal'].item() if not torch.isnan(loss_dict['focal']) else 0.0
            running_tversky += loss_dict['tversky'].item() if not torch.isnan(loss_dict['tversky']) else 0.0
            
            # 训练阶段也使用阈值调整预测
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
            
            progress_bar.set_postfix({"总损失": f"{loss_dict['total'].item():.4f}"})
            data_load_time += (datetime.now() - batch_start).total_seconds()
        
        # 计算 epoch 指标
        epoch_time = (datetime.now() - epoch_start).total_seconds()
        avg_total = running_total / len(train_loader) if len(train_loader) > 0 else 0.0
        avg_dice = running_dice / len(train_loader) if len(train_loader) > 0 else 0.0
        avg_boundary = running_boundary / len(train_loader) if len(train_loader) > 0 else 0.0
        avg_focal = running_focal / len(train_loader) if len(train_loader) > 0 else 0.0
        avg_tversky = running_tversky / len(train_loader) if len(train_loader) > 0 else 0.0
        
        precision, recall, f1 = aggregate_metrics(total_tp, total_tn, total_fp, total_fn)
        train_slide_iou = total_slide_iou / iou_count if iou_count > 0 else 0.0
        train_mean_iou = total_mean_iou / iou_count if iou_count > 0 else 0.0
        
        # 打印训练结果
        logger.info(f"\n[Epoch {epoch}/{config['epochs']}] (总耗时: {epoch_time:.2f}秒)")
        logger.info(f"  时间分布: 数据加载={data_load_time:.2f}秒, 计算={compute_time:.2f}秒")
        logger.info(f"  训练损失详情:")
        logger.info(f"    总损失: {avg_total:.4f} | Dice损失: {avg_dice:.4f}")
        logger.info(f"    边界损失: {avg_boundary:.4f} | Focal损失: {avg_focal:.4f} | Tversky损失: {avg_tversky:.4f}")
        logger.info(f"  滑坡提取训练指标:")
        logger.info(f"    精确率: {precision:.4f} | 召回率: {recall:.4f} | F1分数: {f1:.4f}")
        logger.info(f"    滑坡IoU: {train_slide_iou:.4f} | 平均IoU: {train_mean_iou:.4f}")
        
        # 验证
        if epoch % config['val_interval'] == 0:
            val_result_str, val_metrics = evaluate_model(
                model, val_loader, criterion, device, feature_types, feature_info, full_dataset, 
                phase='Validation', threshold=config['pred_threshold']
            )
            logger.info(val_result_str)
            
            # 学习率调整
            scheduler.step(val_metrics['slide_iou'])
            
            # 保存最佳模型
            if val_metrics['slide_iou'] > best_slide_iou:
                best_slide_iou = val_metrics['slide_iou']
                best_val_metrics = val_metrics
                best_epoch = epoch
                no_improvement_counter = 0
                
                best_model_path = os.path.join(config['checkpoint_dir'], f"best_model_epoch_{epoch}.pth")
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"保存最佳模型到 {best_model_path} (滑坡IoU: {best_slide_iou:.4f})")
            else:
                no_improvement_counter += 1
                logger.info(f"滑坡IoU无改进，计数: {no_improvement_counter}/{early_stop_patience}")
                if no_improvement_counter >= early_stop_patience:
                    logger.info(f"连续{early_stop_patience}个epoch无改进，触发早停")
                    break
        
        # 定期保存模型
        if epoch % config['save_interval'] == 0:
            model_path = os.path.join(config['checkpoint_dir'], f"model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), model_path)
            logger.info(f"定期保存模型到 {model_path}")
    
    # 训练结束，评估最佳模型
    logger.info("\n===== 训练结束 =====")
    logger.info(f"最佳模型在第 {best_epoch} 个epoch，滑坡IoU: {best_slide_iou:.4f}")
    
    if not os.path.exists(best_model_path):
        logger.warning(f"最佳模型文件不存在，使用初始模型替代: {best_model_path}")
        best_model_path = initial_model_path
    
    # 加载最佳模型
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    
    # 1. 评估测试集精度
    test_result_str, test_metrics = evaluate_model(
        model, test_loader, criterion, device, feature_types, feature_info, full_dataset, 
        phase='Test', threshold=config['pred_threshold']
    )
    logger.info("\n" + "="*50)
    logger.info("测试集最终结果:")
    logger.info(test_result_str)
    logger.info("="*50)
    
    # 2. 生成并保存测试集预测图（新增逻辑）
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
    
    # 保存最终模型
    final_model_path = os.path.join(config['checkpoint_dir'], "best_model_final.pth")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"最佳模型最终保存到 {final_model_path}")
    
    return test_metrics

def main(config_path):
    """主函数"""
    # 解析配置文件，支持基础配置继承
    config = parse_config(config_path)
    
    logger = setup_logger(config['log_dir'])
    logger.info("===== 滑坡变化检测模型训练 =====")
    logger.info("最终配置参数:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # 初始化全局随机种子
    setup_random_seeds(config['seed'])
    
    device, device_count = setup_device()
    multiprocessing.set_start_method('spawn', force=True)
    
    # 创建数据加载器（test_loader 含 sample_names 属性）
    train_loader, val_loader, test_loader, feature_info, feature_types, full_dataset = create_dataloaders(config)
    test_sample_names = test_loader.dataset.sample_names  # 获取测试集样本名称列表
    
    # 动态导入模型
    try:
        LandslideNet = dynamic_import_model(config['model_path'], config['model_name'])
        logger.info(f"成功导入模型: {config['model_name']} 来自 {config['model_path']}")
    except Exception as e:
        logger.error(f"模型导入失败: {str(e)}")
        raise
    
    # 准备模型参数
    dynamic_branch_channels = [
        feature_info[feature] for feature in feature_types if feature not in ['before', 'after']
    ]
    logger.info(f"动态特征分支通道数: {dynamic_branch_channels}")
    
    fixed_in_channels = feature_info['before']
    # 实例化模型
    model = LandslideNet(
        fixed_in_channels=fixed_in_channels,
        dynamic_in_channels=dynamic_branch_channels
    )
    logger.info(f"模型初始化完成 - 固定分支通道数: {fixed_in_channels}, 动态分支数: {len(dynamic_branch_channels)}")
    
    # 训练模型（传入测试集样本名称）
    test_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        test_sample_names=test_sample_names,  # 新增：传递测试集样本名称
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
            raise RuntimeError("请提供配置文件路径作为参数")
        config_path = sys.argv[1]
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        main(config_path)
        print('<training_status>0</training_status>')
        print('<training_log>训练成功</training_log>')
    except Exception as e:
        error_msg = str(e).replace('\n', ' ').replace('\t', ' ')
        print('<training_status>1</training_status>')
        print(f'<training_log>{error_msg}</training_log>')
        logger = logging.getLogger('landslide_training')
        if not logger.hasHandlers():
            logger.addHandler(logging.StreamHandler())
        logger.error(f"训练失败: {error_msg}")

    