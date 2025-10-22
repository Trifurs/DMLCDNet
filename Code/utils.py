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
import random  # 导入random模块用于Python原生随机种子控制
import importlib.util
from typing import Type
from PIL import Image  # 用于保存PNG预测图

# 导入自定义模块
from dataset.landslide_dataset import LandslideDataset


# 数值稳定性常量
EPS = 1e-8

# 用于DataLoader的worker初始化函数
def worker_init_fn(worker_id):
    """为每个worker进程设置独立的随机种子"""
    base_seed = getattr(worker_init_fn, 'base_seed', 114514)
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)

def setup_device():
    """设置训练设备（GPU优先）"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_count = torch.cuda.device_count()
        logging.info(f"使用GPU训练，设备数量: {device_count}")
    else:
        device = torch.device("cpu")
        device_count = 1
        logging.info("未检测到GPU，使用CPU训练")
    return device, device_count

def _parse_single_xml(xml_file):
    """解析单个XML文件，返回参数字典"""
    if not os.path.exists(xml_file):
        raise FileNotFoundError(f"配置文件不存在: {xml_file}")
    
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
    解析XML配置文件，支持基础配置继承
    xml_file: 当前配置文件路径
    parsed_files: 已解析的文件集合，用于检测循环引用
    """
    if parsed_files is None:
        parsed_files = set()
    
    # 检查循环引用
    xml_abs_path = os.path.abspath(xml_file)
    if xml_abs_path in parsed_files:
        raise RuntimeError(f"配置文件循环引用: {xml_abs_path}")
    parsed_files.add(xml_abs_path)
    
    # 解析当前配置文件
    current_config = _parse_single_xml(xml_file)
    
    # 检查是否存在基础配置参数
    base_param = current_config.pop('base_param', None)
    if base_param:
        # 处理基础配置文件路径（相对当前配置文件的路径）
        current_dir = os.path.dirname(xml_abs_path)
        base_xml_path = os.path.join(current_dir, base_param)
        
        # 递归解析基础配置
        base_config = parse_config(base_xml_path, parsed_files)
        
        # 用当前配置覆盖基础配置
        base_config.update(current_config)
        return base_config
    else:
        # 添加默认值（含新增的预测图保存目录默认值）
        current_config.setdefault('preprocess_npz', True)
        current_config.setdefault('cache_dir', os.path.join(current_config.get('data_root', ''), 'npz_cache'))
        current_config.setdefault('cache_max_size', 50)
        current_config.setdefault('lr', 1e-4)
        current_config.setdefault('weight_decay', 1e-5)
        current_config.setdefault('pred_threshold', 0.5)
        current_config.setdefault('test_pred_save_dir', os.path.join(current_config.get('checkpoint_dir', ''), 'test_predictions'))
        return current_config

def setup_logger(log_dir):
    """初始化日志系统"""
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
    """初始化所有随机种子，确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logging.info(f"已初始化所有随机种子，种子值: {seed}")
    worker_init_fn.base_seed = seed

def create_dataloaders(config):
    """创建数据加载器（调整为匹配LandslideDataset的sample_names属性）"""
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
    
    # 数据校验
    sample_features, sample_labels = full_dataset[0]
    valid_label_mask = (sample_labels != -100)
    if valid_label_mask.sum() == 0:
        logging.warning("警告：数据集中存在无有效标签的样本，可能导致训练异常")
    if (sample_labels[valid_label_mask] == 1).sum() == 0:
        logging.warning("警告：数据集中未发现滑坡标签（类别1），可能导致指标计算异常")
    
    feature_info = full_dataset.get_feature_info()
    feature_types = full_dataset.feature_types
    logger = logging.getLogger('landslide_training')
    logger.info("特征信息:")
    for feature, bands in feature_info.items():
        logger.info(f"  {feature}: {bands} 波段")
    
    # 验证灾前灾后特征
    assert 'before' in feature_info and 'after' in feature_info, "缺少灾前/灾后特征"
    assert feature_info['before'] == feature_info['after'], "灾前灾后波段数不匹配"
    
    # 划分数据集
    total_samples = len(full_dataset)
    train_size = int(config['train_ratio'] * total_samples)
    val_size = int(config['val_ratio'] * total_samples)
    test_size = total_samples - train_size - val_size
    
    generator = torch.Generator().manual_seed(config['seed'])
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=generator
    )
    
    # 关键修改：从full_dataset的sample_names中提取对应子集（匹配划分后的索引）
    train_dataset.sample_names = [full_dataset.sample_names[idx] for idx in train_dataset.indices]
    val_dataset.sample_names = [full_dataset.sample_names[idx] for idx in val_dataset.indices]
    test_dataset.sample_names = [full_dataset.sample_names[idx] for idx in test_dataset.indices]
    
    # 设置数据集模式
    train_dataset.dataset.mode = 'train'
    val_dataset.dataset.mode = 'val'
    test_dataset.dataset.mode = 'test'
    
    logger.info("\n数据集划分:")
    logger.info(f"  训练集: {len(train_dataset)} 样本")
    logger.info(f"  验证集: {len(val_dataset)} 样本")
    logger.info(f"  测试集: {len(test_dataset)} 样本")
    
    # 优化DataLoader参数
    dataloader_kwargs = {
        'batch_size': config['batch_size'],
        'pin_memory': True,
        'persistent_workers': True,
        'num_workers': min(config['num_workers'], multiprocessing.cpu_count() // 2),
        'worker_init_fn': worker_init_fn
    }
    
    # 创建数据加载器
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
    """动态导入模型"""
    # 处理模型路径
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 将模型路径转换为模块名称
    module_name = os.path.splitext(os.path.basename(model_path))[0]
    
    # 构建模块规范
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    if spec is None:
        raise ImportError(f"无法从路径 {model_path} 创建模块规范")
    
    # 加载模块
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    # 获取模型类
    if not hasattr(module, model_name):
        raise AttributeError(f"模块 {model_path} 中没有找到模型类 {model_name}")
    
    return getattr(module, model_name)

def calculate_iou(pred, target, mask, num_classes=2, device=None):
    """计算IoU"""
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
    """计算批次指标"""
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
    """聚合指标"""
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
    """分离特征"""
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
    
    assert x_before is not None and x_after is not None, "特征分离失败"
    return x_before, x_after, dynamic_inputs

def evaluate_model(model, data_loader, criterion, device, feature_types, feature_info, full_dataset, 
                   phase='Validation', threshold=0.5):
    """评估模型"""
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
            
            # 损失计算
            loss_dict = criterion(outputs, labels)
            
            if torch.isnan(loss_dict['total']):
                logging.warning(f"检测到NaN损失！批次索引: {batch_idx}，跳过该批次")
                continue
            
            running_loss += loss_dict['total'].item()
            running_dice += loss_dict['dice'].item() if not torch.isnan(loss_dict['dice']) else 0.0
            running_boundary += loss_dict['boundary'].item() if not torch.isnan(loss_dict['boundary']) else 0.0
            running_focal += loss_dict['focal'].item() if not torch.isnan(loss_dict['focal']) else 0.0
            running_tversky += loss_dict['tversky'].item() if not torch.isnan(loss_dict['tversky']) else 0.0
            
            # 使用阈值调整预测结果
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
    result_str = (f"\n[{phase} Results] (耗时: {eval_time:.2f}秒, 阈值: {threshold})\n"
                  f"损失详情:\n"
                  f"  总损失: {avg_loss:.4f}\n"
                  f"  Dice损失: {avg_dice:.4f}\n"
                  f"  边界损失: {avg_boundary:.4f}\n"
                  f"  Focal损失: {avg_focal:.4f}\n"
                  f"  Tversky损失: {avg_tversky:.4f}\n"
                  f"滑坡提取指标:\n"
                  f"  精确率: {precision:.4f} | 召回率: {recall:.4f} | F1分数: {f1:.4f}\n"
                  f"  滑坡IoU: {slide_iou:.4f} | 平均IoU: {mean_iou:.4f}")
    
    return result_str, {
        'loss': avg_loss, 'dice': avg_dice, 'boundary': avg_boundary,
        'focal': avg_focal, 'tversky': avg_tversky,
        'precision': precision, 'recall': recall, 'f1': f1,
        'slide_iou': slide_iou, 'mean_iou': mean_iou
    }

def save_test_predictions(model, test_loader, test_sample_names, config, device, feature_types, feature_info, full_dataset):
    """
    生成并保存测试集预测图
    输入：
        - model: 训练好的最佳模型
        - test_loader: 测试集数据加载器
        - test_sample_names: 测试集样本名称列表（与预测图一一对应）
        - config: 配置字典（含保存路径、预测阈值）
        - device: 计算设备
        - feature_types/feature_info: 特征相关信息
        - full_dataset: 完整数据集（用于清理缓存）
    输出：
        - 保存PNG格式预测图到 config['test_pred_save_dir']
    """
    # 创建保存目录
    pred_save_dir = config['test_pred_save_dir']
    os.makedirs(pred_save_dir, exist_ok=True)
    logger = logging.getLogger('landslide_training')
    logger.info(f"\n开始生成测试集预测图，保存路径: {pred_save_dir}")
    
    full_dataset.clear_cache()
    model.eval()
    pred_idx = 0  # 用于匹配测试集样本名称
    
    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(tqdm(test_loader, desc="Generating Test Predictions")):
            features = features.to(device)
            x_before, x_after, dynamic_inputs = split_features(features, feature_types, feature_info)
            outputs = model(x_before, x_after, dynamic_inputs)
            
            # 计算预测结果（0=非滑坡，1=滑坡）
            prob = F.softmax(outputs, dim=1)[:, 1]  # 正类（滑坡）概率
            preds = (prob > config['pred_threshold']).long()  # 应用阈值
            
            # 遍历批次内每个样本，保存为PNG（0→0，1→255）
            for pred in preds:
                # 获取当前样本名称（去掉扩展名，统一用PNG）
                sample_name = test_sample_names[pred_idx]
                pred_save_path = os.path.join(pred_save_dir, f"{sample_name}_pred.png")
                
                # 转换为0-255格式
                pred_255 = (pred.cpu().numpy() * 255).astype(np.uint8)
                
                # 保存PNG
                pred_img = Image.fromarray(pred_255)
                pred_img.save(pred_save_path)
                
                pred_idx += 1
    
    logger.info(f"测试集预测图生成完成！共生成 {pred_idx} 张预测图")
    