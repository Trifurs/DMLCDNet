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

# 导入自定义模块
from dataset.landslide_dataset import LandslideDataset
from loss.loss import LandslideLoss
from utils import *

# 忽略不必要的警告
warnings.filterwarnings("ignore", category=UserWarning)

def parse_xml_config(config_path):
    """解析XML配置文件"""
    tree = ET.parse(config_path)
    root = tree.getroot()
    
    config = {}
    for param in root.findall('param'):
        name = param.find('name').text
        value = param.find('value').text
        
        # 根据类型转换值
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
    """设置日志记录器"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logger = logging.getLogger('landslide_prediction')
    logger.setLevel(logging.INFO)
    
    # console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def setup_device():
    """设置计算设备，增加CUDA配置以避免地址对齐问题"""
    # 尝试解决CUDA地址对齐问题的配置
    if torch.cuda.is_available():
        # 禁用cuDNN对于可能引起对齐问题的操作
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        # 检查CUDA版本和设备兼容性
        cuda_version = torch.version.cuda
        device = torch.device('cuda')
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        logging.info(f"使用GPU进行预测: {device_name}，设备数量: {device_count}，CUDA版本: {cuda_version}")
        
        # 对于某些GPU架构，禁用Tensor Core可能解决对齐问题
        try:
            major, minor = torch.cuda.get_device_capability(device)
            logging.info(f"GPU计算能力: {major}.{minor}")
            if (major == 7 and minor <= 5) or major < 7:
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                logging.info("禁用TF32以提高兼容性")
        except Exception as e:
            logging.warning(f"无法检测GPU计算能力: {str(e)}")
    else:
        device = torch.device('cpu')
        device_count = 1
        logging.info("使用CPU进行预测")
    return device, device_count

def setup_random_seeds(seed=42):
    """设置随机种子以保证可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def dynamic_import_model(model_path, model_name):
    """动态导入模型类"""
    spec = importlib.util.spec_from_file_location(model_name, model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, model_name)

def count_parameters(model):
    """计算模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def move_to_device(obj, device):
    """
    递归地将对象或对象列表移动到指定设备
    处理张量、列表和元组类型
    """
    if isinstance(obj, torch.Tensor):
        # 确保张量内存对齐
        if device.type == 'cuda' and obj.numel() > 0:
            return obj.contiguous().to(device)
        return obj.to(device)
    elif isinstance(obj, list):
        return [move_to_device(item, device) for item in obj]
    elif isinstance(tuple(obj), tuple):
        return tuple(move_to_device(item, device) for item in obj)
    else:
        return obj  # 其他类型不处理

def calculate_layer_flops(layer, input_data):
    """计算单个层的FLOPs"""
    flops = 0
    
    # 卷积层
    if isinstance(layer, torch.nn.Conv2d):
        # 获取输入输出形状
        in_channels = input_data.shape[1]
        out_channels = layer.out_channels
        kernel_size = layer.kernel_size[0]
        output_shape = layer(input_data).shape[2:]
        
        # 计算FLOPs: 输出元素数 × 每个元素的计算量
        elements = output_shape[0] * output_shape[1]
        flops_per_element = in_channels * kernel_size * kernel_size
        flops = out_channels * elements * flops_per_element
        
    # 深度可分离卷积
    elif hasattr(layer, 'depthwise') and hasattr(layer, 'pointwise'):
        # 深度卷积部分
        depthwise = layer.depthwise
        in_channels = input_data.shape[1]
        kernel_size = depthwise.kernel_size[0]
        output_shape = depthwise(input_data).shape[2:]
        
        elements = output_shape[0] * output_shape[1]
        depthwise_flops = in_channels * elements * kernel_size * kernel_size
        
        # 逐点卷积部分
        pointwise = layer.pointwise
        out_channels = pointwise.out_channels
        pointwise_flops = out_channels * elements * in_channels
        
        flops = depthwise_flops + pointwise_flops
        
    # 全连接层
    elif isinstance(layer, torch.nn.Linear):
        in_features = input_data.view(-1).shape[0]
        out_features = layer.out_features
        flops = in_features * out_features
        
    # BatchNorm层
    elif isinstance(layer, torch.nn.BatchNorm2d):
        # 简化计算：每个元素参与均值和方差计算
        elements = input_data.numel()
        flops = 2 * elements  # 均值和方差各一次操作
        
    # 激活函数层
    elif isinstance(layer, (torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.Sigmoid, torch.nn.Softmax)):
        # 每个元素一次操作
        flops = input_data.numel()
        
    # 池化层
    elif isinstance(layer, (torch.nn.MaxPool2d, torch.nn.AvgPool2d)):
        # 每个输出元素需要对 kernel_size^2 个元素操作
        output = layer(input_data)
        kernel_size = layer.kernel_size if isinstance(layer.kernel_size, int) else layer.kernel_size[0]
        flops = output.numel() * (kernel_size ** 2)
        
    return flops

def calculate_model_flops(model, input_shape, device):
    """递归计算整个模型的FLOPs"""
    model = model.to(device)
    model.eval()
    
    # 准备输入数据
    input_data = move_to_device(input_shape, device)
    
    # 存储每层的FLOPs
    layer_flops = {}
    total_flops = 0
    
    # 注册钩子函数来计算FLOPs
    def hook_fn(module, input, output, name):
        nonlocal total_flops
        try:
            # 处理可能的多输入情况
            input_tensor = input[0] if isinstance(input, tuple) else input
            flops = calculate_layer_flops(module, input_tensor)
            layer_flops[name] = flops
            total_flops += flops
        except Exception as e:
            logging.debug(f"计算层 {name} 的FLOPs失败: {str(e)}")
    
    # 为每个层注册钩子
    hooks = []
    for name, module in model.named_modules():
        if name:  # 跳过根模块
            hook = module.register_forward_hook(
                lambda m, i, o, n=name: hook_fn(m, i, o, n)
            )
            hooks.append(hook)
    
    # 执行一次前向传播
    with torch.no_grad():
        model(*input_data)
    
    # 移除钩子
    for hook in hooks:
        hook.remove()
    
    return total_flops, layer_flops

def calculate_model_complexity(model, input_shape, device):
    """精确计算模型复杂度 (FLOPs)"""
    try:
        # 计算精确FLOPs
        total_flops, layer_flops = calculate_model_flops(model, input_shape, device)
        
        # 记录主要层的FLOPs（超过总FLOPs的1%）
        significant_layers = {k: v for k, v in layer_flops.items() if v > total_flops * 0.01}
        if significant_layers:
            logging.debug("主要层FLOPs分布:")
            for name, flops in sorted(significant_layers.items(), key=lambda x: x[1], reverse=True):
                logging.debug(f"  {name}: {flops/1e6:.2f} MFLOPs ({flops/total_flops*100:.1f}%)")
        
        # 计算推理时间
        model = model.to(device)
        model.eval()
        input_on_device = move_to_device(input_shape, device)
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):  # 更多迭代以获得更稳定的时间测量
                model(*input_on_device)
        # 同步设备以确保准确计时
        if device.type == 'cuda':
            torch.cuda.synchronize()
        avg_time = (time.time() - start_time) / 100
        
        return total_flops, avg_time
    except Exception as e:
        logging.error(f"模型复杂度计算过程出错: {str(e)}")
        raise

def create_prediction_dataloader(config):
    """创建用于预测的数据集和数据加载器（不使用NPZ缓存）"""
    dataset = LandslideDataset(
        root_dir=config['data_root'],
        mode='full',  # 加载所有数据
        exclude_features=config['exclude_features'],
        seed=config.get('seed', 42),
        use_data_cache=False,  # 禁用缓存
        preprocess_npz=False,  # 不使用NPZ预处理
        cache_dir=None  # 不设置缓存目录
    )
    
    # 获取特征信息和类型
    feature_info = dataset.get_feature_info()
    feature_types = dataset.feature_types
    
    # 创建数据加载器，设置pin_memory_mode以确保内存对齐
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
    """生成可视化预测结果"""
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    preds_binary = (preds > threshold).astype(np.uint8)
    h, w = preds_binary.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 真阳性 (白色)
    tp_mask = (preds_binary == 1) & (labels == 1)
    vis[tp_mask] = [255, 255, 255]
    
    # 假阳性 (蓝色)
    fp_mask = (preds_binary == 1) & (labels == 0)
    vis[fp_mask] = [52, 172, 254]
    
    # 假阴性 (红色)
    fn_mask = (preds_binary == 0) & (labels == 1)
    vis[fn_mask] = [255, 0, 0]
    
    return vis

def load_parallel_model_weights(model, weights_path, logger):
    """
    加载使用nn.DataParallel训练的模型权重，处理"module."前缀问题
    """
    # 加载保存的状态字典
    saved_state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    
    # 创建新的状态字典，移除"module."前缀
    new_state_dict = {}
    for key, value in saved_state_dict.items():
        # 移除"module."前缀
        if key.startswith('module.'):
            new_key = key[7:]  # 从第7个字符开始取，跳过"module."
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    # 检查模型与状态字典的兼容性
    model_state_dict = model.state_dict()
    compatible_state_dict = {}
    missing_keys = []
    unexpected_keys = []
    
    for key, value in new_state_dict.items():
        if key in model_state_dict:
            # 检查张量形状是否匹配
            if model_state_dict[key].shape == value.shape:
                compatible_state_dict[key] = value
            else:
                logger.warning(f"跳过不匹配的参数 {key}: 模型需要 {model_state_dict[key].shape}, 权重提供 {value.shape}")
                missing_keys.append(key)
        else:
            unexpected_keys.append(key)
    
    # 检查模型是否有缺失的键
    for key in model_state_dict:
        if key not in compatible_state_dict:
            missing_keys.append(key)
    
    # 记录检查结果
    if missing_keys:
        logger.warning(f"检测到 {len(missing_keys)} 个缺失的参数键，将使用默认初始化值")
    if unexpected_keys:
        logger.warning(f"检测到 {len(unexpected_keys)} 个意外的参数键，将被忽略")
    
    # 加载兼容的参数
    model.load_state_dict(compatible_state_dict, strict=False)
    return model

def predict(model, dataloader, config, device, feature_types, feature_info, dataset, logger):
    """使用模型进行预测并保存结果"""
    os.makedirs(config['output_dir'], exist_ok=True)
    logger.info(f"预测结果将保存到: {config['output_dir']}")
    
    # 多GPU支持 - 尝试禁用多GPU以解决对齐问题
    device_count = torch.cuda.device_count()
    use_multi_gpu = device_count > 1 and config.get('use_multi_gpu', False)
    
    if use_multi_gpu:
        logger.info(f"使用 {device_count} 个GPU进行预测")
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.eval()
    
    # 记录总预测时间
    total_start_time = time.time()
    total_samples = len(dataset)
    total_processing_time = 0
    
    # 存储所有样本的指标
    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0
    
    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(dataloader):
            start_idx = batch_idx * dataloader.batch_size
            end_idx = start_idx + len(features)
            batch_sample_names = dataset.sample_names[start_idx:end_idx]
            
            logger.info(f"处理批次 {batch_idx + 1}/{len(dataloader)}，样本: {batch_sample_names[0]} 至 {batch_sample_names[-1]}")
            
            # 记录批次处理时间
            batch_start_time = time.time()
            
            # 确保输入数据内存对齐
            features = features.contiguous().to(device, non_blocking=True)
            labels = labels.contiguous().to(device, non_blocking=True)
            
            # 分割特征
            x_before, x_after, dynamic_inputs = split_features(features, feature_types, feature_info)
            
            # 模型预测（计时）
            inference_start = time.time()
            try:
                outputs = model(x_before, x_after, dynamic_inputs)
            except RuntimeError as e:
                if "misaligned address" in str(e):
                    logger.error("检测到地址对齐错误，尝试单GPU模式...")
                    # 切换到单GPU模式重试
                    model = model.module if isinstance(model, torch.nn.DataParallel) else model
                    outputs = model(x_before, x_after, dynamic_inputs)
                    use_multi_gpu = False
                else:
                    raise
            
            # 同步设备以确保准确计时
            if device.type == 'cuda':
                torch.cuda.synchronize()
            inference_time = time.time() - inference_start
            total_processing_time += inference_time
            
            prob = F.softmax(outputs, dim=1)[:, 1]
            preds = (prob > config['pred_threshold']).long()
            mask = (labels != -100).float()
            
            # 处理每个样本
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
                
                # 生成并保存可视化结果
                vis = generate_visualization(
                    pred.cpu().numpy(), 
                    label.cpu().numpy(),
                    config['pred_threshold']
                )
                
                output_path = os.path.join(config['output_dir'], f"{sample_name}.png")
                Image.fromarray(vis).save(output_path)
            
            # 计算批次FPS
            batch_time = time.time() - batch_start_time
            batch_fps = len(features) / batch_time
            logger.info(f"批次 {batch_idx + 1} 处理完成，耗时: {batch_time:.4f}秒，FPS: {batch_fps:.2f}")
    
    # 计算总体指标
    total_time = time.time() - total_start_time
    overall_fps = total_samples / total_time
    inference_fps = total_samples / total_processing_time
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn) if (total_tp + total_tn + total_fp + total_fn) > 0 else 0
    
    # 记录性能指标
    logger.info("\n===== 预测性能指标 =====")
    logger.info(f"总样本数: {total_samples}")
    logger.info(f"总处理时间: {total_time:.4f}秒")
    logger.info(f"总推理时间: {total_processing_time:.4f}秒")
    logger.info(f"总体FPS (含IO): {overall_fps:.2f}")
    logger.info(f"推理FPS (仅计算): {inference_fps:.2f}")
    logger.info(f"平均样本处理时间: {total_time/total_samples:.4f}秒")
    
    # 记录精度指标
    logger.info("\n===== 预测精度指标 =====")
    logger.info(f"精确率 (Precision): {precision:.4f}")
    logger.info(f"召回率 (Recall): {recall:.4f}")
    logger.info(f"F1分数: {f1:.4f}")
    logger.info(f"准确率 (Accuracy): {accuracy:.4f}")
    logger.info(f"真阳性数量: {total_tp}")
    logger.info(f"真阴性数量: {total_tn}")
    logger.info(f"假阳性数量: {total_fp}")
    logger.info(f"假阴性数量: {total_fn}")
    
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
    """主函数"""
    # 解析配置文件
    config = parse_xml_config(config_path)
    
    # 设置日志
    logger = setup_logger(config['log_dir'])
    logger.info("===== 滑坡预测程序启动 =====")
    logger.info("配置参数:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # 初始化随机种子
    setup_random_seeds(config.get('seed', 42))
    
    # 设置设备
    device, device_count = setup_device()
    multiprocessing.set_start_method('spawn', force=True)
    
    # 创建数据加载器
    dataloader, feature_info, feature_types, dataset = create_prediction_dataloader(config)
    logger.info(f"数据集加载完成，共 {len(dataset)} 个样本")
    
    # 获取输入形状用于复杂度计算
    sample_features, _ = dataset[0]
    sample_features = sample_features.unsqueeze(0)  # 增加批次维度
    x_before, x_after, dynamic_inputs = split_features(sample_features, feature_types, feature_info)
    input_shape = (x_before, x_after, dynamic_inputs)
    
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
    
    # 计算模型参数量
    total_params, trainable_params = count_parameters(model)
    logger.info(f"模型参数量: 总参数={total_params:,} ({total_params/1e6:.2f}M), 可训练参数={trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # 估算模型复杂度
    try:
        flops, avg_inference_time = calculate_model_complexity(model, input_shape, device)
        # 转换为更易读的单位
        if flops >= 1e9:
            flops_str = f"{flops/1e9:.2f} GFLOPs"
        elif flops >= 1e6:
            flops_str = f"{flops/1e6:.2f} MFLOPs"
        else:
            flops_str = f"{flops/1e3:.2f} KFLOPs"
            
        logger.info(f"模型复杂度: {flops_str}, 单次推理时间: {avg_inference_time:.6f}秒")
    except Exception as e:
        logger.warning(f"模型复杂度计算失败: {str(e)}")
        logger.info("继续执行预测流程...")
    
    logger.info(f"模型初始化完成 - 固定分支通道数: {fixed_in_channels}, 动态分支数: {len(dynamic_branch_channels)}")
    
    # 加载训练好的权重（处理DataParallel的情况）
    if not os.path.exists(config['model_weights']):
        logger.error(f"模型权重文件不存在: {config['model_weights']}")
        raise FileNotFoundError(f"模型权重文件不存在: {config['model_weights']}")
    
    # 使用专门的函数加载并行训练的权重
    model = load_parallel_model_weights(model, config['model_weights'], logger)
    logger.info(f"成功加载模型权重: {config['model_weights']}")
    
    # 进行预测
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
    
    logger.info("===== 预测程序完成 =====")
    return metrics

if __name__ == '__main__':
    try:
        if len(sys.argv) < 2:
            raise RuntimeError("请提供配置文件路径作为参数")
        config_path = sys.argv[1]
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        main(config_path)
        print('<prediction_status>0</prediction_status>')
        print('<prediction_log>预测成功</prediction_log>')
    except Exception as e:
        error_msg = str(e).replace('\n', ' ').replace('\t', ' ')
        print('<prediction_status>1</prediction_status>')
        print(f'<prediction_log>{error_msg}</prediction_log>')
        
        logger = logging.getLogger('landslide_prediction')
        if not logger.hasHandlers():
            logger.addHandler(logging.StreamHandler())
        logger.error(f"预测失败: {error_msg}")
        sys.exit(1)

    