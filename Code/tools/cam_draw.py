import sys
import os
import torch
import logging
import warnings
import numpy as np
import multiprocessing
import cv2
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict
from utils import *  

# 忽略不必要的警告
warnings.filterwarnings("ignore", category=UserWarning)

# 设置matplotlib参数
plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = ["Arial", "Helvetica", "sans-serif"]

# 需要可视化的特定层级
TARGET_LAYERS = {
    'decoder.conv_block1',
    'decoder.conv_block2',
    'decoder.conv_block3',
    'fixed_encoder.conv2',
    'fixed_encoder.conv3',
    'fixed_encoder.shared_conv',
    'fixed_encoder.change_detector',
    'fusion.attention'
}

# 特征可视化方式
VISUALIZATION_METHODS = {
    'first_three': "使用前三个波段",
    'max_variance': "使用方差最大的三个波段",
    'pca': "使用PCA降维到三个波段",
    'grayscale_mean': "所有波段的平均值（灰度图）",
    'grayscale_max': "所有波段的最大值（灰度图）"
}

# 默认可视化方式
DEFAULT_VISUALIZATION = 'pca'

# CAM可视化类
class CAMVisualizer:
    """模型类激活映射(CAM)可视化工具"""
    
    def __init__(self, model, feature_names: List[str], device: torch.device, 
                 vis_method: str = DEFAULT_VISUALIZATION):
        """初始化CAM可视化器"""
        # 处理DataParallel包装的模型
        self.model = self._get_unwrapped_model(model).eval()  # 设置为评估模式
        self.feature_names = feature_names
        self.device = device
        self.hooks = []  # 用于存储钩子
        self.feature_maps = {}  # 用于存储中间层特征图
        self.vis_method = vis_method
        
        # 验证可视化方法
        if self.vis_method not in VISUALIZATION_METHODS:
            raise ValueError(f"无效的可视化方法: {self.vis_method}，可选方法: {list(VISUALIZATION_METHODS.keys())}")
        
        # 注册钩子以获取中间层特征
        self._register_hooks()
    
    def _get_unwrapped_model(self, model):
        """获取被DataParallel或DistributedDataParallel包装的原始模型"""
        if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            return model.module
        return model
    
    def _register_hooks(self):
        """为模型关键层注册钩子，只关注目标层级"""
        # 固定编码器相关层
        if hasattr(self.model.fixed_encoder, 'shared_conv'):
            self.hooks.append(self.model.fixed_encoder.shared_conv.register_forward_hook(
                self._get_forward_hook('fixed_encoder.shared_conv')
            ))
        
        if hasattr(self.model.fixed_encoder, 'conv2'):
            self.hooks.append(self.model.fixed_encoder.conv2.register_forward_hook(
                self._get_forward_hook('fixed_encoder.conv2')
            ))
        
        if hasattr(self.model.fixed_encoder, 'conv3'):
            self.hooks.append(self.model.fixed_encoder.conv3.register_forward_hook(
                self._get_forward_hook('fixed_encoder.conv3')
            ))

        if hasattr(self.model.fixed_encoder, 'change_detector'):
            self.hooks.append(self.model.fixed_encoder.change_detector.register_forward_hook(
                self._get_forward_hook('fixed_encoder.change_detector')
            ))
        
        # 融合层相关
        if hasattr(self.model.fusion, 'attention'):
            self.hooks.append(self.model.fusion.attention.register_forward_hook(
                self._get_forward_hook('fusion.attention')
            ))
        
        # 解码器相关层
        if hasattr(self.model.decoder, 'conv_block1'):
            self.hooks.append(self.model.decoder.conv_block1.register_forward_hook(
                self._get_forward_hook('decoder.conv_block1')
            ))
        
        if hasattr(self.model.decoder, 'conv_block2'):
            self.hooks.append(self.model.decoder.conv_block2.register_forward_hook(
                self._get_forward_hook('decoder.conv_block2')
            ))
        
        if hasattr(self.model.decoder, 'conv_block3'):
            self.hooks.append(self.model.decoder.conv_block3.register_forward_hook(
                self._get_forward_hook('decoder.conv_block3')
            ))
    
    def _get_forward_hook(self, name: str):
        """创建一个前向钩子，用于保存特征图"""
        def hook(module, input, output):
            # 只保存目标层级的特征图
            if name in TARGET_LAYERS:
                self.feature_maps[name] = output.detach()
        return hook
    
    def _generate_cam(self, feature_map: torch.Tensor, weights: Optional[torch.Tensor] = None) -> np.ndarray:
        """生成类激活映射"""
        # 确保特征图是4D张量 (N, C, H, W)
        if len(feature_map.shape) != 4:
            # 尝试自动调整维度
            if len(feature_map.shape) == 3:  # 如果是3D (C, H, W)，添加批次维度
                feature_map = feature_map.unsqueeze(0)
            elif len(feature_map.shape) == 5:  # 如果是5D，尝试压缩空间维度
                feature_map = feature_map.squeeze(2)  # 假设中间维度是1
        
        if weights is None:
            # 简单加权平均（类激活映射的基本实现）
            weights = torch.ones(feature_map.size(1), device=self.device) / feature_map.size(1)
        
        # 应用权重并求和
        cam = torch.sum(weights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * feature_map, dim=1)
        cam = cam.squeeze(0).cpu().numpy()
        
        # ReLU激活，因为我们只关心正影响
        cam = np.maximum(cam, 0)
        
        # 归一化到0-1范围
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        
        return cam
    
    def _upsample_cam(self, cam: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """将CAM上采样到目标尺寸，确保输入是2D"""
        # 处理可能的1D或3D输入
        if len(cam.shape) == 1:
            # 如果是1D，扩展为2D
            cam = np.expand_dims(cam, axis=0)
        elif len(cam.shape) > 2:
            # 如果维度超过2D，取平均压缩
            cam = np.mean(cam, axis=0)
        
        # 确保目标尺寸是正确的2D形状
        if len(target_size) != 2:
            raise ValueError(f"目标尺寸必须是2D元组，得到 {target_size}")
            
        return cv2.resize(cam, target_size, interpolation=cv2.INTER_LINEAR)
    
    def visualize_sample(self, 
                        x_before: torch.Tensor, 
                        x_after: torch.Tensor, 
                        dynamic_inputs: List[torch.Tensor],
                        image: np.ndarray,
                        label: np.ndarray,
                        sample_name: str,
                        save_dir: str,
                        threshold: float = 0.5) -> Dict[str, np.ndarray]:
        """可视化单个样本的CAM图（不生成叠加图）"""
        # 确保输入在正确的设备上
        x_before = x_before.to(self.device)
        x_after = x_after.to(self.device)
        dynamic_inputs = [inp.to(self.device) for inp in dynamic_inputs] if dynamic_inputs else []
        
        # 确保输入是4D张量 (C, H, W) -> (1, C, H, W)
        if len(x_before.shape) == 3:
            x_before = x_before.unsqueeze(0)
        if len(x_after.shape) == 3:
            x_after = x_after.unsqueeze(0)
        dynamic_inputs = [inp.unsqueeze(0) if len(inp.shape) == 3 else inp for inp in dynamic_inputs]
        
        # 前向传播获取特征图
        with torch.no_grad():
            outputs = self.model(x_before, x_after, dynamic_inputs)
            preds = (torch.nn.functional.softmax(outputs, dim=1)[:, 1] > threshold).float()
        
        # 获取目标尺寸（与输入图像一致）- 确保是2D
        if len(image.shape) == 3:  # (H, W, C)
            target_size = (image.shape[1], image.shape[0])
        elif len(image.shape) == 2:  # (H, W)
            target_size = (image.shape[1], image.shape[0])
        else:
            raise ValueError(f"图像必须是2D或3D数组，得到形状 {image.shape}")
        
        # 生成各层的CAM图（只处理目标层级）
        cam_results = {}
        for layer_name in TARGET_LAYERS:
            if layer_name in self.feature_maps:
                feature_map = self.feature_maps[layer_name]
                # 检查特征图维度
                if len(feature_map.shape) in [3, 4, 5]:
                    cam = self._generate_cam(feature_map)
                    cam_upsampled = self._upsample_cam(cam, target_size)
                    cam_results[layer_name] = cam_upsampled
        
        # 创建保存目录
        sample_save_dir = os.path.join(save_dir, sample_name)
        os.makedirs(sample_save_dir, exist_ok=True)
        
        # 保存原始图像（无文字）
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.axis('off')  # 关闭坐标轴
        plt.tight_layout(pad=0)  # 移除边距
        plt.savefig(os.path.join(sample_save_dir, "original_image.png"), 
                   bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # 保存标签（无文字）
        plt.figure(figsize=(10, 8))
        plt.imshow(label, cmap='gray')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(sample_save_dir, "label.png"), 
                   bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # 保存预测结果（无文字）
        plt.figure(figsize=(10, 8))
        plt.imshow(preds.squeeze().cpu().numpy(), cmap='gray')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(sample_save_dir, "prediction.png"), 
                   bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # 保存各目标层级的CAM图（仅独立显示，不生成叠加图）
        for layer_name, cam in cam_results.items():
            # 单独显示CAM（无文字）
            plt.figure(figsize=(10, 8))
            plt.imshow(cam, cmap='jet')
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(os.path.join(sample_save_dir, f"cam_{layer_name}.png"), 
                       bbox_inches='tight', pad_inches=0)
            plt.close()
        
        return cam_results
    
    def visualize_dataset(self, 
                         full_dataset,  # 使用完整数据集而非测试集
                         sample_names: List[str],
                         feature_types: List[str],
                         feature_info: Dict[str, int],
                         save_dir: str,
                         num_samples: int = 10,  # 从数据集开头取这么多样本
                         threshold: float = 0.5):
        """可视化数据集中样本的CAM图 - 从数据集开头取指定数量的样本"""
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        print(f"开始生成CAM可视化结果，保存至: {save_dir}")
        print(f"使用特征可视化方法: {self.vis_method} - {VISUALIZATION_METHODS[self.vis_method]}")
        
        # 限制可视化的样本数量（从数据集开头选取）
        num_visualize = min(num_samples, len(full_dataset))
        print(f"将从完整数据集中选取前 {num_visualize} 个样本生成CAM图")
        
        # 逐个样本生成CAM图（从数据集开头开始）
        for i in range(num_visualize):
            # 获取样本数据（从完整数据集的第i个样本开始）
            features, labels = full_dataset[i]
            sample_name = sample_names[i]
            
            # 分割特征
            x_before, x_after, dynamic_inputs = self._split_features(
                features.unsqueeze(0), feature_types, feature_info
            )
            
            # 提取原始图像（使用指定的可视化方法）
            image_np = self._get_visualizable_image(x_before.squeeze(0))
            
            # 生成并保存CAM图
            self.visualize_sample(
                x_before.squeeze(0),
                x_after.squeeze(0),
                [d.squeeze(0) for d in dynamic_inputs],
                image_np,
                labels.numpy(),
                sample_name,
                os.path.join(save_dir, "cam_visualizations"),
                threshold
            )
            if (i + 1) % 5 == 0 or i + 1 == num_visualize:
                print(f"已处理 {i + 1}/{num_visualize} 个样本")
        
        print(f"CAM可视化完成，共生成 {num_visualize} 个样本的可视化结果")
    
    def _split_features(self, features, feature_types, feature_info):
        """分割特征为固定分支和动态分支"""
        # 计算各特征的起始和结束索引
        feature_slices = {}
        current_idx = 0
        for feature in feature_types:
            channels = feature_info[feature]
            feature_slices[feature] = slice(current_idx, current_idx + channels)
            current_idx += channels
        
        # 提取固定分支特征
        x_before = features[:, feature_slices['before'], :, :]
        x_after = features[:, feature_slices['after'], :, :]
        
        # 提取动态分支特征
        dynamic_inputs = []
        for feature in feature_types:
            if feature not in ['before', 'after']:
                dynamic_inputs.append(features[:, feature_slices[feature], :, :])
        
        return x_before, x_after, dynamic_inputs
    
    def _get_visualizable_image(self, x_before: torch.Tensor) -> np.ndarray:
        """从特征中提取可可视化的图像，提供多种可视化方法"""
        # 确保输入是3D张量 (C, H, W)
        if len(x_before.shape) == 4:  # 如果有批次维度，去掉
            x_before = x_before.squeeze(0)
        
        # 转换为numpy数组 (C, H, W)
        features_np = x_before.cpu().numpy()
        num_channels = features_np.shape[0]
        
        # 根据选择的方法处理特征
        if self.vis_method == 'first_three' or num_channels <= 3:
            # 使用前三个波段
            if num_channels >= 3:
                image_data = features_np[:3, :, :]
            else:
                # 如果波段数不足3，则重复现有波段
                image_data = np.repeat(features_np, 3, axis=0)[:3, :, :]
                
        elif self.vis_method == 'max_variance':
            # 选择方差最大的三个波段
            if num_channels <= 3:
                image_data = np.repeat(features_np, 3, axis=0)[:3, :, :]
            else:
                # 计算每个波段的方差
                variances = np.var(features_np, axis=(1, 2))
                # 获取方差最大的三个波段的索引
                top_indices = np.argsort(variances)[-3:][::-1]
                image_data = features_np[top_indices, :, :]
                
        elif self.vis_method == 'pca':
            # 使用PCA将波段降维到3个
            if num_channels <= 3:
                image_data = np.repeat(features_np, 3, axis=0)[:3, :, :]
            else:
                h, w = features_np.shape[1], features_np.shape[2]
                # 重塑为 (C, H*W)
                flattened = features_np.reshape(num_channels, -1)
                # 中心化
                flattened = flattened - np.mean(flattened, axis=1, keepdims=True)
                # 计算协方差矩阵
                cov_matrix = np.cov(flattened)
                # 计算特征值和特征向量
                eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
                # 选择前三个特征向量
                top_indices = np.argsort(eigenvalues)[-3:][::-1]
                top_eigenvectors = eigenvectors[:, top_indices]
                # 投影数据
                pca_result = np.dot(top_eigenvectors.T, flattened)
                # 重塑回 (3, H, W)
                image_data = pca_result.reshape(3, h, w)
                
        elif self.vis_method == 'grayscale_mean':
            # 所有波段的平均值（灰度图）
            mean_channel = np.mean(features_np, axis=0, keepdims=True)
            # 复制到三个通道以形成伪彩色图
            image_data = np.repeat(mean_channel, 3, axis=0)
            
        elif self.vis_method == 'grayscale_max':
            # 所有波段的最大值（灰度图）
            max_channel = np.max(features_np, axis=0, keepdims=True)
            # 复制到三个通道以形成伪彩色图
            image_data = np.repeat(max_channel, 3, axis=0)
        
        # 转置为HWC格式
        image = np.transpose(image_data, (1, 2, 0))
        
        # 归一化到0-255范围（不处理极端值，按原始数据范围）
        for i in range(3):
            min_val = np.min(image[:, :, i])
            max_val = np.max(image[:, :, i])
            if max_val > min_val:
                # 直接使用最大最小值归一化，不处理极端值
                image[:, :, i] = (image[:, :, i] - min_val) / (max_val - min_val) * 255
        
        return image.astype(np.uint8)
    
    def __del__(self):
        """清理钩子"""
        for hook in self.hooks:
            hook.remove()


def generate_cam_from_weights(config):
    """从预训练权重生成CAM可视化结果"""
    # 设置日志
    logger = setup_logger(config['log_dir'])
    logger.info("===== 从预训练权重生成CAM可视化 =====")
    logger.info(f"使用权重文件: {config['weight_path']}")
    
    # 初始化设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_count = torch.cuda.device_count()
    
    # 检查GPU数量
    if device_count < 2:
        logger.warning(f"检测到{device_count}个GPU，而训练时使用了双GPU。尝试继续，但可能出现问题。")
    else:
        logger.info(f"使用双GPU模式，设备: {device}")
    
    multiprocessing.set_start_method('spawn', force=True)
    
    # 创建数据加载器 - 获取完整数据集而非仅测试集
    _, _, _, feature_info, feature_types, full_dataset = create_dataloaders(config)
    # 获取完整数据集的样本名称列表
    sample_names = full_dataset.sample_names  # 假设数据集有sample_names属性包含所有样本名称
    
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
    
    # 加载预训练权重
    try:
        # 先加载到CPU以避免设备不匹配问题
        state_dict = torch.load(config['weight_path'], map_location='cpu')
        
        # 如果有多个GPU，使用DataParallel包装模型
        if device_count >= 2 and torch.cuda.is_available():
            model = torch.nn.DataParallel(model)
        
        # 将模型移动到设备
        model = model.to(device)
        
        # 加载权重
        model.load_state_dict(state_dict)
        logger.info(f"成功加载权重文件: {config['weight_path']}")
    except Exception as e:
        logger.error(f"权重文件加载失败: {str(e)}")
        # 尝试移除权重中的"module."前缀再加载（备选方案）
        try:
            logger.info("尝试移除权重中的'module.'前缀再加载...")
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
            logger.info("移除前缀后成功加载权重")
        except Exception as e2:
            logger.error(f"移除前缀后仍加载失败: {str(e2)}")
            raise
    
    model.eval()  # 设置为评估模式
    
    # 获取可视化方法
    vis_method = config.get('visualization_method', DEFAULT_VISUALIZATION)
    
    # 创建CAM可视化器
    cam_visualizer = CAMVisualizer(
        model=model,
        feature_names=feature_types,
        device=device,
        vis_method=vis_method
    )
    
    # 创建保存目录
    cam_save_dir = os.path.join(config['output_dir'], 'cam_visualizations')
    os.makedirs(cam_save_dir, exist_ok=True)
    logger.info(f"CAM可视化结果将保存至: {cam_save_dir}")
    
    # 生成可视化结果 - 使用完整数据集和指定数量的样本
    cam_visualizer.visualize_dataset(
        full_dataset=full_dataset,  # 传入完整数据集
        sample_names=sample_names,  # 传入完整数据集的样本名称
        feature_types=feature_types,
        feature_info=feature_info,
        save_dir=cam_save_dir,
        num_samples=config.get('cam_samples', 10),  # 由该参数控制从开头取多少样本
        threshold=config['pred_threshold']
    )
    
    return cam_save_dir


def main(config_path):
    """主函数"""
    # 解析配置文件
    config = parse_config(config_path)
    
    # 确保输出目录存在
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # 验证权重文件是否存在
    if not os.path.exists(config['weight_path']):
        raise FileNotFoundError(f"权重文件不存在: {config['weight_path']}")
    
    # 生成CAM可视化
    cam_dir = generate_cam_from_weights(config)
    print(f"CAM可视化已完成，结果保存在: {cam_dir}")
    return cam_dir


if __name__ == '__main__':
    try:
        if len(sys.argv) < 2:
            raise RuntimeError("请提供配置文件路径作为参数")
        
        config_path = sys.argv[1]
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        main(config_path)
        print('<status>0</status>')
        print('<message>CAM可视化生成成功</message>')
    except Exception as e:
        error_msg = str(e).replace('\n', ' ').replace('\t', ' ')
        print('<status>1</status>')
        print(f'<message>{error_msg}</message>')
        logger = logging.getLogger('cam_generator')
        if not logger.hasHandlers():
            logger.addHandler(logging.StreamHandler())
        logger.error(f"处理失败: {error_msg}")
