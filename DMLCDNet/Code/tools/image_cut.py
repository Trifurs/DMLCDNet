import os
import sys
import xml.etree.ElementTree as ET
import rasterio
from rasterio.windows import Window
import numpy as np
from tqdm import tqdm
import warnings
from PIL import Image
import random
import math
import pickle
import time
import traceback
from scipy import ndimage
import tempfile
import shutil

warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)

def parse_xml_config(xml_path):
    """解析XML参数配置文件"""
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"配置文件不存在: {xml_path}")
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    params = {}
    for param in root.findall('param'):
        name = param.find('name').text
        value = param.find('value').text
        
        # 根据参数类型转换数据类型
        param_type = param.find('type').text
        if param_type == 'int':
            value = int(value)
        elif param_type == 'float':
            value = float(value)
        elif param_type == 'bool':
            value = value.lower() == 'true'
        elif param_type == 'list':
            value = eval(value)
        
        params[name] = value
    
    # 添加默认值（确保参数存在）
    params.setdefault('seed', 114514)
    params.setdefault('target_sentinel_bands', ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 
                                               'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'])
    # RGB波段映射，默认B4(红), B3(绿), B2(蓝)
    params.setdefault('rgb_band_mapping', {'red': 'B4', 'green': 'B3', 'blue': 'B2'})
    
    return params

def setup_random_seeds(seed):
    """初始化所有随机种子，确保过程可复现"""
    random.seed(seed)
    np.random.seed(seed)
    print(f"已初始化随机种子: {seed}")

def fill_edge_nodata(file_path, output_path):
    """
    填充DEM派生数据（坡向、坡度）的边缘nodata值
    输入：原始文件路径；输出：处理后文件路径（不修改原始文件）
    使用最近邻有效值填充边缘nodata区域
    """
    try:
        with rasterio.open(file_path) as src:
            # 读取数据和nodata值
            data = src.read()
            nodata = src.nodata
            
            # 创建新数据数组
            filled_data = np.empty_like(data)
            
            # 处理每个波段
            for band_idx in range(data.shape[0]):
                band_data = data[band_idx]
                
                # 创建掩码：有效数据区域为True，nodata为False
                mask = band_data != nodata
                
                # 如果边缘有nodata，进行填充
                if not np.all(mask):
                    # 使用最近邻填充
                    distances, indices = ndimage.distance_transform_edt(
                        ~mask, return_indices=True
                    )
                    
                    # 获取最近有效值的索引
                    rr, cc = indices
                    filled_band = band_data[rr, cc]
                    
                    # 保留原始有效值
                    filled_band[mask] = band_data[mask]
                    filled_data[band_idx] = filled_band
                else:
                    filled_data[band_idx] = band_data
            
            # 获取元数据
            meta = src.meta.copy()
        
        # 保存到输出路径（不覆盖原始文件）
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(filled_data)
        
        return True
    
    except Exception as e:
        print(f"填充边缘nodata值时出错: {file_path}")
        print(f"错误详情: {str(e)}")
        traceback.print_exc()
        return False

def normalize_rgb_with_zero_preserve(band_data):
    """
    RGB波段归一化（保留0值像素）
    逻辑：1. 提取非0有效像素；2. 基于有效像素计算min/max；3. 仅对非0像素归一化到[0,1]；4. 0值像素保持为0
    返回：归一化后的波段数据（float32）
    """
    # 创建非0像素掩码（0值像素不参与计算）
    non_zero_mask = band_data != 0
    non_zero_data = band_data[non_zero_mask]
    
    # 处理全为0的情况
    if non_zero_data.size == 0:
        return np.zeros_like(band_data, dtype=np.float32)
    
    # 基于非0像素计算统计量
    min_val = np.min(non_zero_data)
    max_val = np.max(non_zero_data)
    
    # 避免除零（所有非0像素值相同）
    if max_val - min_val < 1e-6:
        normalized = np.zeros_like(band_data, dtype=np.float32)
        # 非0像素设为0.5（中间值）
        normalized[non_zero_mask] = 0.5
        return normalized
    
    # 仅对非0像素进行归一化，0值保持不变
    normalized = np.zeros_like(band_data, dtype=np.float32)
    normalized[non_zero_mask] = (non_zero_data - min_val) / (max_val - min_val)
    
    return normalized

def generate_full_area_rgb(temp_rgb_path, src_file_path, rgb_band_mapping):
    """
    在波段筛选前生成全区域临时RGB影像（使用完整原始波段）
    关键优化：保留原始0值像素，仅对非0像素计算全区域统计量并归一化
    输入：临时RGB保存路径、原始Sentinel文件路径、RGB波段映射
    输出：生成临时RGB影像，返回是否成功
    """
    try:
        with rasterio.open(src_file_path) as src:
            # 获取所有波段描述（完整原始波段）
            descriptions = src.descriptions if src.descriptions else []
            band_index_map = {desc: idx+1 for idx, desc in enumerate(descriptions)}  # 1-based索引
            
            # 检查RGB所需波段是否存在（原始完整波段中）
            missing_bands = []
            rgb_indices = []
            for color in ['red', 'green', 'blue']:
                band_name = rgb_band_mapping[color]
                if band_name not in band_index_map:
                    missing_bands.append(f"{color}({band_name})")
                else:
                    rgb_indices.append(band_index_map[band_name])
            
            if missing_bands:
                print(f"警告: 原始Sentinel文件 {os.path.basename(src_file_path)} 缺少RGB所需波段: {', '.join(missing_bands)}，无法生成RGB")
                return False
            
            # 读取RGB原始波段数据（保留原始数值，不提前缩放）
            rgb_data = src.read(rgb_indices)  # 形状: (3, H, W)
            
            # 对每个RGB波段独立归一化（保留0值像素，基于全区域非0像素统计）
            normalized_rgb = np.empty_like(rgb_data, dtype=np.float32)
            for i in range(3):
                normalized_rgb[i] = normalize_rgb_with_zero_preserve(rgb_data[i])
            
            # 转换为[0,255]（0值保持为0，其他值缩放）并调整维度为(H,W,3)
            rgb_8bit = (normalized_rgb.transpose(1, 2, 0) * 255).astype(np.uint8)
            
            # 强制保留0值（避免浮点误差导致0值变为1等）
            rgb_8bit[rgb_8bit < 1] = 0
            
            # 保存临时全区域RGB影像（PNG格式）
            img = Image.fromarray(rgb_8bit)
            img.save(temp_rgb_path)
            return True
    
    except Exception as e:
        print(f"生成全区域临时RGB时出错: {src_file_path}")
        print(f"错误详情: {str(e)}")
        traceback.print_exc()
        return False

def filter_sentinel_bands(file_path, output_path, target_bands):
    """
    筛选Sentinel-2影像波段（不修改原始文件）
    输入：原始文件路径、输出路径、目标波段列表；输出：筛选后文件
    """
    try:
        with rasterio.open(file_path) as src:
            # 获取所有波段描述
            descriptions = src.descriptions if src.descriptions else []
            
            # 筛选有效波段索引（rasterio波段索引从1开始）
            valid_band_indices = []
            found_bands = []
            
            for i in range(src.count):
                # 优先使用波段描述，无描述则用波段索引（1-based）
                desc = descriptions[i] if i < len(descriptions) else str(i+1)
                if desc in target_bands:
                    valid_band_indices.append(i+1)  # 转为rasterio读取用的1-based索引
                    found_bands.append(desc)
            
            # 检查找到的波段数量
            if len(valid_band_indices) < len(target_bands):
                missing = [b for b in target_bands if b not in found_bands]
                print(f"警告: {os.path.basename(file_path)} 缺少部分目标波段: {', '.join(missing)}")
            
            # 如果没有找到任何有效波段，保留所有波段
            if not valid_band_indices:
                print(f"警告: {os.path.basename(file_path)} 未找到有效Sentinel波段，保留所有波段")
                valid_band_indices = list(range(1, src.count+1))
                found_bands = [descriptions[i] if i < len(descriptions) else str(i+1) 
                               for i in range(src.count)]
            
            # 读取有效波段数据
            filtered_data = src.read(valid_band_indices)
            
            # 更新波段描述
            filtered_descriptions = found_bands
            
            # 获取元数据并更新波段数量
            meta = src.meta.copy()
            meta.update(count=len(valid_band_indices), dtype=filtered_data.dtype)
        
        # 保存到输出路径（不覆盖原始文件）
        with rasterio.open(output_path, 'w',** meta) as dst:
            dst.write(filtered_data)
            dst.descriptions = filtered_descriptions
        
        return True
        
    except Exception as e:
        print(f"处理Sentinel文件 {os.path.basename(file_path)} 时出错: {str(e)}")
        traceback.print_exc()
        return False

def normalize_image_in_memory(data):
    """
    在内存中对影像数据进行归一化处理（Min-Max归一化）
    每个波段独立归一化到[0, 1]范围
    返回归一化后的数据和每个波段的统计信息
    """
    # 处理单波段影像
    if len(data.shape) == 2:
        data = data[np.newaxis, ...]
    
    # 创建存储统计信息的字典
    band_stats = []
    
    # 对每个波段进行归一化
    normalized_data = np.empty_like(data, dtype=np.float32)
    for band_idx in range(data.shape[0]):
        band_data = data[band_idx]
        
        # 过滤无效值
        valid_mask = ~np.isnan(band_data) & ~np.isinf(band_data)
        valid_data = band_data[valid_mask]
        
        if valid_data.size == 0:
            print(f"警告: 波段{band_idx+1} 无有效数据，跳过归一化")
            normalized_data[band_idx] = band_data
            band_stats.append({'min': 0, 'max': 1})
            continue
        
        # 计算最小值和最大值
        min_val = np.min(valid_data)
        max_val = np.max(valid_data)
        
        # 保存统计信息
        band_stats.append({'min': min_val, 'max': max_val})
        
        # 归一化到[0, 1]范围
        if max_val - min_val > 1e-6:
            normalized_band = (band_data - min_val) / (max_val - min_val)
        else:
            # 如果所有值相同，设为0.5
            normalized_band = np.full_like(band_data, 0.5, dtype=np.float32)
        
        normalized_data[band_idx] = normalized_band
    
    return normalized_data, band_stats

def create_rgb_directories(output_dir):
    """创建RGB图像所需的目录结构"""
    rgb_root = os.path.join(output_dir, 'rgb_images')
    before_dir = os.path.join(rgb_root, 'before')
    after_dir = os.path.join(rgb_root, 'after')
    
    os.makedirs(before_dir, exist_ok=True)
    os.makedirs(after_dir, exist_ok=True)
    
    return before_dir, after_dir

def crop_and_normalize_small_rgb(temp_rgb_array, crop_window):
    """
    裁剪临时全区域RGB并对小影像独立缩放（保留0值）
    逻辑：1. 裁剪对应区域；2. 对裁剪后小影像的每个波段，提取非0像素计算min/max；3. 仅非0像素缩放至0-255；4. 0值保留
    输入：临时全区域RGB数组（H,W,3）、裁剪窗口(Window对象)
    输出：裁剪并缩放后的小影像RGB数组（H,W,3，uint8）
    """
    # 1. 裁剪对应区域（Window: col_off, row_off, width, height）
    left = int(crop_window.col_off)
    upper = int(crop_window.row_off)
    right = left + int(crop_window.width)
    lower = upper + int(crop_window.height)
    small_rgb = temp_rgb_array[upper:lower, left:right, :].copy()  # (crop_size, crop_size, 3)
    
    # 2. 对每个波段独立处理（保留0值，基于小影像自身非0像素统计）
    for band_idx in range(3):
        band_data = small_rgb[:, :, band_idx]
        non_zero_mask = band_data != 0
        non_zero_data = band_data[non_zero_mask]
        
        # 处理全为0的波段
        if non_zero_data.size == 0:
            small_rgb[:, :, band_idx] = 0
            continue
        
        # 基于小影像自身非0像素计算min/max（独立缩放）
        min_val = np.min(non_zero_data)
        max_val = np.max(non_zero_data)
        
        # 避免除零（所有非0像素值相同）
        if max_val - min_val < 1e-6:
            # 非0像素设为128（中间值），0值保持0
            band_data[non_zero_mask] = 128
            small_rgb[:, :, band_idx] = band_data
            continue
        
        # 仅对非0像素缩放至0-255，0值保持不变
        scaled_band = np.zeros_like(band_data, dtype=np.uint8)
        scaled_band[non_zero_mask] = ((non_zero_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        
        # 强制保留0值（消除浮点误差）
        scaled_band[scaled_band < 1] = 0
        small_rgb[:, :, band_idx] = scaled_band
    
    return small_rgb

def crop_temp_rgb(temp_rgb_path, crop_window, output_path):
    """
    从全区域临时RGB影像中裁剪对应窗口并保存（小影像独立缩放+保留0值）
    输入：临时RGB路径、裁剪窗口(Window对象)、输出路径
    输出：裁剪后的RGB图像（uint8，0-255，0值保留）
    """
    try:
        # 读取全区域临时RGB（保留原始uint8数据，避免二次读取损失）
        with Image.open(temp_rgb_path) as img:
            temp_rgb_array = np.array(img, dtype=np.uint8)  # (H,W,3)
        
        # 裁剪并对小影像独立缩放（核心优化步骤）
        small_rgb_array = crop_and_normalize_small_rgb(temp_rgb_array, crop_window)
        
        # 保存裁剪后的RGB
        small_img = Image.fromarray(small_rgb_array)
        small_img.save(output_path)
        return True
    
    except Exception as e:
        print(f"裁剪临时RGB时出错: {temp_rgb_path} -> {output_path}")
        print(f"错误详情: {str(e)}")
        traceback.print_exc()
        return False

def save_as_png(label_data, output_path):
    """
    将标签数据保存为PNG格式
    0 -> 0 (背景)
    1 -> 255 (滑坡)
    """
    # 确保标签是整数类型
    label_data = label_data.astype(np.uint8)
    
    # 转换为0-255范围
    png_data = np.where(label_data == 1, 255, 0).astype(np.uint8)
    
    # 使用PIL保存为PNG
    img = Image.fromarray(png_data)
    img.save(output_path)

def save_sample(params, area, i, j, data_crops, label_crop, label_meta, 
               before_rgb_dir, after_rgb_dir, temp_rgb_paths, crop_window, suffix=""):
    """保存样本（原始或增强）及其标签，同时从临时RGB裁剪对应区域（小影像独立缩放+保留0值）"""
    base_name = f"{area}_{i+1}_{j+1}{suffix}"
    
    # 确保标签是整数类型
    label_crop_int = label_crop.astype(np.uint8)
    
    # 保存标签TIFF
    label_out_path = os.path.join(params['output_dir'], 'label', f"{base_name}.tif")
    try:
        with rasterio.open(
            label_out_path, 'w',
            driver='GTiff',
            height=params['crop_size'],
            width=params['crop_size'],
            count=1,
            dtype=label_crop_int.dtype,
            crs=label_meta.get('crs'),
            transform=rasterio.windows.transform(crop_window, label_meta['transform'])
        ) as dst:
            dst.write(label_crop_int, 1)
    except Exception as e:
        print(f"保存标签 {os.path.basename(label_out_path)} 时出错: {e}")
    
    # 保存标签PNG
    png_out_path = os.path.join(params['label_png_dir'], f"{base_name}.png")
    try:
        save_as_png(label_crop_int, png_out_path)
    except Exception as e:
        print(f"保存PNG {os.path.basename(png_out_path)} 时出错: {e}")
    
    # 保存特征影像
    for data_type, (data_crop, data_meta) in data_crops.items():
        data_subdir = os.path.join(params['output_dir'], 'data', data_type)
        os.makedirs(data_subdir, exist_ok=True)
        
        data_out_path = os.path.join(data_subdir, f"{base_name}.tif")
        try:
            with rasterio.open(
                data_out_path, 'w',
                driver='GTiff',
                height=params['crop_size'],
                width=params['crop_size'],
                count=data_crop.shape[0],
                dtype=data_crop.dtype,
                crs=data_meta.get('crs'),
                transform=rasterio.windows.transform(crop_window, data_meta['transform'])
            ) as dst:
                dst.write(data_crop)
        except Exception as e:
            print(f"保存特征 {os.path.basename(data_out_path)} 时出错: {e}")
            continue
    
    # 从临时全区域RGB裁剪并保存对应区域的RGB（处理增强和原始样本）
    for rgb_type, temp_rgb_path in temp_rgb_paths.items():
        if not os.path.exists(temp_rgb_path):
            continue
        
        # 确定RGB保存目录
        if rgb_type == 'before':
            rgb_save_dir = before_rgb_dir
        elif rgb_type == 'after':
            rgb_save_dir = after_rgb_dir
        else:
            continue
        
        # 保存裁剪后的RGB（小影像独立缩放+保留0值）
        rgb_output_path = os.path.join(rgb_save_dir, f"{base_name}.png")
        crop_temp_rgb(temp_rgb_path, crop_window, rgb_output_path)

def preprocess_data_files(data_dir, temp_data_dir, temp_rgb_dir, params):
    """
    预处理数据文件（不修改原始数据）：
    1. 对aspect和slope文件填充边缘nodata（输出到临时目录）
    2. 对Sentinel-2影像（before和after）：先生成全区域临时RGB，再筛选波段（输出到临时目录）
    3. 其他文件直接复制
    """
    print("\n开始数据预处理...")
    
    # 遍历所有tif文件
    tif_files = [f for f in os.listdir(data_dir) if f.endswith('.tif')]
    
    for tif_file in tqdm(tif_files, desc="预处理数据文件"):
        src_file_path = os.path.join(data_dir, tif_file)
        dst_file_path = os.path.join(temp_data_dir, tif_file)  # 筛选后数据保存路径
        file_name = os.path.basename(src_file_path).lower()
        
        # 处理DEM派生数据（aspect/slope）
        if 'aspect' in file_name or 'slope' in file_name:
            fill_edge_nodata(src_file_path, dst_file_path)
        
        # 处理Sentinel-2影像（before/after）：先生成临时RGB，再筛选波段
        elif 'before' in file_name or 'after' in file_name:
            # 1. 生成全区域临时RGB（在波段筛选前，保留0值）
            rgb_type = 'before' if 'before' in file_name else 'after'
            temp_rgb_path = os.path.join(temp_rgb_dir, f"{rgb_type}_full_rgb.png")
            generate_full_area_rgb(temp_rgb_path, src_file_path, params['rgb_band_mapping'])
            
            # 2. 筛选波段并保存到临时数据目录
            filter_sentinel_bands(
                src_file_path, 
                dst_file_path, 
                params['target_sentinel_bands']  # 从XML读取目标波段
            )
        else:
            # 其他文件直接复制到临时目录（不处理）
            shutil.copy2(src_file_path, dst_file_path)
    
    print("数据预处理完成!")

def apply_image_augmentation(img, method, is_label=False):
    """
    应用图像增强方法
    参数:
    is_label: 是否为标签数据（标签只能应用几何变换）
    """
    if method == 'flip_h':
        return np.flip(img, axis=2)  # 水平翻转
    elif method == 'flip_v':
        return np.flip(img, axis=1)  # 垂直翻转
    elif method == 'rotate90':
        return np.rot90(img, k=1, axes=(1, 2))
    elif method == 'rotate180':
        return np.rot90(img, k=2, axes=(1, 2))
    elif method == 'rotate270':
        return np.rot90(img, k=3, axes=(1, 2))
    
    # 以下增强只应用于特征数据（非标签）
    if not is_label:
        if method == 'brightness':
            factor = 1 + random.uniform(-0.3, 0.3)
            augmented_img = img * factor
            # 裁剪到[0,1]范围
            return np.clip(augmented_img, 0, 1)
        elif method == 'contrast':
            factor = random.uniform(0.7, 1.3)
            mean = np.mean(img, axis=(1, 2), keepdims=True)
            augmented_img = (img - mean) * factor + mean
            # 裁剪到[0,1]范围
            return np.clip(augmented_img, 0, 1)
        elif method == 'gaussian_noise':
            noise = np.random.normal(0, 0.05, img.shape)
            augmented_img = img + noise
            # 裁剪到[0,1]范围
            return np.clip(augmented_img, 0, 1)
    
    return img

def process_dataset(params):
    """处理数据集主函数"""
    # 创建输出目录结构
    os.makedirs(os.path.join(params['output_dir'], 'data'), exist_ok=True)
    os.makedirs(os.path.join(params['output_dir'], 'label'), exist_ok=True)
    os.makedirs(params['label_png_dir'], exist_ok=True)  # PNG标签目录
    
    # 创建RGB图像目录
    before_rgb_dir, after_rgb_dir = create_rgb_directories(params['output_dir'])
    print(f"RGB图像将保存到:\n  灾前: {before_rgb_dir}\n  灾后: {after_rgb_dir}")
    
    # 定义图像增强方法（从XML读取）
    augmentation_methods = params.get('augmentation_methods', [
        'flip_h', 'flip_v', 'rotate90', 'rotate180', 'rotate270',
        'brightness', 'contrast', 'gaussian_noise'
    ])
    augmentation_prob = params.get('augmentation_prob', 0.5)  # 增强概率
    
    # 获取所有区域
    area_dirs = [d for d in os.listdir(params['input_dir']) 
                if os.path.isdir(os.path.join(params['input_dir'], d))]
    if not area_dirs:
        raise ValueError(f"输入目录 {params['input_dir']} 下无区域子目录")
    
    # 创建临时目录用于存储预处理后的数据（不修改原始数据）
    with tempfile.TemporaryDirectory(prefix="landslide_preprocess_") as temp_root_dir:
        print(f"\n创建临时目录用于预处理数据: {temp_root_dir}")
        
        # 预处理每个区域的数据文件（输出到临时目录）
        for area in area_dirs:
            area_src_path = os.path.join(params['input_dir'], area)
            area_temp_path = os.path.join(temp_root_dir, area)
            os.makedirs(area_temp_path, exist_ok=True)
            
            # 临时数据目录（存储预处理后的特征数据）
            temp_data_dir = os.path.join(area_temp_path, 'data')
            os.makedirs(temp_data_dir, exist_ok=True)
            
            # 临时RGB目录（存储全区域临时RGB影像，裁剪后删除）
            temp_rgb_dir = os.path.join(area_temp_path, 'rgb_temp')
            os.makedirs(temp_rgb_dir, exist_ok=True)
            
            # 临时标签目录（直接复制原始标签，不处理）
            temp_label_dir = os.path.join(area_temp_path, 'label')
            os.makedirs(temp_label_dir, exist_ok=True)
            
            # 复制原始标签到临时目录
            src_label_path = os.path.join(area_src_path, 'label', f"{area}.tif")
            if os.path.exists(src_label_path):
                shutil.copy2(src_label_path, os.path.join(temp_label_dir, f"{area}.tif"))
            else:
                print(f"警告: 区域 {area} 原始标签文件不存在: {src_label_path}")
            
            # 预处理特征数据（先生成临时RGB，再筛选波段）
            src_data_dir = os.path.join(area_src_path, 'data')
            if os.path.exists(src_data_dir):
                print(f"\n预处理区域: {area}")
                preprocess_data_files(src_data_dir, temp_data_dir, temp_rgb_dir, params)
            else:
                print(f"警告: 区域 {area} 原始数据目录不存在: {src_data_dir}")
        
        # 执行裁剪处理
        start_time = time.time()
        for area in tqdm(area_dirs, desc="处理区域"):
            area_temp_path = os.path.join(temp_root_dir, area)
            
            # 1. 读取临时目录中的标签
            temp_label_path = os.path.join(area_temp_path, 'label', f"{area}.tif")
            if not os.path.exists(temp_label_path):
                print(f"警告: 区域 {area} 缺少临时标签文件，跳过")
                continue
                
            try:
                with rasterio.open(temp_label_path) as src:
                    label = src.read(1)
                    label_meta = src.meta.copy()
                    label_height, label_width = label.shape
            except Exception as e:
                print(f"读取标签文件 {os.path.basename(temp_label_path)} 时出错: {e}")
                continue
            
            # 2. 读取临时目录中的预处理后特征数据
            temp_data_dir = os.path.join(area_temp_path, 'data')
            if not os.path.exists(temp_data_dir):
                print(f"警告: 区域 {area} 缺少临时数据目录，跳过")
                continue
                
            data_files = [f for f in os.listdir(temp_data_dir) if f.endswith('.tif')]
            if not data_files:
                print(f"警告: 区域 {area} 无临时特征文件，跳过")
                continue
            
            # 3. 获取临时全区域RGB路径（before/after）
            temp_rgb_dir = os.path.join(area_temp_path, 'rgb_temp')
            temp_rgb_paths = {
                'before': os.path.join(temp_rgb_dir, 'before_full_rgb.png'),
                'after': os.path.join(temp_rgb_dir, 'after_full_rgb.png')
            }
            
            # 4. 验证临时RGB是否存在（至少有一个）
            if not any(os.path.exists(p) for p in temp_rgb_paths.values()):
                print(f"警告: 区域 {area} 无有效临时RGB文件，RGB生成功能将失效")
            
            # 5. 计算裁剪位置
            crop_size = params['crop_size']
            if label_height < crop_size or label_width < crop_size:
                print(f"警告: 区域 {area} 影像尺寸({label_height}x{label_width})小于裁剪尺寸({crop_size}x{crop_size})，跳过")
                continue
            
            col_offsets = range(0, label_width - crop_size + 1, params['stride'])
            row_offsets = range(0, label_height - crop_size + 1, params['stride'])
            
            min_landslide_ratio = params.get('min_landslide_ratio', 0.01)
            max_background_ratio = params.get('max_background_ratio', 0.3)
            
            background_count = 0
            total_samples = 0

            # 6. 遍历所有裁剪位置
            for i, row in enumerate(tqdm(row_offsets, desc=f"{area} 行进度", leave=False)):
                for j, col in enumerate(tqdm(col_offsets, desc=f"{area} 列进度", leave=False)):
                    # 创建裁剪窗口对象（统一坐标体系）
                    crop_window = Window(col, row, crop_size, crop_size)
                    
                    # 裁剪标签
                    label_crop = label[row:row+crop_size, col:col+crop_size]
                    
                    # 计算滑坡像素占比
                    landslide_ratio = np.sum(label_crop == 1) / label_crop.size
                    
                    # 智能样本筛选
                    if landslide_ratio < min_landslide_ratio:
                        if background_count / (total_samples + 1) > max_background_ratio:
                            continue
                        background_count += 1
                    total_samples += 1
                    
                    # 7. 处理所有特征影像（从临时目录读取预处理后的数据）
                    data_crops = {}
                    for data_file in data_files:
                        # 提取数据类型（文件名格式：xxx_类型.tif）
                        data_type = data_file.split('_')[-1].split('.')[0]
                        data_path = os.path.join(temp_data_dir, data_file)
                        
                        try:
                            # 读取预处理后的特征数据
                            with rasterio.open(data_path) as src:
                                # 验证数据尺寸与标签一致
                                data_height, data_width = src.height, src.width
                                if data_height != label_height or data_width != label_width:
                                    print(f"警告: 特征文件 {data_file} 尺寸与标签不一致，跳过")
                                    continue
                                
                                # 裁剪数据
                                data_crop = src.read(window=crop_window)
                                data_meta = src.meta.copy()
                                
                                # 内存中归一化
                                normalized_data, _ = normalize_image_in_memory(data_crop)
                                data_crops[data_type] = (normalized_data, data_meta)
                        except Exception as e:
                            print(f"处理特征文件 {os.path.basename(data_file)} 时出错: {e}")
                            continue
                    
                    if not data_crops:
                        print(f"警告: 裁剪位置({row},{col})无有效特征数据，跳过")
                        continue
                    
                    # 8. 保存原始样本（包含从临时RGB裁剪的RGB图像）
                    save_sample(
                        params, area, i, j, 
                        data_crops, label_crop, label_meta,
                        before_rgb_dir, after_rgb_dir,
                        temp_rgb_paths, crop_window,
                        suffix=""
                    )
                    
                    # 9. 应用图像增强（随机过程由seed控制）
                    if random.random() < augmentation_prob:
                        num_augmentations = random.randint(1, 3)
                        selected_methods = random.sample(augmentation_methods, num_augmentations)
                        
                        for method in selected_methods:
                            # 应用增强到特征影像
                            augmented_data = {}
                            for data_type, (data_crop, data_meta) in data_crops.items():
                                augmented_data[data_type] = (
                                    apply_image_augmentation(data_crop, method, is_label=False), 
                                    data_meta
                                )
                            
                            # 应用增强到标签（只允许几何变换）
                            if method in ['flip_h', 'flip_v', 'rotate90', 'rotate180', 'rotate270']:
                                augmented_label = apply_image_augmentation(
                                    label_crop[np.newaxis, ...], method, is_label=True
                                )[0]
                            else:
                                # 辐射变换不应用于标签
                                augmented_label = label_crop.copy()
                            
                            # 保存增强样本（RGB图像无需增强，直接用原始裁剪结果，增强后缀仅标记样本来源）
                            save_sample(
                                params, area, i, j, 
                                augmented_data, augmented_label, label_meta,
                                before_rgb_dir, after_rgb_dir,
                                temp_rgb_paths, crop_window,
                                suffix=f"_{method}"
                            )
        
        print(f"\n临时目录已自动清理（含临时全区域RGB文件）")

def main():
    # 1. 从命令行获取配置文件路径
    if len(sys.argv) < 2:
        print("用法: python image_cut.py <配置文件路径>")
        print("示例: python image_cut.py cut_parameters.xml")
        sys.exit(1)
    config_path = sys.argv[1]
    
    # 2. 解析配置参数
    try:
        params = parse_xml_config(config_path)
    except Exception as e:
        print(f"解析配置文件失败: {str(e)}")
        sys.exit(1)
    
    # 3. 初始化随机种子
    setup_random_seeds(params['seed'])
    
    # 4. 验证输入输出目录
    if not os.path.exists(params['input_dir']):
        print(f"错误: 输入目录 {params['input_dir']} 不存在")
        sys.exit(1)
    os.makedirs(params['output_dir'], exist_ok=True)
    os.makedirs(params['label_png_dir'], exist_ok=True)
    
    # 5. 打印配置信息
    print("===== 滑坡检测数据准备 =====")
    print("配置参数:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # 6. 执行数据处理
    start_time = time.time()
    try:
        process_dataset(params)
    except Exception as e:
        print(f"\n数据处理失败: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
    
    # 7. 计算处理时间
    elapsed = time.time() - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\n===== 数据处理完成 =====")
    print(f"耗时: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    print(f"输出目录: {params['output_dir']}")
    print(f"标签PNG目录: {params['label_png_dir']}")
    print(f"RGB图像目录: {os.path.join(params['output_dir'], 'rgb_images')}")

if __name__ == "__main__":
    main()
    