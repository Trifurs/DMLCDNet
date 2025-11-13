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
    """Parse XML parameter configuration file"""
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Configuration file not found: {xml_path}")
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    params = {}
    for param in root.findall('param'):
        name = param.find('name').text
        value = param.find('value').text
        
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
    
    params.setdefault('seed', 114514)
    params.setdefault('target_sentinel_bands', ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 
                                               'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'])
    params.setdefault('rgb_band_mapping', {'red': 'B4', 'green': 'B3', 'blue': 'B2'})
    
    return params

def setup_random_seeds(seed):
    """Initialize all random seeds to ensure reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    print(f"Random seeds initialized: {seed}")

def fill_edge_nodata(file_path, output_path):
    """
    Fill edge nodata values for DEM-derived data (aspect, slope)
    Input: Original file path; Output: Processed file path (original remains unmodified)
    Uses nearest valid value to fill edge nodata regions
    """
    try:
        with rasterio.open(file_path) as src:
            data = src.read()
            nodata = src.nodata
            
            filled_data = np.empty_like(data)
            
            for band_idx in range(data.shape[0]):
                band_data = data[band_idx]
                
                mask = band_data != nodata
                
                if not np.all(mask):
                    distances, indices = ndimage.distance_transform_edt(
                        ~mask, return_indices=True
                    )
                    
                    rr, cc = indices
                    filled_band = band_data[rr, cc]
                    
                    filled_band[mask] = band_data[mask]
                    filled_data[band_idx] = filled_band
                else:
                    filled_data[band_idx] = band_data
            
            meta = src.meta.copy()
        
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(filled_data)
        
        return True
    
    except Exception as e:
        print(f"Error filling edge nodata values: {file_path}")
        print(f"Error details: {str(e)}")
        traceback.print_exc()
        return False

def normalize_rgb_with_zero_preserve(band_data):
    """
    RGB band normalization (preserving zero-value pixels)
    Logic: 1. Extract non-zero valid pixels; 2. Calculate min/max based on valid pixels; 
           3. Normalize only non-zero pixels to [0,1]; 4. Keep zero-value pixels as 0
    Returns: Normalized band data (float32)
    """
    non_zero_mask = band_data != 0
    non_zero_data = band_data[non_zero_mask]
    
    if non_zero_data.size == 0:
        return np.zeros_like(band_data, dtype=np.float32)
    
    min_val = np.min(non_zero_data)
    max_val = np.max(non_zero_data)
    
    if max_val - min_val < 1e-6:
        normalized = np.zeros_like(band_data, dtype=np.float32)
        normalized[non_zero_mask] = 0.5
        return normalized
    
    normalized = np.zeros_like(band_data, dtype=np.float32)
    normalized[non_zero_mask] = (non_zero_data - min_val) / (max_val - min_val)
    
    return normalized

def generate_full_area_rgb(temp_rgb_path, src_file_path, rgb_band_mapping):
    """
    Generate full-area temporary RGB image before band filtering (using complete original bands)
    Key optimization: Preserve original zero-value pixels, calculate statistics and normalize 
                      only for non-zero pixels across the entire area
    Input: Temporary RGB save path, original Sentinel file path, RGB band mapping
    Output: Generates temporary RGB image, returns success status
    """
    try:
        with rasterio.open(src_file_path) as src:
            descriptions = src.descriptions if src.descriptions else []
            band_index_map = {desc: idx+1 for idx, desc in enumerate(descriptions)}
            
            missing_bands = []
            rgb_indices = []
            for color in ['red', 'green', 'blue']:
                band_name = rgb_band_mapping[color]
                if band_name not in band_index_map:
                    missing_bands.append(f"{color}({band_name})")
                else:
                    rgb_indices.append(band_index_map[band_name])
            
            if missing_bands:
                print(f"Warning: Original Sentinel file {os.path.basename(src_file_path)} missing bands required for RGB: {', '.join(missing_bands)}, cannot generate RGB")
                return False
            
            rgb_data = src.read(rgb_indices)
            
            normalized_rgb = np.empty_like(rgb_data, dtype=np.float32)
            for i in range(3):
                normalized_rgb[i] = normalize_rgb_with_zero_preserve(rgb_data[i])
            
            rgb_8bit = (normalized_rgb.transpose(1, 2, 0) * 255).astype(np.uint8)
            
            rgb_8bit[rgb_8bit < 1] = 0
            
            img = Image.fromarray(rgb_8bit)
            img.save(temp_rgb_path)
            return True
    
    except Exception as e:
        print(f"Error generating full-area temporary RGB: {src_file_path}")
        print(f"Error details: {str(e)}")
        traceback.print_exc()
        return False

def filter_sentinel_bands(file_path, output_path, target_bands):
    """
    Filter Sentinel-2 image bands (original file remains unmodified)
    Input: Original file path, output path, target band list; Output: Filtered file
    """
    try:
        with rasterio.open(file_path) as src:
            descriptions = src.descriptions if src.descriptions else []
            
            valid_band_indices = []
            found_bands = []
            
            for i in range(src.count):
                desc = descriptions[i] if i < len(descriptions) else str(i+1)
                if desc in target_bands:
                    valid_band_indices.append(i+1)
                    found_bands.append(desc)
            
            if len(valid_band_indices) < len(target_bands):
                missing = [b for b in target_bands if b not in found_bands]
                print(f"Warning: {os.path.basename(file_path)} missing some target bands: {', '.join(missing)}")
            
            if not valid_band_indices:
                print(f"Warning: {os.path.basename(file_path)} no valid Sentinel bands found, keeping all bands")
                valid_band_indices = list(range(1, src.count+1))
                found_bands = [descriptions[i] if i < len(descriptions) else str(i+1) 
                               for i in range(src.count)]
            
            filtered_data = src.read(valid_band_indices)
            
            filtered_descriptions = found_bands
            
            meta = src.meta.copy()
            meta.update(count=len(valid_band_indices), dtype=filtered_data.dtype)
        
        with rasterio.open(output_path, 'w',** meta) as dst:
            dst.write(filtered_data)
            dst.descriptions = filtered_descriptions
        
        return True
        
    except Exception as e:
        print(f"Error processing Sentinel file {os.path.basename(file_path)}: {str(e)}")
        traceback.print_exc()
        return False

def normalize_image_in_memory(data):
    """
    Normalize image data in memory (Min-Max normalization)
    Each band is independently normalized to [0, 1] range
    Returns normalized data and statistics for each band
    """
    if len(data.shape) == 2:
        data = data[np.newaxis, ...]
    
    band_stats = []
    
    normalized_data = np.empty_like(data, dtype=np.float32)
    for band_idx in range(data.shape[0]):
        band_data = data[band_idx]
        
        valid_mask = ~np.isnan(band_data) & ~np.isinf(band_data)
        valid_data = band_data[valid_mask]
        
        if valid_data.size == 0:
            print(f"Warning: Band {band_idx+1} has no valid data, skipping normalization")
            normalized_data[band_idx] = band_data
            band_stats.append({'min': 0, 'max': 1})
            continue
        
        min_val = np.min(valid_data)
        max_val = np.max(valid_data)
        
        band_stats.append({'min': min_val, 'max': max_val})
        
        if max_val - min_val > 1e-6:
            normalized_band = (band_data - min_val) / (max_val - min_val)
        else:
            normalized_band = np.full_like(band_data, 0.5, dtype=np.float32)
        
        normalized_data[band_idx] = normalized_band
    
    return normalized_data, band_stats

def create_rgb_directories(output_dir):
    """Create directory structure for RGB images"""
    rgb_root = os.path.join(output_dir, 'rgb_images')
    before_dir = os.path.join(rgb_root, 'before')
    after_dir = os.path.join(rgb_root, 'after')
    
    os.makedirs(before_dir, exist_ok=True)
    os.makedirs(after_dir, exist_ok=True)
    
    return before_dir, after_dir

def crop_and_normalize_small_rgb(temp_rgb_array, crop_window):
    """
    Crop temporary full-area RGB and independently scale small images (preserving 0 values)
    Logic: 1. Crop corresponding area; 2. For each band in cropped small image, extract 
           non-zero pixels to calculate min/max; 3. Scale only non-zero pixels to 0-255; 
           4. Preserve 0 values
    Input: Temporary full-area RGB array (H,W,3), crop window (Window object)
    Output: Cropped and scaled small image RGB array (H,W,3, uint8)
    """
    left = int(crop_window.col_off)
    upper = int(crop_window.row_off)
    right = left + int(crop_window.width)
    lower = upper + int(crop_window.height)
    small_rgb = temp_rgb_array[upper:lower, left:right, :].copy()
    
    for band_idx in range(3):
        band_data = small_rgb[:, :, band_idx]
        non_zero_mask = band_data != 0
        non_zero_data = band_data[non_zero_mask]
        
        if non_zero_data.size == 0:
            small_rgb[:, :, band_idx] = 0
            continue
        
        min_val = np.min(non_zero_data)
        max_val = np.max(non_zero_data)
        
        if max_val - min_val < 1e-6:
            band_data[non_zero_mask] = 128
            small_rgb[:, :, band_idx] = band_data
            continue
        
        scaled_band = np.zeros_like(band_data, dtype=np.uint8)
        scaled_band[non_zero_mask] = ((non_zero_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        
        scaled_band[scaled_band < 1] = 0
        small_rgb[:, :, band_idx] = scaled_band
    
    return small_rgb

def crop_temp_rgb(temp_rgb_path, crop_window, output_path):
    """
    Crop corresponding window from full-area temporary RGB image and save 
    (small image independent scaling + 0 value preservation)
    Input: Temporary RGB path, crop window (Window object), output path
    Output: Cropped RGB image (uint8, 0-255, 0 values preserved)
    """
    try:
        with Image.open(temp_rgb_path) as img:
            temp_rgb_array = np.array(img, dtype=np.uint8)
        
        small_rgb_array = crop_and_normalize_small_rgb(temp_rgb_array, crop_window)
        
        small_img = Image.fromarray(small_rgb_array)
        small_img.save(output_path)
        return True
    
    except Exception as e:
        print(f"Error cropping temporary RGB: {temp_rgb_path} -> {output_path}")
        print(f"Error details: {str(e)}")
        traceback.print_exc()
        return False

def save_as_png(label_data, output_path):
    """
    Save label data as PNG format
    0 -> 0 (background)
    1 -> 255 (landslide)
    """
    label_data = label_data.astype(np.uint8)
    
    png_data = np.where(label_data == 1, 255, 0).astype(np.uint8)
    
    img = Image.fromarray(png_data)
    img.save(output_path)

def save_sample(params, area, i, j, data_crops, label_crop, label_meta, 
               before_rgb_dir, after_rgb_dir, temp_rgb_paths, crop_window, suffix=""):
    """Save sample (original or augmented) and its label, while cropping corresponding area from temporary RGB"""
    base_name = f"{area}_{i+1}_{j+1}{suffix}"
    
    label_crop_int = label_crop.astype(np.uint8)
    
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
        print(f"Error saving label {os.path.basename(label_out_path)}: {e}")
    
    png_out_path = os.path.join(params['label_png_dir'], f"{base_name}.png")
    try:
        save_as_png(label_crop_int, png_out_path)
    except Exception as e:
        print(f"Error saving PNG {os.path.basename(png_out_path)}: {e}")
    
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
            print(f"Error saving feature {os.path.basename(data_out_path)}: {e}")
            continue
    
    for rgb_type, temp_rgb_path in temp_rgb_paths.items():
        if not os.path.exists(temp_rgb_path):
            continue
        
        if rgb_type == 'before':
            rgb_save_dir = before_rgb_dir
        elif rgb_type == 'after':
            rgb_save_dir = after_rgb_dir
        else:
            continue
        
        rgb_output_path = os.path.join(rgb_save_dir, f"{base_name}.png")
        crop_temp_rgb(temp_rgb_path, crop_window, rgb_output_path)

def preprocess_data_files(data_dir, temp_data_dir, temp_rgb_dir, params):
    """
    Preprocess data files (original data remains unmodified):
    1. Fill edge nodata for aspect and slope files (output to temporary directory)
    2. For Sentinel-2 images (before and after): first generate full-area temporary RGB, then filter bands
    3. Directly copy other files
    """
    print("\nStarting data preprocessing...")
    
    tif_files = [f for f in os.listdir(data_dir) if f.endswith('.tif')]
    
    for tif_file in tqdm(tif_files, desc="Preprocessing data files"):
        src_file_path = os.path.join(data_dir, tif_file)
        dst_file_path = os.path.join(temp_data_dir, tif_file)
        file_name = os.path.basename(src_file_path).lower()
        
        if 'aspect' in file_name or 'slope' in file_name:
            fill_edge_nodata(src_file_path, dst_file_path)
        
        elif 'before' in file_name or 'after' in file_name:
            rgb_type = 'before' if 'before' in file_name else 'after'
            temp_rgb_path = os.path.join(temp_rgb_dir, f"{rgb_type}_full_rgb.png")
            generate_full_area_rgb(temp_rgb_path, src_file_path, params['rgb_band_mapping'])
            
            filter_sentinel_bands(
                src_file_path, 
                dst_file_path, 
                params['target_sentinel_bands']
            )
        else:
            shutil.copy2(src_file_path, dst_file_path)
    
    print("Data preprocessing completed!")

def apply_image_augmentation(img, method, is_label=False):
    """
    Apply image augmentation methods
    Parameters:
    is_label: Whether it's label data (labels can only apply geometric transformations)
    """
    if method == 'flip_h':
        return np.flip(img, axis=2)
    elif method == 'flip_v':
        return np.flip(img, axis=1)
    elif method == 'rotate90':
        return np.rot90(img, k=1, axes=(1, 2))
    elif method == 'rotate180':
        return np.rot90(img, k=2, axes=(1, 2))
    elif method == 'rotate270':
        return np.rot90(img, k=3, axes=(1, 2))
    
    if not is_label:
        if method == 'brightness':
            factor = 1 + random.uniform(-0.3, 0.3)
            augmented_img = img * factor
            return np.clip(augmented_img, 0, 1)
        elif method == 'contrast':
            factor = random.uniform(0.7, 1.3)
            mean = np.mean(img, axis=(1, 2), keepdims=True)
            augmented_img = (img - mean) * factor + mean
            return np.clip(augmented_img, 0, 1)
        elif method == 'gaussian_noise':
            noise = np.random.normal(0, 0.05, img.shape)
            augmented_img = img + noise
            return np.clip(augmented_img, 0, 1)
    
    return img

def process_dataset(params):
    """Main function for dataset processing"""
    os.makedirs(os.path.join(params['output_dir'], 'data'), exist_ok=True)
    os.makedirs(os.path.join(params['output_dir'], 'label'), exist_ok=True)
    os.makedirs(params['label_png_dir'], exist_ok=True)
    
    before_rgb_dir, after_rgb_dir = create_rgb_directories(params['output_dir'])
    print(f"RGB images will be saved to:\n  Before disaster: {before_rgb_dir}\n  After disaster: {after_rgb_dir}")
    
    augmentation_methods = params.get('augmentation_methods', [
        'flip_h', 'flip_v', 'rotate90', 'rotate180', 'rotate270',
        'brightness', 'contrast', 'gaussian_noise'
    ])
    augmentation_prob = params.get('augmentation_prob', 0.5)
    
    area_dirs = [d for d in os.listdir(params['input_dir']) 
                if os.path.isdir(os.path.join(params['input_dir'], d))]
    if not area_dirs:
        raise ValueError(f"No area subdirectories found in input directory {params['input_dir']}")
    
    with tempfile.TemporaryDirectory(prefix="landslide_preprocess_") as temp_root_dir:
        print(f"\nCreated temporary directory for preprocessing data: {temp_root_dir}")
        
        for area in area_dirs:
            area_src_path = os.path.join(params['input_dir'], area)
            area_temp_path = os.path.join(temp_root_dir, area)
            os.makedirs(area_temp_path, exist_ok=True)
            
            temp_data_dir = os.path.join(area_temp_path, 'data')
            os.makedirs(temp_data_dir, exist_ok=True)
            
            temp_rgb_dir = os.path.join(area_temp_path, 'rgb_temp')
            os.makedirs(temp_rgb_dir, exist_ok=True)
            
            temp_label_dir = os.path.join(area_temp_path, 'label')
            os.makedirs(temp_label_dir, exist_ok=True)
            
            src_label_path = os.path.join(area_src_path, 'label', f"{area}.tif")
            if os.path.exists(src_label_path):
                shutil.copy2(src_label_path, os.path.join(temp_label_dir, f"{area}.tif"))
            else:
                print(f"Warning: Original label file for area {area} not found: {src_label_path}")
            
            src_data_dir = os.path.join(area_src_path, 'data')
            if os.path.exists(src_data_dir):
                print(f"\nPreprocessing area: {area}")
                preprocess_data_files(src_data_dir, temp_data_dir, temp_rgb_dir, params)
            else:
                print(f"Warning: Original data directory for area {area} not found: {src_data_dir}")
        
        start_time = time.time()
        for area in tqdm(area_dirs, desc="Processing areas"):
            area_temp_path = os.path.join(temp_root_dir, area)
            
            temp_label_path = os.path.join(area_temp_path, 'label', f"{area}.tif")
            if not os.path.exists(temp_label_path):
                print(f"Warning: Missing temporary label file for area {area}, skipping")
                continue
                
            try:
                with rasterio.open(temp_label_path) as src:
                    label = src.read(1)
                    label_meta = src.meta.copy()
                    label_height, label_width = label.shape
            except Exception as e:
                print(f"Error reading label file {os.path.basename(temp_label_path)}: {e}")
                continue
            
            temp_data_dir = os.path.join(area_temp_path, 'data')
            if not os.path.exists(temp_data_dir):
                print(f"Warning: Missing temporary data directory for area {area}, skipping")
                continue
                
            data_files = [f for f in os.listdir(temp_data_dir) if f.endswith('.tif')]
            if not data_files:
                print(f"Warning: No temporary feature files for area {area}, skipping")
                continue
            
            temp_rgb_dir = os.path.join(area_temp_path, 'rgb_temp')
            temp_rgb_paths = {
                'before': os.path.join(temp_rgb_dir, 'before_full_rgb.png'),
                'after': os.path.join(temp_rgb_dir, 'after_full_rgb.png')
            }
            
            if not any(os.path.exists(p) for p in temp_rgb_paths.values()):
                print(f"Warning: No valid temporary RGB files for area {area}, RGB generation will be disabled")
            
            crop_size = params['crop_size']
            if label_height < crop_size or label_width < crop_size:
                print(f"Warning: Area {area} image size({label_height}x{label_width}) smaller than crop size({crop_size}x{crop_size}), skipping")
                continue
            
            col_offsets = range(0, label_width - crop_size + 1, params['stride'])
            row_offsets = range(0, label_height - crop_size + 1, params['stride'])
            
            min_landslide_ratio = params.get('min_landslide_ratio', 0.01)
            max_background_ratio = params.get('max_background_ratio', 0.3)
            
            background_count = 0
            total_samples = 0

            for i, row in enumerate(tqdm(row_offsets, desc=f"{area} row progress", leave=False)):
                for j, col in enumerate(tqdm(col_offsets, desc=f"{area} column progress", leave=False)):
                    crop_window = Window(col, row, crop_size, crop_size)
                    
                    label_crop = label[row:row+crop_size, col:col+crop_size]
                    
                    landslide_ratio = np.sum(label_crop == 1) / label_crop.size
                    
                    if landslide_ratio < min_landslide_ratio:
                        if background_count / (total_samples + 1) > max_background_ratio:
                            continue
                        background_count += 1
                    total_samples += 1
                    
                    data_crops = {}
                    for data_file in data_files:
                        data_type = data_file.split('_')[-1].split('.')[0]
                        data_path = os.path.join(temp_data_dir, data_file)
                        
                        try:
                            with rasterio.open(data_path) as src:
                                data_height, data_width = src.height, src.width
                                if data_height != label_height or data_width != label_width:
                                    print(f"Warning: Feature file {data_file} size does not match label, skipping")
                                    continue
                                
                                data_crop = src.read(window=crop_window)
                                data_meta = src.meta.copy()
                                
                                normalized_data, _ = normalize_image_in_memory(data_crop)
                                data_crops[data_type] = (normalized_data, data_meta)
                        except Exception as e:
                            print(f"Error processing feature file {os.path.basename(data_file)}: {e}")
                            continue
                    
                    if not data_crops:
                        print(f"Warning: No valid feature data at crop position({row},{col}), skipping")
                        continue
                    
                    save_sample(
                        params, area, i, j, 
                        data_crops, label_crop, label_meta,
                        before_rgb_dir, after_rgb_dir,
                        temp_rgb_paths, crop_window,
                        suffix=""
                    )
                    
                    if random.random() < augmentation_prob:
                        num_augmentations = random.randint(1, 3)
                        selected_methods = random.sample(augmentation_methods, num_augmentations)
                        
                        for method in selected_methods:
                            augmented_data = {}
                            for data_type, (data_crop, data_meta) in data_crops.items():
                                augmented_data[data_type] = (
                                    apply_image_augmentation(data_crop, method, is_label=False), 
                                    data_meta
                                )
                            
                            if method in ['flip_h', 'flip_v', 'rotate90', 'rotate180', 'rotate270']:
                                augmented_label = apply_image_augmentation(
                                    label_crop[np.newaxis, ...], method, is_label=True
                                )[0]
                            else:
                                augmented_label = label_crop.copy()
                            
                            save_sample(
                                params, area, i, j, 
                                augmented_data, augmented_label, label_meta,
                                before_rgb_dir, after_rgb_dir,
                                temp_rgb_paths, crop_window,
                                suffix=f"_{method}"
                            )
        
        print(f"\nTemporary directory automatically cleaned up (including temporary full-area RGB files)")

def main():
    if len(sys.argv) < 2:
        print("Usage: python image_cut.py <configuration file path>")
        print("Example: python image_cut.py cut_parameters.xml")
        sys.exit(1)
    config_path = sys.argv[1]
    
    try:
        params = parse_xml_config(config_path)
    except Exception as e:
        print(f"Failed to parse configuration file: {str(e)}")
        sys.exit(1)
    
    setup_random_seeds(params['seed'])
    
    if not os.path.exists(params['input_dir']):
        print(f"Error: Input directory {params['input_dir']} does not exist")
        sys.exit(1)
    os.makedirs(params['output_dir'], exist_ok=True)
    os.makedirs(params['label_png_dir'], exist_ok=True)
    
    print("===== Landslide Detection Data Preparation =====")
    print("Configuration parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    start_time = time.time()
    try:
        process_dataset(params)
    except Exception as e:
        print(f"\nData processing failed: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
    
    elapsed = time.time() - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\n===== Data Processing Completed =====")
    print(f"Time elapsed: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    print(f"Output directory: {params['output_dir']}")
    print(f"Label PNG directory: {params['label_png_dir']}")
    print(f"RGB images directory: {os.path.join(params['output_dir'], 'rgb_images')}")

if __name__ == "__main__":
    main()
    
