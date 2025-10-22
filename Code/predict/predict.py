import os
import sys
import xml.etree.ElementTree as ET
import rasterio
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import DataParallel
from tqdm import tqdm
import warnings
import tempfile
import shutil
from PIL import Image
from scipy import ndimage
import traceback
import time

warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)
warnings.filterwarnings('ignore', category=UserWarning)

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
        elif param_type == 'dict':
            value = eval(value)
        
        params[name] = value
    
    params.setdefault('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    params.setdefault('batch_size', 8)
    params.setdefault('target_sentinel_bands', ['B1', 'B2', 'B3', 'B4', 'B5'])
    params.setdefault('use_multi_gpu', True if torch.cuda.device_count() > 1 else False)
    
    return params

def setup_device(device_name, use_multi_gpu=True):
    """Set up computing device, support multi-GPU"""
    if device_name == 'cuda':
        if not torch.cuda.is_available():
            print("Warning: CUDA is not available, will use CPU")
            return torch.device('cpu'), False
        
        gpu_count = torch.cuda.device_count()
        if gpu_count >= 2 and use_multi_gpu:
            print(f"Found {gpu_count} GPUs, will use dual GPU parallel computing")
            return torch.device('cuda'), True
        else:
            print(f"Using single GPU: {torch.cuda.get_device_name(0)}")
            return torch.device('cuda'), False
    else:
        print("Using CPU for computation")
        return torch.device('cpu'), False

def dynamic_import_model(model_path, class_name):
    """Dynamically import model class"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model definition file not found: {model_path}")
    
    model_dir = os.path.dirname(model_path)
    if model_dir not in sys.path:
        sys.path.append(model_dir)
    
    module_name = os.path.splitext(os.path.basename(model_path))[0]
    module = __import__(module_name)
    
    if not hasattr(module, class_name):
        raise AttributeError(f"Class {class_name} not found in model file {model_path}")
    
    return getattr(module, class_name)

def fill_edge_nodata(file_path, output_path):
    """Fill edge nodata values"""
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
        print(f"Error when filling edge nodata values: {file_path}")
        print(f"Error details: {str(e)}")
        traceback.print_exc()
        return False

def filter_sentinel_bands(file_path, output_path, target_bands):
    """Filter Sentinel-2 image bands"""
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
                print(f"Warning: {os.path.basename(file_path)} is missing some target bands: {', '.join(missing)}")
            
            if not valid_band_indices:
                print(f"Warning: No valid Sentinel bands found in {os.path.basename(file_path)}, retaining all bands")
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
    """Normalize image data in memory"""
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

def preprocess_area_data(area, src_dir, temp_dir, params):
    """Preprocess data for a specified area"""
    print(f"\nPreprocessing area: {area}")
    
    area_temp_dir = os.path.join(temp_dir, area)
    os.makedirs(area_temp_dir, exist_ok=True)
    
    data_temp_dir = os.path.join(area_temp_dir, 'data')
    os.makedirs(data_temp_dir, exist_ok=True)
    
    src_data_dir = os.path.join(src_dir, area, 'data')
    if not os.path.exists(src_data_dir):
        raise ValueError(f"Data directory for area {area} does not exist: {src_data_dir}")
    
    tif_files = [f for f in os.listdir(src_data_dir) if f.endswith('.tif')]
    if not tif_files:
        raise ValueError(f"No TIFF files found in data directory for area {area}")
    
    before_file = None
    after_file = None
    other_files = []
    
    for tif_file in tif_files:
        if 'before' in tif_file.lower():
            before_file = tif_file
        elif 'after' in tif_file.lower():
            after_file = tif_file
        else:
            other_files.append(tif_file)
    
    if not before_file or not after_file:
        raise ValueError(f"Missing before or after file for area {area}")
    
    src_before_path = os.path.join(src_data_dir, before_file)
    dst_before_path = os.path.join(data_temp_dir, before_file)
    filter_sentinel_bands(src_before_path, dst_before_path, params['target_sentinel_bands'])
    
    src_after_path = os.path.join(src_data_dir, after_file)
    dst_after_path = os.path.join(data_temp_dir, after_file)
    filter_sentinel_bands(src_after_path, dst_after_path, params['target_sentinel_bands'])
    
    for tif_file in other_files:
        src_path = os.path.join(src_data_dir, tif_file)
        dst_path = os.path.join(data_temp_dir, tif_file)
        
        if 'aspect' in tif_file.lower() or 'slope' in tif_file.lower():
            fill_edge_nodata(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)
    
    feature_info = {}
    
    with rasterio.open(dst_before_path) as src:
        feature_info['before'] = src.count
    
    with rasterio.open(dst_after_path) as src:
        feature_info['after'] = src.count
    
    for tif_file in other_files:
        data_type = tif_file.split('_')[-1].split('.')[0]
        if data_type in params['exclude_features']:
            continue
            
        file_path = os.path.join(data_temp_dir, tif_file)
        with rasterio.open(file_path) as src:
            feature_info[data_type] = src.count
    
    feature_types = ['before', 'after']
    for tif_file in other_files:
        data_type = tif_file.split('_')[-1].split('.')[0]
        if data_type not in params['exclude_features'] and data_type not in feature_types:
            feature_types.append(data_type)
    
    with rasterio.open(dst_before_path) as src:
        meta = src.meta.copy()
        height, width = src.height, src.width
    
    return {
        'data_dir': data_temp_dir,
        'before_file': before_file,
        'after_file': after_file,
        'other_files': other_files,
        'feature_info': feature_info,
        'feature_types': feature_types,
        'meta': meta,
        'height': height,
        'width': width
    }

def generate_windows(height, width, window_size, stride_x, stride_y):
    """Generate coordinates for all sliding windows"""
    windows = []
    
    num_windows_x = int(np.ceil((width - window_size) / stride_x)) + 1
    num_windows_y = int(np.ceil((height - window_size) / stride_y)) + 1
    
    last_x = max(0, width - window_size)
    last_y = max(0, height - window_size)
    
    for i in range(num_windows_y):
        y = last_y if i == num_windows_y - 1 else i * stride_y
        if y + window_size > height:
            y = height - window_size
        
        for j in range(num_windows_x):
            x = last_x if j == num_windows_x - 1 else j * stride_x
            if x + window_size > width:
                x = width - window_size
            
            windows.append((x, y, window_size, window_size))
    
    return windows

def load_window_features(window, data_info, params):
    """Load all feature data within a window"""
    x, y, w, h = window
    features = {}
    
    before_path = os.path.join(data_info['data_dir'], data_info['before_file'])
    with rasterio.open(before_path) as src:
        window_obj = rasterio.windows.Window(x, y, w, h)
        data = src.read(window=window_obj)
        normalized_data, _ = normalize_image_in_memory(data)
        features['before'] = normalized_data
    
    after_path = os.path.join(data_info['data_dir'], data_info['after_file'])
    with rasterio.open(after_path) as src:
        window_obj = rasterio.windows.Window(x, y, w, h)
        data = src.read(window=window_obj)
        normalized_data, _ = normalize_image_in_memory(data)
        features['after'] = normalized_data
    
    for tif_file in data_info['other_files']:
        data_type = tif_file.split('_')[-1].split('.')[0]
        if data_type in params['exclude_features']:
            continue
            
        file_path = os.path.join(data_info['data_dir'], tif_file)
        with rasterio.open(file_path) as src:
            window_obj = rasterio.windows.Window(x, y, w, h)
            data = src.read(window=window_obj)
            normalized_data, _ = normalize_image_in_memory(data)
            features[data_type] = normalized_data
    
    return features

def split_features(features_dict, feature_types):
    """Convert feature dictionary to model input format"""
    x_before = features_dict['before']
    x_after = features_dict['after']
    
    dynamic_inputs = []
    for feature_type in feature_types:
        if feature_type not in ['before', 'after'] and feature_type in features_dict:
            dynamic_inputs.append(features_dict[feature_type])
    
    return x_before, x_after, dynamic_inputs

def predict_windows(model, windows, data_info, params, device, use_multi_gpu):
    """Predict for all windows"""
    window_size = params['window_size']
    height, width = data_info['height'], data_info['width']
    
    predictions = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.int32)
    
    batch_size = params['batch_size']
    num_batches = int(np.ceil(len(windows) / batch_size))
    
    model.eval()
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Processing prediction batches"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(windows))
            batch_windows = windows[start_idx:end_idx]
            
            batch_features = []
            for window in batch_windows:
                features_dict = load_window_features(window, data_info, params)
                x_before, x_after, dynamic_inputs = split_features(features_dict, data_info['feature_types'])
                
                x_before = torch.from_numpy(x_before).float().unsqueeze(0)
                x_after = torch.from_numpy(x_after).float().unsqueeze(0)
                dynamic_inputs = [torch.from_numpy(d).float().unsqueeze(0) for d in dynamic_inputs]
                
                batch_features.append((x_before, x_after, dynamic_inputs))
            
            batch_outputs = []
            for x_before, x_after, dynamic_inputs in batch_features:
                x_before = x_before.to(device)
                x_after = x_after.to(device)
                dynamic_inputs = [d.to(device) for d in dynamic_inputs]
                
                output = model(x_before, x_after, dynamic_inputs)
                prob = F.softmax(output, dim=1)[:, 1]
                batch_outputs.append(prob.cpu().numpy().squeeze())
            
            for i, (x, y, w, h) in enumerate(batch_windows):
                pred = batch_outputs[i]
                
                y_end = min(y + h, height)
                x_end = min(x + w, width)
                pred_cropped = pred[:y_end - y, :x_end - x]
                
                predictions[y:y_end, x:x_end] += pred_cropped
                counts[y:y_end, x:x_end] += 1
    
    counts[counts == 0] = 1
    
    predictions /= counts
    
    binary_pred = (predictions >= params['pred_threshold']).astype(np.uint8) * 255
    
    return predictions, binary_pred

def save_prediction_results(area, predictions, binary_pred, data_info, output_dir):
    """Save prediction results"""
    area_output_dir = os.path.join(output_dir, area)
    os.makedirs(area_output_dir, exist_ok=True)
    
    prob_path = os.path.join(area_output_dir, f"{area}_prob.tif")
    meta = data_info['meta'].copy()
    meta.update(dtype=np.float32, count=1)
    
    with rasterio.open(prob_path, 'w', **meta) as dst:
        dst.write(predictions.astype(np.float32), 1)
    
    binary_path = os.path.join(area_output_dir, f"{area}_prediction.tif")
    meta = data_info['meta'].copy()
    meta.update(dtype=np.uint8, count=1)
    
    with rasterio.open(binary_path, 'w',** meta) as dst:
        dst.write(binary_pred, 1)
    
    png_path = os.path.join(area_output_dir, f"{area}_prediction.png")
    img = Image.fromarray(binary_pred)
    img.save(png_path)
    
    print(f"Prediction results saved to: {area_output_dir}")

def process_area(area, params, device, model, use_multi_gpu):
    """Process prediction for a single area"""
    print(f"\n===== Starting processing area: {area} =====")
    
    with tempfile.TemporaryDirectory(prefix=f"pred_{area}_") as temp_dir:
        try:
            data_info = preprocess_area_data(
                area=area,
                src_dir=params['input_dir'],
                temp_dir=temp_dir,
                params=params
            )
            
            print(f"Generating sliding windows...")
            windows = generate_windows(
                height=data_info['height'],
                width=data_info['width'],
                window_size=params['window_size'],
                stride_x=params['stride_x'],
                stride_y=params['stride_y']
            )
            print(f"Generated {len(windows)} sliding windows")
            
            print(f"Starting prediction...")
            start_time = time.time()
            predictions, binary_pred = predict_windows(
                model=model,
                windows=windows,
                data_info=data_info,
                params=params,
                device=device,
                use_multi_gpu=use_multi_gpu
            )
            elapsed = time.time() - start_time
            print(f"Prediction completed, time elapsed: {elapsed:.2f} seconds")
            
            save_prediction_results(
                area=area,
                predictions=predictions,
                binary_pred=binary_pred,
                data_info=data_info,
                output_dir=params['output_dir']
            )
            
            print(f"===== Area {area} processing completed =====")
            return True
            
        except Exception as e:
            print(f"Error processing area {area}: {str(e)}")
            traceback.print_exc()
            return False

def load_model_with_parallel(model_class, model_path, device, use_multi_gpu, fixed_in_channels, dynamic_in_channels):
    """Load model with multi-GPU support, handle module prefix in state_dict"""
    model = model_class(
        fixed_in_channels=fixed_in_channels,
        dynamic_in_channels=dynamic_in_channels
    )
    
    state_dict = torch.load(model_path, map_location=device)
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    
    if use_multi_gpu and device.type == 'cuda':
        model = DataParallel(model, device_ids=[0, 1])
    
    model = model.to(device)
    return model

def main():
    if len(sys.argv) < 2:
        print("Usage: python sliding_window_prediction.py <configuration file path>")
        print("Example: python sliding_window_prediction.py prediction_parameters.xml")
        sys.exit(1)
    config_path = sys.argv[1]
    
    try:
        params = parse_xml_config(config_path)
    except Exception as e:
        print(f"Failed to parse configuration file: {str(e)}")
        sys.exit(1)
    
    device, use_multi_gpu = setup_device(params['device'], params.get('use_multi_gpu', True))
    print(f"Using computing device: {device}, Multi-GPU mode: {'Enabled' if use_multi_gpu else 'Disabled'}")
    
    os.makedirs(params['output_dir'], exist_ok=True)
    
    print("===== Landslide Detection Sliding Window Prediction =====")
    print("Configuration parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    try:
        model_class = dynamic_import_model(params['model_def_path'], params['model_class_name'])
        print(f"Successfully imported model class: {params['model_class_name']}")
    except Exception as e:
        print(f"Failed to import model: {str(e)}")
        sys.exit(1)
    
    area_dirs = [d for d in os.listdir(params['input_dir']) 
                if os.path.isdir(os.path.join(params['input_dir'], d))]
    if not area_dirs:
        print(f"Error: No area subdirectories found in input directory {params['input_dir']}")
        sys.exit(1)
    print(f"Found {len(area_dirs)} areas: {', '.join(area_dirs)}")
    
    try:
        with tempfile.TemporaryDirectory(prefix="model_init_") as temp_dir:
            data_info = preprocess_area_data(
                area=area_dirs[0],
                src_dir=params['input_dir'],
                temp_dir=temp_dir,
                params=params
            )
            
            dynamic_branch_channels = [
                data_info['feature_info'][feature] 
                for feature in data_info['feature_types'] 
                if feature not in ['before', 'after']
            ]
            fixed_in_channels = data_info['feature_info']['before']
            
            model = load_model_with_parallel(
                model_class=model_class,
                model_path=params['model_path'],
                device=device,
                use_multi_gpu=use_multi_gpu,
                fixed_in_channels=fixed_in_channels,
                dynamic_in_channels=dynamic_branch_channels
            )
            
            print(f"Successfully loaded model weights: {params['model_path']}")
            print(f"Model structure - Fixed branch channels: {fixed_in_channels}, Number of dynamic branches: {len(dynamic_branch_channels)}")
            
    except Exception as e:
        print(f"Failed to initialize model: {str(e)}")
        sys.exit(1)
    
    start_time = time.time()
    success_count = 0
    
    for area in area_dirs:
        if process_area(area, params, device, model, use_multi_gpu):
            success_count += 1
    
    elapsed = time.time() - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    
    print("\n===== Sliding Window Prediction Completed =====")
    print(f"Total time elapsed: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    print(f"Processed areas: {len(area_dirs)} total, Success: {success_count}, Failed: {len(area_dirs) - success_count}")
    print(f"Prediction results root directory: {params['output_dir']}")

if __name__ == "__main__":
    main()
    