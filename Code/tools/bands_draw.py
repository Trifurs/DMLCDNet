import os
import sys
import xml.etree.ElementTree as ET
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from tqdm import tqdm
import warnings
import traceback
import time
from matplotlib.backends.backend_agg import FigureCanvasAgg

# 统一字体设置 - 所有字体大小在这里集中定义
FONT_SETTINGS = {
    "font.family": ["Arial", "sans-serif"],
    "axes.unicode_minus": False,  # 解决负号显示问题
    "font.size": 26,              # 基础字体大小
    "axes.titlesize": 40,         # 子图标题字体大小
    "axes.labelsize": 28,         # 轴标签字体大小
    "xtick.labelsize": 14,        # x轴刻度字体大小
    "ytick.labelsize": 14,        # y轴刻度字体大小
    "legend.fontsize": 38         # 图例字体大小
}

# 应用字体设置
for key, value in FONT_SETTINGS.items():
    plt.rcParams[key] = value

# 创建加粗字体属性
bold_font = FontProperties(weight='bold', size=FONT_SETTINGS["font.size"])
bold_title_font = FontProperties(weight='bold', size=FONT_SETTINGS["axes.titlesize"])
bold_label_font = FontProperties(weight='bold', size=FONT_SETTINGS["axes.labelsize"])
bold_legend_font = FontProperties(weight='bold', size=FONT_SETTINGS["legend.fontsize"])

# 忽略不必要的警告
warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)
warnings.filterwarnings('ignore', category=UserWarning)  # 忽略matplotlib的警告

def parse_xml_config(xml_path):
    """Parse XML configuration file"""
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Configuration file not found: {xml_path}")
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    params = {}
    for param in root.findall('param'):
        name = param.find('name').text
        value = param.find('value').text
        
        # Convert data type according to parameter type
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
    
    # Add default parameters
    params.setdefault('bins', 100)  # Number of histogram bins
    # 动态计算合适的图大小，基于字体大小和子图布局
    subplot_layout = params.get('subplot_layout', (4, 3))
    base_width = 20  # 每个子图的基础宽度
    base_height = 6  # 每个子图的基础高度
    params.setdefault('figsize', (subplot_layout[1] * base_width, subplot_layout[0] * base_height))
    params.setdefault('dpi', 600)  # Image resolution
    params.setdefault('before_color', 'steelblue')  # Before-landslide histogram color
    params.setdefault('after_color', 'crimson')     # Post-landslide histogram color
    params.setdefault('alpha', 0.7)                # Transparency
    params.setdefault('subplot_layout', (4, 3))     # Subplot arrangement (rows, columns)
    params.setdefault('max_samples_per_band', None)  # Max samples per band
    
    # Ensure target bands parameter exists
    if 'target_sentinel_bands' not in params:
        params['target_sentinel_bands'] = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 
                                          'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
    
    return params

def remove_extremes(band_data):
    """Remove maximum and minimum values, return processed data"""
    if len(band_data) <= 2:  # Too few data points, don't remove
        return band_data.copy()
    
    # Find indices of maximum and minimum values
    max_idx = np.argmax(band_data)
    min_idx = np.argmin(band_data)
    
    # If max and min are the same element (all values identical)
    if max_idx == min_idx:
        return band_data.copy()
    
    # Create mask to exclude max and min values
    mask = np.ones(band_data.shape, dtype=bool)
    mask[max_idx] = False
    mask[min_idx] = False
    
    return band_data[mask]

def normalize_without_extremes(band_data):
    """Normalize after removing extremes, return normalized data and statistics used"""
    if len(band_data) == 0:
        return band_data, (0, 1)
    
    # Remove maximum and minimum values
    filtered_data = remove_extremes(band_data)
    
    if len(filtered_data) < 2:  # Too few data points to calculate valid range
        return np.zeros_like(band_data, dtype=np.float32), (0, 1)
    
    # Use second smallest and second largest values for normalization
    min_val = np.min(filtered_data)
    max_val = np.max(filtered_data)
    
    # Avoid division by zero
    if max_val - min_val < 1e-10:
        return np.zeros_like(band_data, dtype=np.float32), (min_val, max_val)
    
    # Normalize original data (including extremes) using statistics from filtered data
    normalized_data = (band_data - min_val) / (max_val - min_val)
    
    # Clip extreme values outside [0,1] range
    normalized_data = np.clip(normalized_data, 0, 1)
    
    return normalized_data, (min_val, max_val)

def get_band_index(band_name, descriptions, default_names):
    """Get band index based on band name"""
    if descriptions:
        for idx, desc in enumerate(descriptions):
            if desc == band_name:
                return idx
    if band_name in default_names:
        return default_names.index(band_name)
    return -1  # Not found

def get_band_name(band_idx, descriptions, default_names):
    """Get band name, prioritizing descriptions, then default names, then index"""
    if descriptions and band_idx < len(descriptions) and descriptions[band_idx]:
        return descriptions[band_idx]
    if band_idx < len(default_names):
        return default_names[band_idx]
    return f"Band{band_idx + 1}"

def read_label_mask(label_path):
    """Read label file, return mask for landslide areas (1 indicates landslide)"""
    try:
        with rasterio.open(label_path) as src:
            # Read first band
            label_data = src.read(1)
            # Ensure integer data type
            label_data = label_data.astype(np.int32)
            # Create mask: True for landslide areas, False otherwise
            mask = (label_data == 1)
            return mask, src.transform, src.crs
    except Exception as e:
        print(f"Error reading label file: {os.path.basename(label_path)}")
        print(f"Error details: {str(e)}")
        return None, None, None

def mask_band_data(band_data, band_transform, band_crs, mask, mask_transform, mask_crs):
    """
    Use mask to filter band data for landslide areas only
    Handles cases with different coordinate systems and resolutions
    """
    try:
        # Check if reprojection is needed
        if band_crs != mask_crs:
            # If coordinate systems differ, reproject mask to band's coordinate system
            from rasterio.warp import reproject, Resampling
            
            # Create mask array with same shape as band
            reprojected_mask = np.zeros(band_data.shape, dtype=np.bool_)
            
            # Reproject
            reproject(
                source=mask.astype(np.float32),
                destination=reprojected_mask,
                src_transform=mask_transform,
                src_crs=mask_crs,
                dst_transform=band_transform,
                dst_crs=band_crs,
                resampling=Resampling.nearest
            )
            
            mask = reprojected_mask.astype(np.bool_)
        else:
            # Same coordinate system but may need resizing to match
            if band_data.shape != mask.shape:
                # Resize mask using nearest neighbor interpolation
                from scipy.ndimage import zoom
                
                # Calculate zoom factors
                zoom_factor = (
                    band_data.shape[0] / mask.shape[0],
                    band_data.shape[1] / mask.shape[1]
                )
                
                # Resize mask
                mask = zoom(mask.astype(np.float32), zoom_factor, order=0) > 0.5
        
        # Apply mask, keeping only landslide area data
        masked_data = band_data[mask]
        return masked_data
        
    except Exception as e:
        print(f"Error applying mask")
        print(f"Error details: {str(e)}")
        return None

def process_single_area(area_dir, area_name, output_dir, global_before_data, global_after_data, params):
    """Process single area, generate histogram for the area, and add data to global dataset"""
    # Data and label directories
    data_dir = os.path.join(area_dir, 'data')
    label_dir = os.path.join(area_dir, 'label')
    
    if not os.path.exists(data_dir):
        print(f"Warning: Data directory does not exist - {data_dir}, skipping area")
        return False, None, None
    
    if not os.path.exists(label_dir):
        print(f"Warning: Label directory does not exist - {label_dir}, skipping area")
        return False, None, None
    
    # Find label file
    label_path = None
    for filename in os.listdir(label_dir):
        if filename.lower() == f"{area_name.lower()}.tif":
            label_path = os.path.join(label_dir, filename)
            break
    
    if not label_path:
        print(f"Warning: Label file not found for area {area_name}, skipping area")
        return False, None, None
    
    # Read label mask
    mask, mask_transform, mask_crs = read_label_mask(label_path)
    if mask is None:
        return False, None, None
    
    # Find before-landslide and post-landslide images
    before_image = None
    after_image = None
    
    for filename in os.listdir(data_dir):
        if filename.lower().endswith('.tif'):
            if '_before' in filename.lower():
                before_image = os.path.join(data_dir, filename)
            elif '_after' in filename.lower():
                after_image = os.path.join(data_dir, filename)
    
    # Verify images exist
    if not before_image:
        print(f"Warning: Before-landslide image not found for area {area_name}, skipping area")
        return False, None, None
    if not after_image:
        print(f"Warning: Post-landslide image not found for area {area_name}, skipping area")
        return False, None, None
    
    # Read before-landslide image band data (landslide areas only)
    try:
        with rasterio.open(before_image) as src:
            before_descriptions = src.descriptions if src.descriptions else []
            before_bands = {}
            
            # Process only target bands
            for band_name in params['target_sentinel_bands']:
                band_idx = get_band_index(band_name, before_descriptions, params['target_sentinel_bands'])
                if band_idx == -1:
                    print(f"Warning: Before-landslide image for area {area_name} missing band {band_name}")
                    continue
                
                # Read band data (rasterio uses 1-based index)
                band_data = src.read(band_idx + 1)
                
                # Apply mask to keep only landslide areas
                masked_data = mask_band_data(
                    band_data, src.transform, src.crs,
                    mask, mask_transform, mask_crs
                )
                
                if masked_data is None or len(masked_data) == 0:
                    print(f"Warning: No valid landslide area data for before-landslide band {band_name} in area {area_name}")
                    continue
                
                # Filter invalid values
                valid_data = masked_data[~np.isnan(masked_data) & ~np.isinf(masked_data)]
                
                # Random sampling if max sample count is set
                if params['max_samples_per_band'] and valid_data.size > params['max_samples_per_band']:
                    valid_data = np.random.choice(valid_data, size=params['max_samples_per_band'], replace=False)
                
                before_bands[band_name] = valid_data
    except Exception as e:
        print(f"Error reading before-landslide image: {os.path.basename(before_image)}")
        print(f"Error details: {str(e)}")
        return False, None, None
    
    # Read post-landslide image band data (landslide areas only)
    try:
        with rasterio.open(after_image) as src:
            after_descriptions = src.descriptions if src.descriptions else []
            after_bands = {}
            
            # Process only target bands
            for band_name in params['target_sentinel_bands']:
                band_idx = get_band_index(band_name, after_descriptions, params['target_sentinel_bands'])
                if band_idx == -1:
                    print(f"Warning: Post-landslide image for area {area_name} missing band {band_name}")
                    continue
                
                # Read band data (rasterio uses 1-based index)
                band_data = src.read(band_idx + 1)
                
                # Apply mask to keep only landslide areas
                masked_data = mask_band_data(
                    band_data, src.transform, src.crs,
                    mask, mask_transform, mask_crs
                )
                
                if masked_data is None or len(masked_data) == 0:
                    print(f"Warning: No valid landslide area data for post-landslide band {band_name} in area {area_name}")
                    continue
                
                # Filter invalid values
                valid_data = masked_data[~np.isnan(masked_data) & ~np.isinf(masked_data)]
                
                # Random sampling if max sample count is set
                if params['max_samples_per_band'] and valid_data.size > params['max_samples_per_band']:
                    valid_data = np.random.choice(valid_data, size=params['max_samples_per_band'], replace=False)
                
                after_bands[band_name] = valid_data
    except Exception as e:
        print(f"Error reading post-landslide image: {os.path.basename(after_image)}")
        print(f"Error details: {str(e)}")
        return False, None, None
    
    # Create output directory for this area
    area_output_dir = os.path.join(output_dir, 'area_plots')
    os.makedirs(area_output_dir, exist_ok=True)
    
    # Normalize each band for this area (removing extremes)
    area_before_norm = {}
    area_after_norm = {}
    
    for band_name in params['target_sentinel_bands']:
        # Process before-landslide data
        if band_name in before_bands and len(before_bands[band_name]) > 0:
            norm_data, _ = normalize_without_extremes(before_bands[band_name])
            area_before_norm[band_name] = norm_data
            # Add to global data
            if band_name not in global_before_data:
                global_before_data[band_name] = []
            global_before_data[band_name].extend(before_bands[band_name])
        
        # Process post-landslide data
        if band_name in after_bands and len(after_bands[band_name]) > 0:
            norm_data, _ = normalize_without_extremes(after_bands[band_name])
            area_after_norm[band_name] = norm_data
            # Add to global data
            if band_name not in global_after_data:
                global_after_data[band_name] = []
            global_after_data[band_name].extend(after_bands[band_name])
    
    # Plot histogram for this area
    output_filename = f"{area_name}_band_comparison_histogram.png"
    output_path = os.path.join(area_output_dir, output_filename)
    plot_band_histograms(
        area_name, 
        area_before_norm, 
        area_after_norm, 
        params['target_sentinel_bands'],
        output_path, 
        params,
        is_global=False
    )
    
    return True, before_descriptions, after_descriptions

def plot_band_histograms(title, before_data, after_data, target_bands, 
                        output_path, params, is_global=False):
    """Plot band comparison histograms with fixed layout to prevent bottom >= top error"""
    try:
        # Determine bands to plot (only target bands with data)
        valid_bands = [band for band in target_bands 
                      if (band in before_data and len(before_data[band]) > 0) or 
                      (band in after_data and len(after_data[band]) > 0)]
        
        if not valid_bands:
            print(f"Warning: No valid band data, cannot plot histogram - {title}")
            return False
        
        # Get subplot layout
        rows, cols = params['subplot_layout']
        
        # Adjust layout to fit number of bands
        if rows * cols < len(valid_bands):
            rows = (len(valid_bands) + cols - 1) // cols
        
        # 计算合适的图大小，确保有足够空间
        fig_width = cols * 7  # 每个列7单位宽度
        fig_height = rows * 6 + 3  # 每个行6单位高度 + 底部3单位图例空间
        
        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
        if rows * cols > 1:
            axes = axes.flatten()  # Convert to 1D array for easy indexing
        else:
            axes = [axes]  # Ensure list format
        
        # Plot comparison histogram for each band
        for i, band_name in enumerate(valid_bands):
            ax = axes[i]
            
            # Plot before-landslide histogram
            if band_name in before_data and len(before_data[band_name]) > 0:
                # Only add label to first subplot for global legend
                if i == 0:
                    ax.hist(before_data[band_name], bins=params['bins'], 
                            alpha=params['alpha'], color=params['before_color'], 
                            edgecolor='black', label='Pre-landslide')
                else:
                    ax.hist(before_data[band_name], bins=params['bins'], 
                            alpha=params['alpha'], color=params['before_color'], 
                            edgecolor='black')
            
            # Plot post-landslide histogram
            if band_name in after_data and len(after_data[band_name]) > 0:
                # Only add label to first subplot for global legend
                if i == 0:
                    ax.hist(after_data[band_name], bins=params['bins'], 
                            alpha=params['alpha'], color=params['after_color'], 
                            edgecolor='black', label='Post-landslide')
                else:
                    ax.hist(after_data[band_name], bins=params['bins'], 
                            alpha=params['alpha'], color=params['after_color'], 
                            edgecolor='black')
            
            # Set subplot title and labels with bold font
            ax.set_title(f'{band_name}', fontproperties=bold_title_font)
            ax.set_xlabel('Normalized Pixel Value', fontproperties=bold_label_font)
            ax.set_ylabel('Pixel Count', fontproperties=bold_label_font)
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # 调整刻度标签与轴标签之间的距离
            ax.tick_params(axis='x', pad=10)
            ax.tick_params(axis='y', pad=10)
        
        # Hide unused subplots
        for i in range(len(valid_bands), len(axes)):
            axes[i].axis('off')
        
        # 计算安全的底部边距，确保不会超过顶部
        # 使用固定比例而非动态计算，避免出现bottom >= top
        bottom_margin = 0.12  # 固定底部边距比例
        
        # Add global legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, 
                  bbox_to_anchor=(0.5, 0.02), 
                  frameon=True, prop=bold_legend_font)
        
        # 先调用tight_layout确保基本布局合理
        plt.tight_layout()
        
        # 再进行微调，使用安全的边距值
        # 只调整底部边距，其他保持自动计算的值
        plt.subplots_adjust(
            bottom=bottom_margin,
            hspace=0.4,  # 子图之间的垂直间距
            wspace=0.3   # 子图之间的水平间距
        )
        
        # Save image
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(output_path, dpi=params['dpi'], bbox_inches='tight')
        plt.close(fig)
        
        return True
    
    except Exception as e:
        print(f"Error plotting histogram: {title}")
        print(f"Error details: {str(e)}")
        traceback.print_exc()
        return False

def main():
    # 1. Get configuration file path from command line
    if len(sys.argv) < 2:
        print("Usage: python landslide_fixed_histograms.py <config_file_path>")
        print("Example: python landslide_fixed_histograms.py histogram_parameters.xml")
        sys.exit(1)
    config_path = sys.argv[1]
    
    # 2. Parse configuration parameters
    try:
        params = parse_xml_config(config_path)
    except Exception as e:
        print(f"Failed to parse configuration file: {str(e)}")
        sys.exit(1)
    
    # 3. Verify input and output directories
    if not os.path.exists(params['input_dir']):
        print(f"Error: Input directory does not exist - {params['input_dir']}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(params['histogram_dir'], exist_ok=True)
    
    # 4. Print configuration information
    print("===== Landslide Area Image Band Comparison Histogram Generator =====")
    print("Configuration parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # 5. Get all landslide area directories
    area_dirs = [
        d for d in os.listdir(params['input_dir']) 
        if os.path.isdir(os.path.join(params['input_dir'], d))
    ]
    
    if not area_dirs:
        print(f"Error: No area subdirectories found in input directory {params['input_dir']}")
        sys.exit(1)
    
    # 6. Process each area and collect global data
    start_time = time.time()
    # Use dictionaries to store global data, keyed by band name
    global_before_data = {}
    global_after_data = {}
    
    try:
        print("\nStarting processing of individual areas...")
        for area_name in tqdm(area_dirs, desc="Processing areas"):
            area_dir = os.path.join(params['input_dir'], area_name)
            tqdm.write(f"Processing area: {area_name}")
            
            process_single_area(
                area_dir, area_name, 
                params['histogram_dir'],
                global_before_data, 
                global_after_data,
                params
            )
        
        # 7. Normalize global data (removing extremes)
        print("\nNormalizing global dataset...")
        global_before_norm = {}
        global_after_norm = {}
        
        for band_name in params['target_sentinel_bands']:
            # Process before-landslide global data
            if band_name in global_before_data and len(global_before_data[band_name]) > 0:
                norm_data, _ = normalize_without_extremes(np.array(global_before_data[band_name]))
                global_before_norm[band_name] = norm_data
            
            # Process post-landslide global data
            if band_name in global_after_data and len(global_after_data[band_name]) > 0:
                norm_data, _ = normalize_without_extremes(np.array(global_after_data[band_name]))
                global_after_norm[band_name] = norm_data
        
        # 8. Plot global combined histogram
        global_output_dir = os.path.join(params['histogram_dir'], 'global_plot')
        os.makedirs(global_output_dir, exist_ok=True)
        
        output_filename = "all_areas_band_comparison_histogram.png"
        output_path = os.path.join(global_output_dir, output_filename)
        
        print(f"Generating global combined histogram: {output_filename}")
        plot_band_histograms(
            "Combined All Areas", 
            global_before_norm, 
            global_after_norm, 
            params['target_sentinel_bands'],
            output_path, 
            params,
            is_global=True
        )
        
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
    
    # 9. Calculate and display total processing time
    elapsed = time.time() - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\n===== All Processing Completed =====")
    print(f"Total time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    print(f"Individual area plots saved to: {os.path.join(params['histogram_dir'], 'area_plots')}")
    print(f"Global combined plot saved to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    main()
    