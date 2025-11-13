import os
import sys
import xml.etree.ElementTree as ET
import rasterio
import warnings
from tqdm import tqdm
from typing import List, Dict
import shutil

warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)


def parse_xml_config(xml_path: str) -> Dict:
    """
    Parse the shared tool_parameters.xml configuration file, adapting to the same parameter structure as image_cut.py
    :param xml_path: Path to the XML configuration file
    :return: Configuration parameter dictionary
    """
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Configuration file not found: {xml_path}")
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    params = {}
    for param in root.findall('param'):
        name = param.find('name').text.strip()
        value = param.find('value').text.strip()
        param_type = param.find('type').text.strip()
        
        if param_type == 'int':
            params[name] = int(value)
        elif param_type == 'float':
            params[name] = float(value)
        elif param_type == 'str':
            params[name] = value
        elif param_type == 'list':
            params[name] = eval(value)
        elif param_type == 'folder':
            params[name] = os.path.normpath(value)
    
    params.setdefault('target_sentinel_bands', ['B2', 'B3', 'B4'])
    params.setdefault('input_dir', os.path.join(os.getcwd(), 'Data', 'origin'))
    params.setdefault('output_dir_no_cut', os.path.join(os.getcwd(), 'Data', 'band_select'))
    
    required_params = ['input_dir', 'output_dir_no_cut', 'target_sentinel_bands']
    for req_param in required_params:
        if req_param not in params:
            raise ValueError(f"XML configuration file missing required parameter: {req_param}")
    
    if not os.path.exists(params['input_dir']):
        raise NotADirectoryError(f"Input root directory does not exist: {params['input_dir']}")
    
    return params


def filter_sentinel_bands(input_path: str, output_path: str, target_bands: List[str]) -> bool:
    """
    Filter target bands from Sentinel-2 images (core logic unchanged, adapted for path output)
    :param input_path: Path to original Sentinel image (GeoTIFF format)
    :param output_path: Path to save filtered image
    :param target_bands: List of target bands (e.g., ["B2", "B3", "B4", "B8"])
    :return: True if processing succeeds, False otherwise
    """
    try:
        with rasterio.open(input_path) as src:
            band_descriptions = src.descriptions if src.descriptions else []
            band_name_to_idx = {desc: idx + 1 for idx, desc in enumerate(band_descriptions)}
            
            valid_indices = []
            found_bands = []
            missing_bands = []
            
            for target_band in target_bands:
                if target_band in band_name_to_idx:
                    valid_indices.append(band_name_to_idx[target_band])
                    found_bands.append(target_band)
                else:
                    missing_bands.append(target_band)
            
            if missing_bands:
                print(f"⚠️  Image {os.path.basename(input_path)} missing target bands: {', '.join(missing_bands)}")
            if not valid_indices:
                print(f"⚠️  Image {os.path.basename(input_path)} found no target bands, keeping all bands")
                valid_indices = list(range(1, src.count + 1))
                found_bands = band_descriptions or [f"band_{i}" for i in range(1, src.count + 1)]
            
            filtered_data = src.read(valid_indices)
            
            meta = src.meta.copy()
            meta.update(
                count=len(valid_indices),
                dtype=filtered_data.dtype
            )
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with rasterio.open(output_path, 'w',** meta) as dst:
            dst.write(filtered_data)
            dst.descriptions = found_bands
        
        return True
    
    except Exception as e:
        print(f"❌ Failed to process image {os.path.basename(input_path)}: {str(e)}")
        return False


def batch_process_by_area(params: Dict) -> None:
    """
    Batch process Sentinel images by area (adapting to input_dir/area_subdir/data structure)
    :param params: Configuration parameter dictionary (including input_dir, output_dir_no_cut, etc.)
    """
    input_root = params['input_dir']
    area_dirs = [
        d for d in os.listdir(input_root)
        if os.path.isdir(os.path.join(input_root, d))
        and not d.startswith('.')
    ]
    
    if not area_dirs:
        print(f"ℹ️  No valid area subdirectories found in input root {input_root}, program exiting")
        return
    
    print(f"\n📊 Found {len(area_dirs)} area subdirectories, starting batch processing...")
    for area in tqdm(area_dirs, desc="Area processing progress"):
        area_input_path = os.path.join(input_root, area)
        area_data_input = os.path.join(area_input_path, 'data')
        area_label_input = os.path.join(area_input_path, 'label')
        
        area_output_path = os.path.join(params['output_dir_no_cut'], area)
        area_data_output = os.path.join(area_output_path, 'data')
        area_label_output = os.path.join(area_output_path, 'label')
        
        if not os.path.exists(area_data_input):
            tqdm.write(f"⚠️  Area {area} missing data folder, skipping area")
            continue
        
        tif_files = [
            f for f in os.listdir(area_data_input)
            if f.lower().endswith(('.tif', '.tiff'))
            and ('before' in f.lower() or 'after' in f.lower())
        ]
        
        if not tif_files:
            tqdm.write(f"⚠️  No before/after TIFF images in data folder for area {area}, skipping area")
            continue
        
        if os.path.exists(area_label_input):
            os.makedirs(area_label_output, exist_ok=True)
            for label_file in os.listdir(area_label_input):
                src_label = os.path.join(area_label_input, label_file)
                dst_label = os.path.join(area_label_output, label_file)
                shutil.copy2(src_label, dst_label)
            tqdm.write(f"📋 Copied label folder for area {area}")
        
        tqdm.write(f"\n🔍 Starting processing of {len(tif_files)} image files for area {area}:")
        for tif_file in tqdm(tif_files, desc=f"{area} image processing", leave=False):
            src_tif_path = os.path.join(area_data_input, tif_file)
            dst_tif_name = f"{os.path.splitext(tif_file)[0]}_filtered.tif"
            dst_tif_path = os.path.join(area_data_output, dst_tif_name)
            
            success = filter_sentinel_bands(
                input_path=src_tif_path,
                output_path=dst_tif_path,
                target_bands=params['target_sentinel_bands']
            )
            
            if success:
                tqdm.write(f"✅ Area {area}: {tif_file} → saved to {dst_tif_name}")
            else:
                tqdm.write(f"❌ Area {area}: failed to process {tif_file}")
    
    print(f"\n🎉 All areas processed! Filtered images saved to: {params['output_dir_no_cut']}")
    print(f"📂 Output structure explanation:")
    print(f"   - Each area corresponds to a subdirectory (e.g., {params['output_dir_no_cut']}/Area1)")
    print(f"   - Each area subdirectory contains data (filtered images) and label (original labels)")


def main():
    """Main function: parse configuration → batch process → output results (same calling method as image_cut.py)"""
    if len(sys.argv) < 2:
        print("Usage: python select_sentinel_bands.py <XML configuration file path>")
        print("Example: python select_sentinel_bands.py tool_parameters.xml")
        sys.exit(1)
    config_path = sys.argv[1]
    
    try:
        print(f"📋 Parsing configuration file: {config_path}")
        params = parse_xml_config(config_path)
        
        print("\n🔧 Key configuration parameters confirmed:")
        print(f"  Input root directory (contains area subdirectories): {params['input_dir']}")
        print(f"  Output root directory (maintains area structure): {params['output_dir_no_cut']}")
        print(f"  Target bands to filter: {params['target_sentinel_bands']}")
        print(f"  Number of areas to process: {len([d for d in os.listdir(params['input_dir']) if os.path.isdir(os.path.join(params['input_dir'], d))])}")
    except Exception as e:
        print(f"❌ Failed to parse configuration file: {str(e)}")
        sys.exit(1)
    
    try:
        batch_process_by_area(params)
    except Exception as e:
        print(f"❌ Batch processing error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import traceback
    main()
    
