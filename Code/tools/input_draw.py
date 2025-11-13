import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import rowcol
from sklearn.preprocessing import MinMaxScaler
import glob
from matplotlib.path import Path
from matplotlib.patches import PathPatch

plt.rcParams["font.family"] = ["Times New Roman", "serif"]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.2

plt.rcParams['font.size'] = 12
plt.rcParamsParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

def ensure_dir(path):
    """Ensure directory exists, create if it doesn't"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created: {path}")

def normalize_data(data):
    """Normalize data to 0-1 range using Min-Max scaling"""
    if data.size == 0:
        return data
    if np.all(data == data.flat[0]):
        return np.zeros_like(data)
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    if np.max(normalized) - np.min(normalized) < 1e-6:
        normalized += np.random.normal(0, 1e-6, size=normalized.shape)
    return normalized

def remove_nodata_values(data, threshold=-1e30):
    """Remove no-data values and NaN values"""
    if data is None:
        return np.array([], dtype=float)
    data = data.flatten()
    data = data[~np.isnan(data)]
    valid_data = data[data > threshold]
    removed_count = len(data) - len(valid_data)
    if removed_count > 0:
        print(f"  Removed {removed_count} no-data values")
    return valid_data

def format_factor_label(factor):
    """Format factor labels"""
    if factor == 'last':
        return 'Post'
    if factor is None:
        return ''
    if factor.lower() in ['dem', 'twi']:
        return factor.upper()
    if len(factor) == 0:
        return factor
    if not factor[0].isalpha():
        return factor
    return factor[0].upper() + factor[1:]

def build_right_half_path_from_vertices(verts, mid_x, min_width=0.03, max_width=0.12):
    """
    Build a vertically aligned "right half violin" Path from original vertices.
    - verts: N x 2 vertex array
    - mid_x: x value of central axis (boxplot center)
    Returns matplotlib.path.Path object
    """
    if verts is None or len(verts) == 0:
        return None

    right_side = verts[verts[:, 0] >= mid_x]
    if len(right_side) < 3:
        tol = (np.max(verts[:, 0]) - np.min(verts[:, 0])) * 0.15
        right_side = verts[verts[:, 0] >= mid_x - tol]
    if len(right_side) < 3:
        y_min = np.min(verts[:, 1])
        y_max = np.max(verts[:, 1])
        narrow = np.array([
            [mid_x, y_min],
            [mid_x + min_width, y_min],
            [mid_x + min_width, y_max],
            [mid_x, y_max],
            [mid_x, y_min]
        ])
        return Path(narrow, closed=True)

    right_side = right_side[np.argsort(right_side[:, 1])]
    y_vals = right_side[:, 1]
    x_vals = right_side[:, 0]

    unique_y, idx_first = np.unique(y_vals, return_index=True)
    y_to_maxx = []
    for uy in unique_y:
        mask = np.isclose(y_vals, uy)
        y_to_maxx.append((uy, np.max(x_vals[mask])))

    y_grid = np.array([p[0] for p in y_to_maxx])
    x_grid = np.array([p[1] for p in y_to_maxx])

    raw_width = np.max(x_grid) - mid_x
    if raw_width <= 0:
        raw_width = min_width
    width = np.clip(raw_width, min_width, max_width)

    if raw_width != 0:
        scaled_x = mid_x + (x_grid - mid_x) / raw_width * width
    else:
        scaled_x = mid_x + np.full_like(x_grid, width)

    right_vertices = np.column_stack([scaled_x, y_grid])

    y_min = y_grid[0]
    y_max = y_grid[-1]
    new_vertices = np.vstack([
        [mid_x, y_min],
        right_vertices,
        [mid_x, y_max],
        [mid_x, y_min]
    ])

    return Path(new_vertices, closed=True)

def plot_area_factor_halfviolin(data_root, factors, output_root, mask=0, 
                               mask_dir=r'D:\lb\myCode\Landslide_detection\Data\SBAS_Con',
                               nodata_threshold=-1e30):
    """
    Plot charts showing only the right half of violin plots
    
    Parameters:
        data_root: Data root directory
        factors: List of factors to plot
        output_root: Output image directory
        mask: Mask threshold (between 0-1), default 0 means no mask
        mask_dir: Directory containing mask files, default to specified path
        nodata_threshold: No-data value threshold
    """
    ensure_dir(output_root)

    colors = [
        '#2ecc71', '#e74c3c', '#f39c12',
        '#9b59b6', '#3498db', '#1abc9c', '#34495e'
    ]

    for area in os.listdir(data_root):
        area_path = os.path.join(data_root, area)
        if not os.path.isdir(area_path):
            continue

        print(f"Processing area: {area}")

        data_dir = os.path.join(area_path, 'data')
        if not os.path.exists(data_dir):
            print(f"Warning: Data directory for area {area} does not exist, skipped")
            continue

        mask_src = None
        if mask > 0:
            mask_file = os.path.join(mask_dir, f"{area}.tif")
            if not os.path.exists(mask_file):
                print(f"Warning: Mask file {mask_file} for area {area} does not exist, will not use mask")
            else:
                try:
                    mask_src = rasterio.open(mask_file)
                    print(f"Loaded mask file for {area}, using threshold {mask} for extraction")
                except Exception as e:
                    print(f"Error reading mask file for {area}: {str(e)}, will not use mask")
                    mask_src = None

        factor_data = {factor: [] for factor in factors}

        for factor in factors:
            tif_pattern = os.path.join(data_dir, f"{area}_{factor}.tif")
            tif_files = glob.glob(tif_pattern)

            if not tif_files:
                print(f"Warning: No {area}_{factor}.tif files found in {data_dir}")
                continue

            aggregated_values = []
            for tif_file in tif_files:
                try:
                    with rasterio.open(tif_file) as src:
                        img_data = src.read(1)
                        
                        if mask > 0 and mask_src is not None:
                            if not src.crs == mask_src.crs:
                                print(f"Warning: Spatial reference of {tif_file} does not match mask file, will not use mask")
                            else:
                                height, width = img_data.shape
                                cols, rows = np.meshgrid(np.arange(width), np.arange(height))
                                
                                xs, ys = rasterio.transform.xy(src.transform, rows, cols)
                                xs = np.array(xs)
                                ys = np.array(ys)
                                
                                mask_rows, mask_cols = rowcol(
                                    mask_src.transform, 
                                    xs.flatten(), 
                                    ys.flatten()
                                )
                                
                                mask_rows = np.array(mask_rows, dtype=int)
                                mask_cols = np.array(mask_cols, dtype=int)
                                
                                valid = (
                                    (mask_rows >= 0) & 
                                    (mask_rows < mask_src.height) & 
                                    (mask_cols >= 0) & 
                                    (mask_cols < mask_src.width)
                                )
                                
                                mask_array = mask_src.read(1)
                                
                                valid_mask_values = mask_array[mask_rows[valid], mask_cols[valid]]
                                
                                data_mask = np.zeros(img_data.size, dtype=bool)
                                data_mask[valid] = (valid_mask_values >= mask)
                                data_mask = data_mask.reshape(img_data.shape)
                                
                                img_data = img_data[data_mask]
                                print(f"  {img_data.size} pixels retained after applying spatial mask")
                        
                        img_data = remove_nodata_values(img_data, threshold=nodata_threshold)
                        if img_data.size == 0:
                            print(f"Warning: {tif_file} is empty after removing no-data values, skipped")
                            continue
                        aggregated_values.append(img_data)
                except Exception as e:
                    print(f"Error processing {tif_file}: {str(e)}")

            if len(aggregated_values) == 0:
                continue

            all_vals = np.concatenate([v.flatten() for v in aggregated_values])
            if all_vals.size == 0:
                continue

            try:
                normalized_data = normalize_data(all_vals)
                factor_data[factor] = normalized_data
            except Exception as e:
                print(f"Error normalizing {factor}: {e}")
                factor_data[factor] = all_vals

        if mask_src is not None:
            mask_src.close()

        has_data = any(len(v) > 0 for v in factor_data.values())
        if not has_data:
            print(f"Warning: No available data for area {area}, plotting skipped")
            continue

        valid_factors = [f for f in factors if len(factor_data[f]) > 0]
        formatted_labels = [format_factor_label(f) for f in valid_factors]
        plot_data = [factor_data[f] for f in valid_factors]
        num_factors = len(valid_factors)

        plt.figure(figsize=(0.9 * num_factors, 3.5))

        for i, (data, factor) in enumerate(zip(plot_data, valid_factors)):
            pos = 0.6 * i + 1
            color = colors[i % len(colors)]

            if len(data) > 1000:
                sample_indices = np.random.choice(len(data), 1000, replace=False)
                scatter_data = data[sample_indices]
            else:
                scatter_data = data

            x_scatter = np.random.normal(pos - 0.12, 0.025, size=len(scatter_data))
            plt.scatter(x_scatter, scatter_data, s=3, color=color, alpha=0.5, edgecolors='none')

            bp = plt.boxplot([data], positions=[pos], widths=0.06,
                           patch_artist=True, showfliers=False,
                           showmeans=False, showcaps=True)

            for box in bp['boxes']:
                box.set_facecolor('white')
                box.set_edgecolor('black')
                box.set_linewidth(1.2)
                box.set_zorder(3)

            for median in bp['medians']:
                median.set_visible(True)
                median.set_color('black')
                median.set_linewidth(1.2)
                median.set_zorder(4)

            for part in ('whiskers', 'caps'):
                for element in bp[part]:
                    element.set_color('black')
                    element.set_linewidth(1.2)
                    element.set_zorder(3)

            vp = plt.violinplot([data], positions=[pos], showmeans=False,
                              showmedians=False, showextrema=True, widths=0.40)

            for pc in vp['bodies']:
                try:
                    pc.set_facecolor(color)
                    pc.set_edgecolor('black')
                    pc.set_linewidth(1.2)
                    pc.set_alpha(0.7)
                    pc.set_zorder(2)
                except Exception:
                    pass

                verts = None
                if hasattr(pc, 'get_paths'):
                    paths = pc.get_paths()
                    if paths and len(paths) > 0:
                        try:
                            verts = paths[0].vertices
                        except Exception:
                            verts = None
                if verts is None and hasattr(pc, 'get_verts'):
                    try:
                        verts = pc.get_verts()
                    except Exception:
                        verts = None

                if verts is None or len(verts) == 0:
                    continue

                mid_x = pos
                new_path = build_right_half_path_from_vertices(verts, mid_x, min_width=0.03, max_width=0.12)
                if new_path is None:
                    continue

                if hasattr(pc, 'set_paths'):
                    try:
                        pc.set_paths([new_path])
                    except Exception:
                        if hasattr(pc, 'set_verts'):
                            try:
                                pc.set_verts([new_path.vertices])
                            except Exception:
                                pass
                elif hasattr(pc, 'set_verts'):
                    try:
                        pc.set_verts([new_path.vertices])
                    except Exception:
                        pass

            for partname in ('cbars', 'cmins', 'cmaxes'):
                vp_part = vp[partname]
                vp_part.set_edgecolor('black')
                vp_part.set_linewidth(1.2)
                vp_part.set_zorder(1)

                if partname == 'cbars':
                    try:
                        segments = vp_part.get_segments()
                        new_segments = []
                        for seg in segments:
                            new_seg = [[pos, seg[0, 1]], [pos, seg[1, 1]]]
                            new_segments.append(new_seg)
                        vp_part.set_segments(new_segments)
                    except Exception:
                        pass

        plt.title(f'Distribution in {area}', fontsize=16, pad=10, weight='bold')
        plt.xlabel('Factors', fontsize=14, labelpad=8, weight='bold')
        plt.ylabel('Normalized Value (0-1)', fontsize=14, labelpad=5, weight='bold')

        plt.xticks([0.6 * i + 1 for i in range(num_factors)], formatted_labels, 
                   fontsize=12, weight='bold')
        plt.xlim(0.5, 0.6 * (num_factors - 1) + 1.5)
        plt.ylim(-0.05, 1.05)
        plt.yticks(np.arange(0, 1.1, 0.2), fontsize=12, weight='bold')

        plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)

        plt.tight_layout()

        output_path = os.path.join(output_root, f'{area}_factor_halfviolin.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Custom chart for {area} saved to: {output_path}")

if __name__ == "__main__":
    data_root = r"D:\lb\myCode\Landslide_detection\Data\origin"
    
    # factors1 = ['dem', 'slope', 'aspect', 'profile', 'plan', 'TWI']
    # output_root1 = r"D:\lb\myCode\Landslide_detection\Data\halfviolin\Terrain"
    # plot_area_factor_halfviolin(
    #     data_root=data_root,
    #     factors=factors1,
    #     output_root=output_root1,
    #     mask=0,
    #     nodata_threshold=-1e30
    # )
    
    factors2 = ['last', '1m', '2m', '3m', '6m', '1y']
    output_root2 = r"D:\lb\myCode\Landslide_detection\Data\halfviolin\InSAR"
    plot_area_factor_halfviolin(
        data_root=data_root,
        factors=factors2,
        output_root=output_root2,
        mask=0.5,
        mask_dir=r'D:\lb\myCode\Landslide_detection\Data\SBAS_Con',
        nodata_threshold=-1e30
    )
    
