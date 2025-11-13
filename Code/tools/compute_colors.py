import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from collections import defaultdict
import shutil

plt.rcParams["font.family"] = ["Times New Roman", "serif"]
plt.rcParams["font.size"] = 45
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['xtick.major.pad'] = 20
plt.rcParams['ytick.major.pad'] = 20

def get_place_name(filename):
    """Extract place name from filename"""
    return filename.split('_')[0]

def count_colors(image_path):
    """Count pixel quantities for four colors in the image"""
    color_map = {
        (255, 0, 0): "False Negative",
        (52, 172, 254): "False Positive",
        (255, 255, 255): "True Positive",
        (0, 0, 0): "Background"
    }
    
    try:
        with Image.open(image_path) as img:
            img_rgb = img.convert('RGB')
            img_array = np.array(img_rgb)
            width, height = img_rgb.size
            total_pixels = width * height
            
            color_counts = {}
            for color, name in color_map.items():
                mask = np.all(img_array == color, axis=-1)
                color_counts[name] = np.sum(mask)
        
        color_stats = {}
        for color_name in color_map.values():
            count = color_counts.get(color_name, 0)
            ratio = count / total_pixels if total_pixels > 0 else 0
            color_stats[color_name] = {
                "count": count,
                "ratio": round(ratio, 6)
            }
        
        return color_stats, total_pixels
        
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None, 0

def draw_place_comparison(place, image_stats, output_dir):
    """Draw horizontal bar chart with adjusted font size and margins"""
    ours_stats = [stat for stat in image_stats if stat["folder_name"] == "Ours"]
    other_stats = [stat for stat in image_stats if stat["folder_name"] != "Ours"]
    
    sorted_stats = other_stats + ours_stats
    sorted_stats = sorted_stats[::-1]
    total_items = len(sorted_stats)
    
    MAX_VALUE = 2000
    folder_names = [stat["folder_name"] for stat in sorted_stats]
    false_negatives_actual = [stat["False Negative count"] for stat in sorted_stats]
    false_positives_actual = [stat["False Positive count"] for stat in sorted_stats]
    
    false_negatives_plotted = [min(val, MAX_VALUE) for val in false_negatives_actual]
    false_positives_plotted = [min(val, MAX_VALUE) for val in false_positives_actual]
    
    fig_height = max(10, total_items * 1.2)
    plt.figure(figsize=(14, fig_height))
    
    group_spacing = 1.4
    inter_group_spacing = group_spacing * 1.5
    y_positions = []
    current_pos = 0
    
    for i in range(total_items):
        if i == 0:
            y_positions.append(current_pos)
            current_pos += group_spacing
        else:
            y_positions.append(current_pos)
            if (i + 1) % 3 == 0 and i != total_items - 1:
                current_pos += inter_group_spacing
            else:
                current_pos += group_spacing
    
    y = np.array(y_positions)
    height = 0.6
    
    plt.barh(y - height/2, false_negatives_plotted, height, color='#FF0000')
    plt.barh(y + height/2, false_positives_plotted, height, color='#34ACFE')
    
    labels = [''] * total_items
    for i in range(total_items):
        if i % 3 == 1:
            clean_name = folder_names[i].replace('\n', ' ').replace('\r', '')
            labels[i] = clean_name
    
    plt.yticks(y, labels, rotation=90, ha='center', va='center')
    
    def add_bar_labels(y_pos, actual_values, plotted_values, height):
        max_plotted = max(plotted_values) if plotted_values else 0
        for pos, actual_val, plotted_val in zip(y_pos, actual_values, plotted_values):
            if plotted_val > 0:
                display_text = f'>{MAX_VALUE}' if actual_val > MAX_VALUE else f'{actual_val}'
                x_pos = plotted_val + max_plotted * 0.01 if max_plotted > 0 else plotted_val + 10
                plt.text(x_pos, pos, display_text, 
                        ha='left', va='center', fontsize=40,
                        fontweight='bold')
    
    add_bar_labels(y - height/2, false_negatives_actual, false_negatives_plotted, height)
    add_bar_labels(y + height/2, false_positives_actual, false_positives_plotted, height)
    
    max_plotted_width = max(max(false_negatives_plotted), max(false_positives_plotted)) if (false_negatives_plotted or false_positives_plotted) else 0
    if max_plotted_width > 0:
        right_margin = height
        plt.xlim(0, max_plotted_width + right_margin)
    
    if total_items > 0:
        bottom_margin = height
        top_margin = height
        plt.ylim(y[0] - bottom_margin, y[-1] + top_margin)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f'{place}_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def main(root_dir="predict_result", draw_result="statistics_result"):
    os.makedirs(draw_result, exist_ok=True)
    
    if not os.path.exists(root_dir):
        print(f"Error: Folder {root_dir} does not exist")
        return
    
    place_groups = defaultdict(list)
    
    for subdir, _, files in os.walk(root_dir):
        rel_subdir = os.path.relpath(subdir, root_dir)
        folder_name = rel_subdir if rel_subdir != '.' else os.path.basename(root_dir)
        
        for file in files:
            if file.lower().endswith('.png'):
                img_rel_path = os.path.relpath(os.path.join(subdir, file), root_dir)
                place_name = get_place_name(file)
                place_groups[place_name].append((img_rel_path, folder_name))
    
    output_csv = os.path.join(draw_result, "color_statistics.csv")
    
    place_statistics = defaultdict(list)
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        for place, image_info_list in place_groups.items():
            writer.writerow([f"Place: {place}"])
            
            writer.writerow([
                "Image Path", 
                "False Negative (count)", "False Negative (ratio)",
                "False Positive (count)", "False Positive (ratio)",
                "True Positive (count)", "True Positive (ratio)",
                "Background (count)", "Background (ratio)",
                "Total Pixels"
            ])
            
            for img_rel_path, folder_name in image_info_list:
                full_path = os.path.join(root_dir, img_rel_path)
                color_stats, total_pixels = count_colors(full_path)
                
                if color_stats:
                    writer.writerow([
                        img_rel_path,
                        color_stats["False Negative"]["count"], color_stats["False Negative"]["ratio"],
                        color_stats["False Positive"]["count"], color_stats["False Positive"]["ratio"],
                        color_stats["True Positive"]["count"], color_stats["True Positive"]["ratio"],
                        color_stats["Background"]["count"], color_stats["Background"]["ratio"],
                        total_pixels
                    ])
                    
                    place_statistics[place].append({
                        "image_path": img_rel_path,
                        "folder_name": folder_name,
                        "False Negative count": color_stats["False Negative"]["count"],
                        "False Positive count": color_stats["False Positive"]["count"]
                    })
            
            writer.writerow([])
    
    print(f"Statistics completed, CSV file saved to {output_csv}")
    
    for place, stats in place_statistics.items():
        plot_path = draw_place_comparison(place, stats, draw_result)
        print(f"{place} comparison chart saved to {plot_path}")
    
    print(f"All results output to {draw_result} folder")

if __name__ == "__main__":
    img_dir = r"D:\lb\myCode\Landslide_detection\Result_test\draw\imgs"
    result_dir = r"D:\lb\myCode\Landslide_detection\Result_test\draw\results"
    
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)
    
    main(root_dir=img_dir, draw_result=result_dir)
    
