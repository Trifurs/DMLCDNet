import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from collections import defaultdict
import shutil

# 设置字体为Times New Roman，进一步增大字体并加粗
plt.rcParams["font.family"] = ["Times New Roman", "serif"]
plt.rcParams["font.size"] = 45  # 字体进一步增大
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
plt.rcParams["axes.labelweight"] = "bold"  # 轴标签加粗
plt.rcParams['xtick.major.pad'] = 20  # 增加x轴标签与轴线的间隔
plt.rcParams['ytick.major.pad'] = 20  # 增加y轴标签与轴线的间隔

def get_place_name(filename):
    """从文件名中提取地名"""
    return filename.split('_')[0]

def count_colors(image_path):
    """统计图像中四种颜色的像素数量"""
    # 定义颜色映射
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
        
        # 计算比例
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
    """绘制横向条形图，增大字体，调整首末列边缘间隔"""
    # 分离Ours和其他子文件夹数据
    ours_stats = [stat for stat in image_stats if stat["folder_name"] == "Ours"]
    other_stats = [stat for stat in image_stats if stat["folder_name"] != "Ours"]
    
    # 合并数据：其他子文件夹 + Ours子文件夹（放在最后）
    sorted_stats = other_stats + ours_stats
    # 颠倒数据顺序（上下颠倒显示）
    sorted_stats = sorted_stats[::-1]
    total_items = len(sorted_stats)
    
    # 准备数据，处理超过2000的值
    MAX_VALUE = 2000
    folder_names = [stat["folder_name"] for stat in sorted_stats]
    false_negatives_actual = [stat["False Negative count"] for stat in sorted_stats]
    false_positives_actual = [stat["False Positive count"] for stat in sorted_stats]
    
    # 绘制用的值（超过2000的按2000处理）
    false_negatives_plotted = [min(val, MAX_VALUE) for val in false_negatives_actual]
    false_positives_plotted = [min(val, MAX_VALUE) for val in false_positives_actual]
    
    # 创建画布，根据项目数量调整高度
    fig_height = max(10, total_items * 1.2)  # 增加高度以适应更大的字体
    plt.figure(figsize=(14, fig_height))
    
    # 计算y位置，组间距为组内间距的1.5倍
    group_spacing = 1.4  # 增加组内间距以适应更大的字体和更宽的柱体
    inter_group_spacing = group_spacing * 1.5  # 组间间距
    y_positions = []
    current_pos = 0
    
    for i in range(total_items):
        # 第一列之前的间隔调整：第一个元素不额外增加间隔
        if i == 0:
            y_positions.append(current_pos)
            current_pos += group_spacing
        else:
            y_positions.append(current_pos)
            # 每3个项目增加组间间距（除了最后一组）
            if (i + 1) % 3 == 0 and i != total_items - 1:
                current_pos += inter_group_spacing
            else:
                current_pos += group_spacing
    
    y = np.array(y_positions)
    height = 0.6  # 进一步增加柱体宽度
    
    # 绘制横向条形图
    plt.barh(y - height/2, false_negatives_plotted, height, color='#FF0000')  # False Negative - red
    plt.barh(y + height/2, false_positives_plotted, height, color='#34ACFE')  # False Positive - light blue
    
    # 纵坐标标签处理：只标注第2,5,8...个位置（0-based索引：1,4,7...）
    # 逆时针旋转90度，确保单行显示
    labels = [''] * total_items
    for i in range(total_items):
        if i % 3 == 1:  # 第2,5,8...个位置（索引1,4,7...）
            # 移除可能导致换行的字符
            clean_name = folder_names[i].replace('\n', ' ').replace('\r', '')
            labels[i] = clean_name
    
    plt.yticks(y, labels, rotation=90, ha='center', va='center')  # 逆时针旋转90度
    
    # 添加数值标签，水平显示，超过2000的标注为">2000"
    def add_bar_labels(y_pos, actual_values, plotted_values, height):
        max_plotted = max(plotted_values) if plotted_values else 0
        for pos, actual_val, plotted_val in zip(y_pos, actual_values, plotted_values):
            if plotted_val > 0:
                # 确定显示的文本
                display_text = f'>{MAX_VALUE}' if actual_val > MAX_VALUE else f'{actual_val}'
                # 确定文本位置
                x_pos = plotted_val + max_plotted * 0.01 if max_plotted > 0 else plotted_val + 10
                plt.text(x_pos, pos, display_text, 
                        ha='left', va='center', fontsize=40,  # 增大标签字体
                        fontweight='bold')
    
    add_bar_labels(y - height/2, false_negatives_actual, false_negatives_plotted, height)
    add_bar_labels(y + height/2, false_positives_actual, false_positives_plotted, height)
    
    # 设置x轴范围，留出标签空间，同时控制首末列边缘间隔
    max_plotted_width = max(max(false_negatives_plotted), max(false_positives_plotted)) if (false_negatives_plotted or false_positives_plotted) else 0
    if max_plotted_width > 0:
        # 调整x轴范围，使右侧边缘间隔与柱体宽度一致
        right_margin = height  # 右侧间隔等于柱体宽度
        plt.xlim(0, max_plotted_width + right_margin)
    
    # 调整y轴范围，使上下边缘间隔与柱体宽度一致
    if total_items > 0:
        bottom_margin = height  # 底部间隔等于柱体宽度
        top_margin = height     # 顶部间隔等于柱体宽度
        plt.ylim(y[0] - bottom_margin, y[-1] + top_margin)
    
    # 调整布局，确保标签完整显示
    plt.tight_layout()
    
    # 保存图像
    plot_path = os.path.join(output_dir, f'{place}_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')  # 确保所有元素都在图中
    plt.close()
    
    return plot_path

def main(root_dir="predict_result", draw_result="statistics_result"):
    # 创建输出文件夹
    os.makedirs(draw_result, exist_ok=True)
    
    # 输入文件夹检查
    if not os.path.exists(root_dir):
        print(f"Error: Folder {root_dir} does not exist")
        return
    
    # 收集所有图像文件，按地名分组
    place_groups = defaultdict(list)  # {place: [(image_path, folder_name), ...]}
    
    # 遍历所有子文件夹
    for subdir, _, files in os.walk(root_dir):
        # 获取子文件夹名称
        rel_subdir = os.path.relpath(subdir, root_dir)
        folder_name = rel_subdir if rel_subdir != '.' else os.path.basename(root_dir)
        
        for file in files:
            if file.lower().endswith('.png'):
                img_rel_path = os.path.relpath(os.path.join(subdir, file), root_dir)
                place_name = get_place_name(file)
                place_groups[place_name].append((img_rel_path, folder_name))
    
    # 准备CSV文件路径
    output_csv = os.path.join(draw_result, "color_statistics.csv")
    
    # 存储每个地区的统计数据，用于绘图
    place_statistics = defaultdict(list)
    
    # 写入CSV文件
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # 按地名分组处理
        for place, image_info_list in place_groups.items():
            writer.writerow([f"Place: {place}"])
            
            # 写入表头（英文）
            writer.writerow([
                "Image Path", 
                "False Negative (count)", "False Negative (ratio)",
                "False Positive (count)", "False Positive (ratio)",
                "True Positive (count)", "True Positive (ratio)",
                "Background (count)", "Background (ratio)",
                "Total Pixels"
            ])
            
            # 处理该地名下的所有影像
            for img_rel_path, folder_name in image_info_list:
                full_path = os.path.join(root_dir, img_rel_path)
                color_stats, total_pixels = count_colors(full_path)
                
                if color_stats:
                    # 写入CSV
                    writer.writerow([
                        img_rel_path,
                        color_stats["False Negative"]["count"], color_stats["False Negative"]["ratio"],
                        color_stats["False Positive"]["count"], color_stats["False Positive"]["ratio"],
                        color_stats["True Positive"]["count"], color_stats["True Positive"]["ratio"],
                        color_stats["Background"]["count"], color_stats["Background"]["ratio"],
                        total_pixels
                    ])
                    
                    # 保存统计数据用于绘图
                    place_statistics[place].append({
                        "image_path": img_rel_path,
                        "folder_name": folder_name,
                        "False Negative count": color_stats["False Negative"]["count"],
                        "False Positive count": color_stats["False Positive"]["count"]
                    })
            
            writer.writerow([])
    
    print(f"Statistics completed, CSV file saved to {output_csv}")
    
    # 为每个地区绘制对比图
    for place, stats in place_statistics.items():
        plot_path = draw_place_comparison(place, stats, draw_result)
        print(f"{place} comparison chart saved to {plot_path}")
    
    print(f"All results output to {draw_result} folder")

if __name__ == "__main__":
    # 指定输入文件夹和结果输出文件夹
    img_dir = r"D:\lb\myCode\Landslide_detection\Result_test\draw\imgs"
    result_dir = r"D:\lb\myCode\Landslide_detection\Result_test\draw\results"
    
    # 清空并重建结果文件夹
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)
    
    main(root_dir=img_dir, draw_result=result_dir)
    