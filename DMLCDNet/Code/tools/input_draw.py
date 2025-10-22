import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import rowcol
from sklearn.preprocessing import MinMaxScaler
import glob
from matplotlib.path import Path
from matplotlib.patches import PathPatch

# 设置字体为Times New Roman，并配置全局字体属性
plt.rcParams["font.family"] = ["Times New Roman", "serif"]
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
plt.rcParams['axes.linewidth'] = 1.2  # 坐标轴线条粗细

# 文字增大并加粗的全局设置（兼容旧版本matplotlib）
plt.rcParams['font.size'] = 12  # 基础字体大小增大
plt.rcParams['font.weight'] = 'bold'  # 基础字体加粗
plt.rcParams['axes.labelweight'] = 'bold'  # 轴标签加粗
plt.rcParams['axes.titleweight'] = 'bold'  # 标题加粗
plt.rcParams['xtick.labelsize'] = 11  # x轴刻度标签增大
plt.rcParams['ytick.labelsize'] = 11  # y轴刻度标签增大

def ensure_dir(path):
    """确保目录存在，不存在则创建"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"已创建目录: {path}")

def normalize_data(data):
    """使用Min-Max缩放将数据归一化到0-1范围"""
    if data.size == 0:
        return data
    # 如果数组所有值都相同，返回全零数组
    if np.all(data == data.flat[0]):
        return np.zeros_like(data)
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    # 防止数值极其接近导致后续问题
    if np.max(normalized) - np.min(normalized) < 1e-6:
        normalized += np.random.normal(0, 1e-6, size=normalized.shape)
    return normalized

def remove_nodata_values(data, threshold=-1e30):
    """移除无数据值和NaN值"""
    if data is None:
        return np.array([], dtype=float)
    data = data.flatten()
    data = data[~np.isnan(data)]
    valid_data = data[data > threshold]
    removed_count = len(data) - len(valid_data)
    if removed_count > 0:
        print(f"  已移除 {removed_count} 个无数据值")
    return valid_data

def format_factor_label(factor):
    """格式化因子标签"""
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
    从原始顶点构建一个竖直对齐的“右半小提琴”Path。
    - verts: N x 2 顶点数组
    - mid_x: 中轴线 x 值（箱线图中心）
    返回 matplotlib.path.Path 对象
    """
    if verts is None or len(verts) == 0:
        return None

    # 尝试提取右侧顶点（x >= mid_x）
    right_side = verts[verts[:, 0] >= mid_x]
    # 如果右侧顶点不足，用更宽松的阈值回退
    if len(right_side) < 3:
        tol = (np.max(verts[:, 0]) - np.min(verts[:, 0])) * 0.15
        right_side = verts[verts[:, 0] >= mid_x - tol]
    if len(right_side) < 3:
        # 仍然不足，直接返回一个窄条（从 mid_x 到 mid_x + min_width）
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

    # 按 y 排序
    right_side = right_side[np.argsort(right_side[:, 1])]
    y_vals = right_side[:, 1]
    x_vals = right_side[:, 0]

    # 有可能 x_vals 在不同 y 值存在抖动，取每个唯一 y 的最大 x
    unique_y, idx_first = np.unique(y_vals, return_index=True)
    # 为每个 unique_y 找对应的最大 x
    y_to_maxx = []
    for uy in unique_y:
        mask = np.isclose(y_vals, uy)
        y_to_maxx.append((uy, np.max(x_vals[mask])))

    y_grid = np.array([p[0] for p in y_to_maxx])
    x_grid = np.array([p[1] for p in y_to_maxx])

    # 计算实际宽度并限制
    raw_width = np.max(x_grid) - mid_x
    if raw_width <= 0:
        raw_width = min_width
    width = np.clip(raw_width, min_width, max_width)

    # 将 x_grid 线性缩放到 width 区间： mid_x + (x - mid_x) / raw_width * width
    if raw_width != 0:
        scaled_x = mid_x + (x_grid - mid_x) / raw_width * width
    else:
        scaled_x = mid_x + np.full_like(x_grid, width)

    # 构造新的右半边顶点（从底到顶）
    right_vertices = np.column_stack([scaled_x, y_grid])

    # 为了形成闭合的半边多边形，路径顺序为：
    # (mid_x, y_min) -> right_vertices (底->顶) -> (mid_x, y_max) -> (mid_x, y_min)
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
    绘制仅显示右半部分小提琴图的图表
    
    参数:
        data_root: 数据根目录
        factors: 要绘制的因子列表
        output_root: 输出图像目录
        mask: 掩膜阈值（0-1之间），默认为0表示不使用掩膜
        mask_dir: 掩膜文件所在目录，默认为指定路径
        nodata_threshold: 无数据值阈值
    """
    ensure_dir(output_root)

    # 定义颜色方案
    colors = [
        '#2ecc71', '#e74c3c', '#f39c12',
        '#9b59b6', '#3498db', '#1abc9c', '#34495e'
    ]

    # 遍历所有区域
    for area in os.listdir(data_root):
        area_path = os.path.join(data_root, area)
        if not os.path.isdir(area_path):
            continue

        print(f"正在处理区域: {area}")

        data_dir = os.path.join(area_path, 'data')
        if not os.path.exists(data_dir):
            print(f"警告: 区域 {area} 的数据目录不存在，已跳过")
            continue

        # 读取掩膜文件（如果需要）
        mask_src = None
        if mask > 0:
            mask_file = os.path.join(mask_dir, f"{area}.tif")
            if not os.path.exists(mask_file):
                print(f"警告: 区域 {area} 的掩膜文件 {mask_file} 不存在，将不使用掩膜处理")
            else:
                try:
                    mask_src = rasterio.open(mask_file)
                    print(f"已加载 {area} 的掩膜文件，使用阈值 {mask} 进行提取")
                except Exception as e:
                    print(f"读取 {area} 的掩膜文件时出错: {str(e)}，将不使用掩膜处理")
                    mask_src = None

        factor_data = {factor: [] for factor in factors}

        # 处理每个因子
        for factor in factors:
            tif_pattern = os.path.join(data_dir, f"{area}_{factor}.tif")
            tif_files = glob.glob(tif_pattern)

            if not tif_files:
                print(f"警告: 在 {data_dir} 中未找到 {area}_{factor}.tif 文件")
                continue

            # 可能存在多个同名文件（不同波段/时间），把所有栅格拼接为1维数组
            aggregated_values = []
            for tif_file in tif_files:
                try:
                    with rasterio.open(tif_file) as src:
                        # 读取数据
                        img_data = src.read(1)
                        
                        # 如果需要且有可用的掩膜数据，应用掩膜
                        if mask > 0 and mask_src is not None:
                            # 检查空间参考是否匹配
                            if not src.crs == mask_src.crs:
                                print(f"警告: {tif_file} 与掩膜文件空间参考不匹配，将不使用掩膜处理")
                            else:
                                # 创建数据的坐标网格
                                height, width = img_data.shape
                                cols, rows = np.meshgrid(np.arange(width), np.arange(height))
                                
                                # 将数据的行列号转换为地理坐标
                                xs, ys = rasterio.transform.xy(src.transform, rows, cols)
                                xs = np.array(xs)
                                ys = np.array(ys)
                                
                                # 将地理坐标转换为掩膜文件中的行列号
                                mask_rows, mask_cols = rowcol(
                                    mask_src.transform, 
                                    xs.flatten(), 
                                    ys.flatten()
                                )
                                
                                # 将掩膜行列号转换为整数索引
                                mask_rows = np.array(mask_rows, dtype=int)
                                mask_cols = np.array(mask_cols, dtype=int)
                                
                                # 确保索引在掩膜范围内
                                valid = (
                                    (mask_rows >= 0) & 
                                    (mask_rows < mask_src.height) & 
                                    (mask_cols >= 0) & 
                                    (mask_cols < mask_src.width)
                                )
                                
                                # 读取掩膜数据
                                mask_array = mask_src.read(1)
                                
                                # 获取有效掩膜值
                                valid_mask_values = mask_array[mask_rows[valid], mask_cols[valid]]
                                
                                # 创建数据的掩码（与数据形状相同）
                                data_mask = np.zeros(img_data.size, dtype=bool)
                                data_mask[valid] = (valid_mask_values >= mask)
                                data_mask = data_mask.reshape(img_data.shape)
                                
                                # 应用掩码
                                img_data = img_data[data_mask]
                                print(f"  应用空间掩膜后保留 {img_data.size} 个像素")
                        
                        # 移除无数据值
                        img_data = remove_nodata_values(img_data, threshold=nodata_threshold)
                        if img_data.size == 0:
                            print(f"警告: {tif_file} 在移除无数据值后为空，已跳过")
                            continue
                        aggregated_values.append(img_data)
                except Exception as e:
                    print(f"处理 {tif_file} 时出错: {str(e)}")

            if len(aggregated_values) == 0:
                continue

            all_vals = np.concatenate([v.flatten() for v in aggregated_values])
            if all_vals.size == 0:
                continue

            # 归一化
            try:
                normalized_data = normalize_data(all_vals)
                factor_data[factor] = normalized_data
            except Exception as e:
                print(f"归一化 {factor} 时出错: {e}")
                factor_data[factor] = all_vals  # 回退为原值

        # 关闭掩膜文件
        if mask_src is not None:
            mask_src.close()

        # 检查是否有可用数据
        has_data = any(len(v) > 0 for v in factor_data.values())
        if not has_data:
            print(f"警告: 区域 {area} 没有可用数据，已跳过绘图")
            continue

        # 准备绘图数据和格式化标签
        valid_factors = [f for f in factors if len(factor_data[f]) > 0]
        formatted_labels = [format_factor_label(f) for f in valid_factors]
        plot_data = [factor_data[f] for f in valid_factors]
        num_factors = len(valid_factors)

        # 创建图形 - 调整宽度以适应更小的间距
        plt.figure(figsize=(0.9 * num_factors, 3.5))

        # 为每个因子绘制组合图表
        for i, (data, factor) in enumerate(zip(plot_data, valid_factors)):
            pos = 0.6 * i + 1
            color = colors[i % len(colors)]

            # 1. 左侧：散点图
            if len(data) > 1000:
                sample_indices = np.random.choice(len(data), 1000, replace=False)
                scatter_data = data[sample_indices]
            else:
                scatter_data = data

            x_scatter = np.random.normal(pos - 0.12, 0.025, size=len(scatter_data))
            plt.scatter(x_scatter, scatter_data, s=3, color=color, alpha=0.5, edgecolors='none')

            # 2. 中间：箱线图 - 显示中位数线，保持"日"字形外观
            bp = plt.boxplot([data], positions=[pos], widths=0.06,
                           patch_artist=True, showfliers=False,
                           showmeans=False, showcaps=True)

            # 设置箱体样式 - 填充白色，确保覆盖下方内容
            for box in bp['boxes']:
                box.set_facecolor('white')  # 白色填充
                box.set_edgecolor('black')  # 黑色边框
                box.set_linewidth(1.2)      # 边框粗细
                box.set_zorder(3)           # 置于顶层，确保覆盖其他元素

            # 显示中位数线并设置样式
            for median in bp['medians']:
                median.set_visible(True)    # 确保中位数线可见
                median.set_color('black')   # 中位数线为黑色
                median.set_linewidth(1.2)   # 中位数线粗细
                median.set_zorder(4)        # 置于箱体上方，确保可见

            # 设置须线和帽线样式
            for part in ('whiskers', 'caps'):
                for element in bp[part]:
                    element.set_color('black')
                    element.set_linewidth(1.2)
                    element.set_zorder(3)  # 置于顶层

            # 3. 右侧：小提琴图（只显示右半部分）
            vp = plt.violinplot([data], positions=[pos], showmeans=False,
                              showmedians=False, showextrema=True, widths=0.40)

            # 对每个 body 做处理，替换为仅右半并竖直对齐
            for pc in vp['bodies']:
                try:
                    pc.set_facecolor(color)
                    pc.set_edgecolor('black')
                    pc.set_linewidth(1.2)
                    pc.set_alpha(0.7)
                    pc.set_zorder(2)  # 置于箱线图下方
                except Exception:
                    pass

                # 尝试获取原始顶点
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

                # 设置新的路径
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

            # 调整小提琴图的须线和极值位置
            for partname in ('cbars', 'cmins', 'cmaxes'):
                vp_part = vp[partname]
                vp_part.set_edgecolor('black')
                vp_part.set_linewidth(1.2)
                vp_part.set_zorder(1)  # 置于底层

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

        # 设置标题和标签
        plt.title(f'Distribution in {area}', fontsize=16, pad=10, weight='bold')
        plt.xlabel('Factors', fontsize=14, labelpad=8, weight='bold')
        plt.ylabel('Normalized Value (0-1)', fontsize=14, labelpad=5, weight='bold')

        # 设置坐标轴刻度和范围
        plt.xticks([0.6 * i + 1 for i in range(num_factors)], formatted_labels, 
                   fontsize=12, weight='bold')
        plt.xlim(0.5, 0.6 * (num_factors - 1) + 1.5)
        plt.ylim(-0.05, 1.05)
        plt.yticks(np.arange(0, 1.1, 0.2), fontsize=12, weight='bold')

        # 添加网格线
        plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)

        # 调整布局
        plt.tight_layout()

        # 保存图像
        output_path = os.path.join(output_root, f'{area}_factor_halfviolin.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"{area} 的自定义图表已保存至: {output_path}")

# 示例使用
if __name__ == "__main__":
    data_root = r"D:\lb\myCode\Landslide_detection\Data\origin"
    
    # # 地形因子（不使用掩膜）
    # factors1 = ['dem', 'slope', 'aspect', 'profile', 'plan', 'TWI']
    # output_root1 = r"D:\lb\myCode\Landslide_detection\Data\halfviolin\Terrain"
    # plot_area_factor_halfviolin(
    #     data_root=data_root,
    #     factors=factors1,
    #     output_root=output_root1,
    #     mask=0,  # 不使用掩膜
    #     nodata_threshold=-1e30
    # )
    
    # InSAR因子（使用掩膜）
    factors2 = ['last', '1m', '2m', '3m', '6m', '1y']
    output_root2 = r"D:\lb\myCode\Landslide_detection\Data\halfviolin\InSAR"
    plot_area_factor_halfviolin(
        data_root=data_root,
        factors=factors2,
        output_root=output_root2,
        mask=0.5,  # 使用掩膜，阈值
        mask_dir=r'D:\lb\myCode\Landslide_detection\Data\SBAS_Con',  # 掩膜目录
        nodata_threshold=-1e30
    )
