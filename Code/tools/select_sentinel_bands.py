import os
import sys
import xml.etree.ElementTree as ET
import rasterio
import warnings
from tqdm import tqdm  # 进度条显示
from typing import List, Dict
import shutil  # 用于创建目录结构

# 忽略非地理参考警告（部分Sentinel影像可能触发）
warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)


def parse_xml_config(xml_path: str) -> Dict:
    """
    解析共用的 tool_parameters.xml 配置文件，适配 image_cut.py 相同参数结构
    :param xml_path: XML配置文件路径
    :return: 配置参数字典
    """
    # 检查配置文件是否存在
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"配置文件不存在: {xml_path}")
    
    # 解析XML结构（适配 <startParam> 根节点和 <param> 子节点）
    tree = ET.parse(xml_path)
    root = tree.getroot()  # 根节点为 <startParam>
    
    params = {}
    # 遍历所有 <param> 子节点（忽略根节点属性）
    for param in root.findall('param'):
        # 提取参数名、值、类型（严格匹配 tool_parameters.xml 结构）
        name = param.find('name').text.strip()
        value = param.find('value').text.strip()
        param_type = param.find('type').text.strip()
        
        # 根据参数类型转换值（适配image_cut.py的参数类型）
        if param_type == 'int':
            params[name] = int(value)
        elif param_type == 'float':
            params[name] = float(value)
        elif param_type == 'str':
            params[name] = value
        elif param_type == 'list':
            # 解析列表（如 ['B2', 'B3', 'B4', 'B8']）
            params[name] = eval(value)  # 确保XML配置可靠，避免恶意代码
        elif param_type == 'folder':
            # 路径标准化（处理Windows/Linux路径分隔符）
            params[name] = os.path.normpath(value)
    
    # 设置默认参数（防止配置文件缺失关键项，与image_cut.py保持一致）
    params.setdefault('target_sentinel_bands', ['B2', 'B3', 'B4'])  # 默认RGB波段
    params.setdefault('input_dir', os.path.join(os.getcwd(), 'Data', 'origin'))  # 默认输入根目录
    params.setdefault('output_dir_no_cut', os.path.join(os.getcwd(), 'Data', 'band_select'))  # 默认输出根目录
    
    # 验证关键参数（确保数据处理所需路径和波段存在）
    required_params = ['input_dir', 'output_dir_no_cut', 'target_sentinel_bands']
    for req_param in required_params:
        if req_param not in params:
            raise ValueError(f"XML配置文件缺少必填参数: {req_param}")
    
    # 验证输入根目录是否存在
    if not os.path.exists(params['input_dir']):
        raise NotADirectoryError(f"输入根目录不存在: {params['input_dir']}")
    
    return params


def filter_sentinel_bands(input_path: str, output_path: str, target_bands: List[str]) -> bool:
    """
    筛选Sentinel-2影像的目标波段（核心逻辑不变，适配路径输出）
    :param input_path: 原始Sentinel影像路径（GeoTIFF格式）
    :param output_path: 筛选后影像保存路径
    :param target_bands: 目标波段列表（如["B2", "B3", "B4", "B8"]）
    :return: 处理成功返回True，失败返回False
    """
    try:
        # 读取原始影像
        with rasterio.open(input_path) as src:
            # 1. 获取原始影像的波段描述（Sentinel影像通常为"B1"/"B2"等）
            band_descriptions = src.descriptions if src.descriptions else []
            band_name_to_idx = {desc: idx + 1 for idx, desc in enumerate(band_descriptions)}  # 1-based索引
            
            # 2. 筛选目标波段对应的索引
            valid_indices = []  # 有效波段索引
            found_bands = []    # 实际找到的波段
            missing_bands = []  # 缺失的波段
            
            for target_band in target_bands:
                if target_band in band_name_to_idx:
                    valid_indices.append(band_name_to_idx[target_band])
                    found_bands.append(target_band)
                else:
                    missing_bands.append(target_band)
            
            # 3. 处理缺失波段警告
            if missing_bands:
                print(f"⚠️  影像 {os.path.basename(input_path)} 缺失目标波段: {', '.join(missing_bands)}")
            # 若未找到任何目标波段，保留所有波段（避免输出空文件）
            if not valid_indices:
                print(f"⚠️  影像 {os.path.basename(input_path)} 未找到任何目标波段，保留所有波段")
                valid_indices = list(range(1, src.count + 1))
                found_bands = band_descriptions or [f"band_{i}" for i in range(1, src.count + 1)]
            
            # 4. 读取目标波段数据
            filtered_data = src.read(valid_indices)  # 形状: (波段数, 高度, 宽度)
            
            # 5. 更新影像元数据（波段数、描述）
            meta = src.meta.copy()
            meta.update(
                count=len(valid_indices),  # 更新波段数
                dtype=filtered_data.dtype  # 保持原始数据类型（避免精度损失）
            )
        
        # 6. 确保输出目录存在（避免路径不存在报错）
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 7. 保存筛选后的影像
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(filtered_data)
            dst.descriptions = found_bands  # 写入新的波段描述
        
        return True
    
    except Exception as e:
        print(f"❌ 处理影像 {os.path.basename(input_path)} 失败: {str(e)}")
        return False


def batch_process_by_area(params: Dict) -> None:
    """
    按区域批量处理Sentinel影像（适配 input_dir/区域子目录/data 的结构）
    :param params: 配置参数字典（含input_dir、output_dir_no_cut等）
    """
    # 1. 获取输入根目录下的所有区域子目录（如 Area1、Area2）
    input_root = params['input_dir']
    area_dirs = [
        d for d in os.listdir(input_root) 
        if os.path.isdir(os.path.join(input_root, d))  # 仅保留目录
        and not d.startswith('.')  # 排除隐藏目录（如.git）
    ]
    
    if not area_dirs:
        print(f"ℹ️  输入根目录 {input_root} 下无有效区域子目录，程序退出")
        return
    
    # 2. 遍历每个区域子目录处理
    print(f"\n📊 发现 {len(area_dirs)} 个区域子目录，开始批量处理...")
    for area in tqdm(area_dirs, desc="区域处理进度"):
        # 构建当前区域的输入/输出路径（与image_cut.py结构对齐）
        area_input_path = os.path.join(input_root, area)  # 区域子目录路径
        area_data_input = os.path.join(area_input_path, 'data')  # 区域下的data文件夹（存放Sentinel影像）
        area_label_input = os.path.join(area_input_path, 'label')  # 区域下的label文件夹（无需处理，仅复制）
        
        # 输出路径：保持与输入相同的区域子目录结构
        area_output_path = os.path.join(params['output_dir_no_cut'], area)
        area_data_output = os.path.join(area_output_path, 'data')  # 筛选后的影像保存到data子目录
        area_label_output = os.path.join(area_output_path, 'label')  # 复制原始label到输出目录
        
        # 3. 检查当前区域的data文件夹是否存在
        if not os.path.exists(area_data_input):
            tqdm.write(f"⚠️  区域 {area} 缺少 data 文件夹，跳过该区域")
            continue
        
        # 4. 获取当前区域data文件夹下的Sentinel影像（灾前before/灾后after）
        tif_files = [
            f for f in os.listdir(area_data_input) 
            if f.lower().endswith(('.tif', '.tiff'))  # 仅处理TIFF格式
            and ('before' in f.lower() or 'after' in f.lower())  # 仅处理灾前/灾后影像
        ]
        
        if not tif_files:
            tqdm.write(f"⚠️  区域 {area} 的 data 文件夹下无灾前/灾后TIFF影像，跳过该区域")
            continue
        
        # 5. 复制当前区域的label文件夹（保持数据完整性，便于后续裁剪）
        if os.path.exists(area_label_input):
            os.makedirs(area_label_output, exist_ok=True)
            for label_file in os.listdir(area_label_input):
                src_label = os.path.join(area_label_input, label_file)
                dst_label = os.path.join(area_label_output, label_file)
                shutil.copy2(src_label, dst_label)  # 复制文件（保留元数据）
            tqdm.write(f"📋 已复制区域 {area} 的 label 文件夹")
        
        # 6. 处理当前区域的Sentinel影像（筛选波段）
        tqdm.write(f"\n🔍 开始处理区域 {area} 的 {len(tif_files)} 个影像文件:")
        for tif_file in tqdm(tif_files, desc=f"{area} 影像处理", leave=False):
            # 构建单个影像的输入/输出路径
            src_tif_path = os.path.join(area_data_input, tif_file)
            # 输出文件名：在原始文件名后添加 "_filtered" 标识（如 before.tif → before_filtered.tif）
            dst_tif_name = f"{os.path.splitext(tif_file)[0]}_filtered.tif"
            dst_tif_path = os.path.join(area_data_output, dst_tif_name)
            
            # 筛选波段并保存
            success = filter_sentinel_bands(
                input_path=src_tif_path,
                output_path=dst_tif_path,
                target_bands=params['target_sentinel_bands']
            )
            
            # 打印单个文件处理结果
            if success:
                tqdm.write(f"✅ 区域 {area}: {tif_file} → 保存至 {dst_tif_name}")
            else:
                tqdm.write(f"❌ 区域 {area}: 处理 {tif_file} 失败")
    
    # 处理完成提示
    print(f"\n🎉 所有区域处理完成！筛选后的影像保存至: {params['output_dir_no_cut']}")
    print(f"📂 输出结构说明:")
    print(f"   - 每个区域对应一个子目录（如 {params['output_dir_no_cut']}/Area1）")
    print(f"   - 每个区域子目录下包含 data（筛选后影像）和 label（原始标签）")


def main():
    """主函数：解析配置→批量处理→输出结果（与image_cut.py调用方式一致）"""
    # 1. 从命令行获取XML配置文件路径（与image_cut.py统一）
    if len(sys.argv) < 2:
        print("用法: python select_sentinel_bands.py <XML配置文件路径>")
        print("示例: python select_sentinel_bands.py tool_parameters.xml")
        sys.exit(1)
    config_path = sys.argv[1]
    
    # 2. 解析配置文件（复用tool_parameters.xml）
    try:
        print(f"📋 正在解析配置文件: {config_path}")
        params = parse_xml_config(config_path)
        
        # 打印关键配置信息（确认参数正确）
        print("\n🔧 关键配置参数确认:")
        print(f"  输入根目录（含区域子目录）: {params['input_dir']}")
        print(f"  输出根目录（保持区域结构）: {params['output_dir_no_cut']}")
        print(f"  目标筛选波段: {params['target_sentinel_bands']}")
        print(f"  待处理区域数量: {len([d for d in os.listdir(params['input_dir']) if os.path.isdir(os.path.join(params['input_dir'], d))])}")
    except Exception as e:
        print(f"❌ 配置文件解析失败: {str(e)}")
        sys.exit(1)
    
    # 3. 按区域批量处理影像
    try:
        batch_process_by_area(params)
    except Exception as e:
        print(f"❌ 批量处理异常: {str(e)}")
        traceback.print_exc()  # 打印详细错误栈
        sys.exit(1)


if __name__ == "__main__":
    # 补充导入traceback（用于打印详细错误信息）
    import traceback
    main()
    