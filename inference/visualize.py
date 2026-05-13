import os
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 配置参数 ---
# JSON 结果文件所在的目录
ROOT_DIR = "/data/yuzhiyuan/outputs_LLM-CL/Llama-3.2-1B-Instruct/cl/upcycle/grouped_3of6_perlayer_full/predictions_single"
# RESULTS_DIR = os.path.join(ROOT_DIR, "predictions")
RESULTS_DIR = ROOT_DIR
# 输出图片的文件名
# OUTPUT_IMAGE_FILE = "/data/yuzhiyuan/outputs_LLM-CL/naive_full/evaluation_matrix.png"
OUTPUT_IMAGE_FILE = os.path.join(ROOT_DIR, "evaluation_matrix.png")
# 输出 Excel 的文件名
# OUTPUT_EXCEL_FILE = "/data/yuzhiyuan/outputs_LLM-CL/naive_full/evaluation_matrix.xlsx"
OUTPUT_EXCEL_FILE = os.path.join(ROOT_DIR, "evaluation_matrix.xlsx")


def extract_metric_value(eval_metrics, task_name=None):
    """
    从评估指标字典中提取一个主要数值用于热力图映射。
    优先级：sari (for 20Minuten) > rouge-L (for MeetingBank) > accuracy > bleu-4 > rouge-L > bleu-1 > similarity
    
    Args:
        eval_metrics (dict): 评估指标字典
        task_name (str): 任务名称，用于特殊处理（如 20Minuten 优先使用 sari, MeetingBank 优先使用 rouge-L）
    
    Returns:
        float or None: 提取的数值，如果找不到合适的指标则返回 None
    """
    if not eval_metrics:
        return None
    
    # 对于 20Minuten 任务，优先使用 sari
    if task_name and ('20Minuten' in task_name or '20minuten' in task_name.lower()):
        if 'sari' in eval_metrics:
            value = eval_metrics['sari']
            # 处理嵌套结构（如 sari 可能是 [{'sari': 39.3}]）
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], dict):
                    # 尝试从字典中提取 'sari' 键
                    if 'sari' in value[0]:
                        value = value[0]['sari']
                    else:
                        # 如果字典中没有 'sari' 键，尝试取第一个数值
                        for sub_key, sub_value in value[0].items():
                            if isinstance(sub_value, (int, float)):
                                value = sub_value
                                break
            # sari 值需要归一化到 0-1（假设范围在0-100）
            if isinstance(value, (int, float)):
                return min(value / 100.0, 1.0) if value > 0 else 0.0
    
    # 对于 MeetingBank 任务，优先使用 rouge-L
    if task_name and ('MeetingBank' in task_name or 'meetingbank' in task_name.lower() or 'meeting' in task_name.lower()):
        if 'rouge-L' in eval_metrics:
            value = eval_metrics['rouge-L']
            # rouge-L 通常在 0-1 范围内
            if isinstance(value, (int, float)):
                return float(value)
    
    # 优先级顺序提取指标
    priority_keys = ['accuracy', 'bleu-4', 'rouge-L', 'bleu-1', 'similarity', 'sari']
    
    for key in priority_keys:
        if key in eval_metrics:
            value = eval_metrics[key]
            # 处理嵌套结构（如 sari 可能是 [{'sari': 39.3}]）
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], dict) and key in value[0]:
                    value = value[0][key]
                elif isinstance(value[0], dict):
                    # 尝试直接取字典中的第一个值
                    for sub_key, sub_value in value[0].items():
                        if isinstance(sub_value, (int, float)):
                            value = sub_value
                            break
            # similarity 需要除以100归一化到 0-1 范围
            if key == 'similarity' and isinstance(value, (int, float)):
                return min(value / 100.0, 1.0) if value > 0 else 0.0
            # sari 值需要归一化到 0-1（假设范围在0-100）
            if key == 'sari' and isinstance(value, (int, float)):
                return min(value / 100.0, 1.0) if value > 0 else 0.0
            # 其他指标通常在0-1范围内
            if isinstance(value, (int, float)):
                return float(value)
    
    # 如果没有找到优先指标，尝试提取第一个数值
    for key, value in eval_metrics.items():
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, list) and len(value) > 0:
            if isinstance(value[0], dict):
                for sub_key, sub_value in value[0].items():
                    if isinstance(sub_value, (int, float)):
                        return float(sub_value)
    
    return None


def parse_results(directory):
    """
    解析指定目录下的所有结果 JSON 文件。

    Args:
        directory (str): 包含 JSON 文件的目录路径。

    Returns:
        pd.DataFrame: 一个包含解析后数据的 Pandas DataFrame。
                      列包括 'round', 'task_id', 'task_name', 'metrics_str', 'metric_value'。
    """
    records = []
    # 正则表达式，用于从文件名中提取信息
    # e.g., results-4-3-Py150.json -> round=4, task_id=3, task_name=Py150
    pattern = re.compile(r"results-(\d+)-(\d+)-(.+)\.json")

    print(f"🔍 开始扫描目录: {directory}")
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            round_num = int(match.group(1))
            task_id = int(match.group(2))
            task_name = match.group(3)

            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 获取 'eval' 字典，如果不存在则为空字典
                eval_metrics = data.get('eval', {})
                
                # 将评估指标字典格式化为多行字符串
                # e.g., {'accuracy': 0.85, 'f1': 0.92} -> "accuracy: 0.85\nf1: 0.92"
                metrics_str = "\n".join([f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}" 
                                         for key, value in eval_metrics.items()])
                
                if not metrics_str:
                    metrics_str = "N/A" # 如果没有评估指标
                
                # 提取用于热力图的数值
                metric_value = extract_metric_value(eval_metrics, task_name)

                records.append({
                    "round": round_num,
                    "task_id": task_id,
                    "task_name": task_name,
                    "metrics_str": metrics_str,
                    "metric_value": metric_value
                })
            except (json.JSONDecodeError, IOError) as e:
                print(f"⚠️ 警告: 无法读取或解析文件 {filename}: {e}")
    
    if not records:
        print("❌ 错误: 未找到任何匹配的结果文件。请检查目录和文件名格式。")
        return pd.DataFrame()

    print(f"✅ 成功解析 {len(records)} 个文件。")
    return pd.DataFrame(records)


def create_visualization(df):
    """
    根据解析后的数据创建并保存可视化矩阵图片。
    """
    if df.empty:
        return

    # --- 准备数据透视表 ---
    # 1. 确定任务的顺序 (按 task_id 排序)
    task_order = df.sort_values('task_id').drop_duplicates('task_name')['task_name'].tolist()
    # 2. 确定轮次的顺序
    round_order = sorted(df['round'].unique())

    # 3. 创建文本数据透视表（用于注释）
    pivot_text = df.pivot_table(
        index='task_name', 
        columns='round', 
        values='metrics_str', 
        aggfunc='first'
    )

    # 4. 创建数值数据透视表（用于热力图颜色映射）
    pivot_values = df.pivot_table(
        index='task_name', 
        columns='round', 
        values='metric_value', 
        aggfunc='first'
    )

    # 5. 按照我们确定的顺序重新索引，确保坐标轴正确
    pivot_text = pivot_text.reindex(index=task_order, columns=round_order)
    pivot_values = pivot_values.reindex(index=task_order, columns=round_order)

    # --- 绘图 ---
    print("🎨 正在生成可视化图表...")
    # 设置字体以支持中文（如果需要）
    # plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False 

    # 创建一个足够大的图布
    fig_height = max(6, len(task_order) * 1.2)
    fig_width = max(8, len(round_order) * 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # 计算实际数据的最大值和最小值，用于设置颜色范围
    # 由于最大值可能只有0.6多，我们将vmax设置为实际最大值的1.2倍，以增强中间值的对比度
    valid_values = pivot_values.values[~np.isnan(pivot_values.values)]
    if len(valid_values) > 0:
        data_min = np.min(valid_values)
        data_max = np.max(valid_values)
        # 设置 vmax 为实际最大值的1.2倍，但不超过1.0，以增强对比度
        vmax = min(data_max * 1.2, 1.0) if data_max > 0 else 1.0
        vmin = 0.0
        print(f"📊 数据范围: {data_min:.4f} - {data_max:.4f}, 使用颜色范围: {vmin:.4f} - {vmax:.4f}")
    else:
        vmin = 0.0
        vmax = 1.0
        print(f"⚠️ 警告: 未找到有效数据，使用默认颜色范围: {vmin:.4f} - {vmax:.4f}")

    # 创建自定义颜色映射：从橙色到蓝色
    # 使用 matplotlib 的颜色映射创建器
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#FF8C00', '#FFA500', '#FFD700', '#87CEEB', '#4169E1', '#0000CD']  # 橙->黄->浅蓝->蓝
    n_bins = 256
    custom_cmap = LinearSegmentedColormap.from_list('orange_to_blue', colors, N=n_bins)

    # 使用 Seaborn 的 heatmap 来绘制带注释的热力图
    heatmap = sns.heatmap(
        pivot_values.fillna(np.nan),  # 数值数据用于颜色映射，NaN保持为空
        annot=pivot_text.fillna(""),  # 用我们的文本数据作为注释, NaN部分留空
        fmt="s",                      # 指定注释格式为字符串
        cmap=custom_cmap,             # 自定义橙到蓝颜色映射
        vmin=vmin,                    # 最小值
        vmax=vmax,                    # 最大值（基于实际数据调整以增强对比度）
        center=None,                  # 不使用中心点
        cbar_kws={'label': 'Metric Value'},  # 颜色条的标签
        linewidths=0.5,               # 单元格之间的线条宽度
        linecolor='grey',             # 线条颜色
        annot_kws={"size": 10, "va": "center", "ha": "center"},  # 注释文本的样式
        ax=ax
    )

    # --- 美化图表 ---
    ax.set_title('Continual Learning Evaluation Matrix', fontsize=16, pad=20)
    ax.set_xlabel('Training Round', fontsize=12, labelpad=10)
    ax.set_ylabel('Evaluation Task', fontsize=12, labelpad=10)
    
    # 设置 Y 轴刻度标签（任务名）的旋转角度为0度（水平）
    plt.yticks(rotation=0)
    
    # 设置颜色条标签的字体大小
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Metric Value', fontsize=12, rotation=270, labelpad=20)

    # 确保布局紧凑，所有内容都可见
    plt.tight_layout(pad=1.5)

    # --- 保存图表和 Excel ---
    try:
        plt.savefig(OUTPUT_IMAGE_FILE, dpi=300, bbox_inches='tight')
        print(f"🖼️ 图片已成功保存到: {OUTPUT_IMAGE_FILE}")
        
        # 将数据透视表保存为 Excel 文件（保存文本版本）
        pivot_text.to_excel(OUTPUT_EXCEL_FILE)
        print(f"📊 Excel 文件已成功保存到: {OUTPUT_EXCEL_FILE}")
    except Exception as e:
        print(f"❌ 保存文件时出错: {e}")

    # 显示图表（如果是在 Jupyter Notebook 等环境中）
    # plt.show()


if __name__ == "__main__":
    # 1. 解析数据
    results_df = parse_results(RESULTS_DIR)
    
    # 2. 创建并保存可视化结果
    if not results_df.empty:
        create_visualization(results_df)