import os
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 配置参数 ---
# JSON 结果文件所在的目录
ROOT_DIR = "/data/yuzhiyuan/outputs_LLM-CL/naive_llama3_1B_500/predictions_single"
# RESULTS_DIR = os.path.join(ROOT_DIR, "predictions")
RESULTS_DIR = ROOT_DIR
# 输出图片的文件名
# OUTPUT_IMAGE_FILE = "/data/yuzhiyuan/outputs_LLM-CL/naive_full/evaluation_matrix.png"
OUTPUT_IMAGE_FILE = os.path.join(ROOT_DIR, "evaluation_matrix.png")
# 输出 Excel 的文件名
# OUTPUT_EXCEL_FILE = "/data/yuzhiyuan/outputs_LLM-CL/naive_full/evaluation_matrix.xlsx"
OUTPUT_EXCEL_FILE = os.path.join(ROOT_DIR, "evaluation_matrix.xlsx")


def parse_results(directory):
    """
    解析指定目录下的所有结果 JSON 文件。

    Args:
        directory (str): 包含 JSON 文件的目录路径。

    Returns:
        pd.DataFrame: 一个包含解析后数据的 Pandas DataFrame。
                      列包括 'round', 'task_id', 'task_name', 'metrics_str'。
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

                records.append({
                    "round": round_num,
                    "task_id": task_id,
                    "task_name": task_name,
                    "metrics_str": metrics_str
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

    # 3. 创建数据透视表，行为任务，列为轮次
    pivot_df = df.pivot_table(
        index='task_name', 
        columns='round', 
        values='metrics_str', 
        aggfunc='first' # 每个 (task, round) 只有一个值，first 即可
    )

    # 4. 按照我们确定的顺序重新索引，确保坐标轴正确
    pivot_df = pivot_df.reindex(index=task_order, columns=round_order)

    # --- 绘图 ---
    print("🎨 正在生成可视化图表...")
    # 设置字体以支持中文（如果需要）
    # plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False 

    # 创建一个足够大的图布
    fig_height = max(6, len(task_order) * 1.2)
    fig_width = max(8, len(round_order) * 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # 使用 Seaborn 的 heatmap 来绘制带注释的网格
    # 我们创建一个虚拟的数值矩阵用于着色，实际内容由 annot 参数决定
    sns.heatmap(
        np.zeros(pivot_df.shape),  # 虚拟数据，只为画格子
        annot=pivot_df.fillna(""), # 用我们的文本数据作为注释, NaN部分留空
        fmt="s",                   # 指定注释格式为字符串
        cmap="coolwarm",           # 背景颜色 (几乎不可见)
        cbar=False,                # 不显示颜色条
        linewidths=0.5,            # 单元格之间的线条宽度
        linecolor='grey',          # 线条颜色
        annot_kws={"size": 10, "va": "center", "ha": "center"} # 注释文本的样式
    )

    # --- 美化图表 ---
    ax.set_title('Continual Learning Evaluation Matrix', fontsize=16, pad=20)
    ax.set_xlabel('Training Round', fontsize=12, labelpad=10)
    ax.set_ylabel('Evaluation Task', fontsize=12, labelpad=10)
    
    # 设置 Y 轴刻度标签（任务名）的旋转角度为0度（水平）
    plt.yticks(rotation=0)

    # 确保布局紧凑，所有内容都可见
    plt.tight_layout(pad=1.5)

    # --- 保存图表和 Excel ---
    try:
        plt.savefig(OUTPUT_IMAGE_FILE, dpi=300, bbox_inches='tight')
        print(f"🖼️ 图片已成功保存到: {OUTPUT_IMAGE_FILE}")
        
        # 将数据透视表保存为 Excel 文件
        pivot_df.to_excel(OUTPUT_EXCEL_FILE)
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