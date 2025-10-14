import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np

# 1. 定义文件路径和对应的 Tag
file_data = [
    {
        'path': '/data/yuzhiyuan/outputs_LLM-CL/Llama-3.2-1B-Instruct/cl/upcycle/even_new_stable/predictions_single/evaluation_matrix.xlsx',
        'tag': 'Upcycle-2L'
    },
    {
        'path': '/data/yuzhiyuan/outputs_LLM-CL/Llama-3.2-1B-Instruct/cl/upcycle/four_new/predictions_single/evaluation_matrix.xlsx',
        'tag': 'Upcycle-4L'
    },
    {
        'path': '/data/yuzhiyuan/outputs_LLM-CL/Llama-3.2-1B-Instruct/cl/upcycle/newnew/predictions_single/evaluation_matrix.xlsx',
        'tag': 'Upcycle-4L-DimR'
    },
    {
        'path': '/data/yuzhiyuan/outputs_LLM-CL/naive_llama3_1B_full/predictions/evaluation_matrix.xlsx',
        'tag': 'SFT-Full'
    },
    {
        'path': '/data/yuzhiyuan/outputs_LLM-CL/naive_llama3_1B_500/predictions_single/evaluation_matrix.xlsx',
        'tag': 'SFT-Subset'
    }
]

# 用于解析指标字符串的函数
def parse_metrics(cell_value):
    """Parses a string like 'metric1: value1\nmetric2: value2' into a dictionary."""
    if pd.isna(cell_value) or cell_value == '':
        return {}
    
    metrics = {}
    # 使用正则表达式匹配 'metric_name: value'
    matches = re.findall(r'(\S+?):\s*([\d\.]+)', str(cell_value))
    for metric_name, value in matches:
        try:
            metrics[metric_name] = float(value)
        except ValueError:
            # Skip if value cannot be converted to float
            continue
    return metrics

# 2. 读取、处理和合并数据
all_data = []

for data_entry in file_data:
    try:
        # 尝试以 Excel 格式读取，如果失败则尝试 CSV（基于您上传的 CSV 样例）
        try:
            df = pd.read_excel(data_entry['path'])
        except FileNotFoundError:
            print(f"File not found: {data_entry['path']}. Skipping.")
            continue
        except Exception:
             # 如果不是标准的 Excel，尝试读取 CSV (假设 Excel/CSV 格式相同)
            df = pd.read_csv(data_entry['path'], encoding='utf-8')
    except Exception as e:
        print(f"Error reading {data_entry['path']}: {e}. Skipping.")
        continue
    
    df.rename(columns={df.columns[0]: 'task_name'}, inplace=True)
    checkpoints = [col for col in df.columns if col not in ['task_name']]
    
    # 遍历任务行
    for index, row in df.iterrows():
        task_name = row['task_name']
        
        # 收集该任务在所有 checkpoint 的指标
        task_metrics_by_checkpoint = {}
        
        for cp in checkpoints:
            metrics_dict = parse_metrics(row[cp])
            
            # 如果是第一个 non-empty checkpoint，用它来初始化所有指标
            if not task_metrics_by_checkpoint:
                for metric in metrics_dict.keys():
                    task_metrics_by_checkpoint[metric] = {c: np.nan for c in checkpoints}
            
            # 填充当前 checkpoint 的数值
            for metric, value in metrics_dict.items():
                if metric in task_metrics_by_checkpoint:
                    task_metrics_by_checkpoint[metric][cp] = value
        
        # 为每个指标创建一个新行
        for metric, cp_values in task_metrics_by_checkpoint.items():
            new_row = {
                'Task Name': task_name,
                'Metric': metric,
                'Method Tag': data_entry['tag']
            }
            new_row.update(cp_values)
            all_data.append(new_row)

# 3. 生成最终的 DataFrame 和 Excel 文件
final_df = pd.DataFrame(all_data)

# 将 Checkpoint 列转换为整数类型
checkpoint_cols = [col for col in final_df.columns if str(col).isdigit()]
final_df[checkpoint_cols] = final_df[checkpoint_cols].apply(pd.to_numeric, errors='coerce')

# 重新组织列顺序：Task Name, Method Tag, Metric, Checkpoints...
final_cols = ['Task Name', 'Method Tag', 'Metric'] + sorted(checkpoint_cols, key=int)
final_df = final_df[final_cols]

# 保存为新的 Excel 文件
output_excel_path = 'combined_evaluation_matrix_analysis.xlsx'
final_df.to_excel(output_excel_path, index=False)
print(f"数据已成功保存至: {output_excel_path}")


# 4. 生成示意图脚本 (以 MeetingBank - rouge-L 为例)

# 找到一个用于绘图的代表性指标
plot_task = 'MeetingBank'
plot_metric = 'rouge-L'

# 过滤出用于绘图的数据
plot_data = final_df[
    (final_df['Task Name'] == plot_task) & 
    (final_df['Metric'] == plot_metric)
]

if not plot_data.empty:
    plt.figure(figsize=(12, 6))
    
    # 绘制每种方法的性能曲线
    for tag in plot_data['Method Tag'].unique():
        method_data = plot_data[plot_data['Method Tag'] == tag].iloc[0]
        performance = method_data[checkpoint_cols].dropna()
        
        if not performance.empty:
            checkpoints_to_plot = performance.index.astype(int)
            values_to_plot = performance.values
            plt.plot(checkpoints_to_plot, values_to_plot, 
                     marker='o', linestyle='-', label=tag)

    plt.title(f'Performance Comparison on {plot_task} - {plot_metric}')
    plt.xlabel('Checkpoint')
    plt.ylabel(plot_metric)
    plt.xticks(checkpoints_to_plot) # 确保 X 轴刻度是 Checkpoint 编号
    plt.legend(title='Method Tag')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图表
    output_image_path = f'plot_{plot_task}_{plot_metric}.png'
    plt.savefig(output_image_path)
    plt.close()
    print(f"示意图已成功保存至: {output_image_path}")
else:
    print(f"未找到任务 '{plot_task}' 的指标 '{plot_metric}' 数据，跳过绘图。")