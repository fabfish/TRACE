import pandas as pd
import numpy as np
import os
import re
from openpyxl import Workbook
from openpyxl.utils.cell import get_column_letter
from openpyxl.styles import Alignment, Border, Side, Font
import dataframe_image as dfi
import matplotlib.pyplot as plt
import ast

# --- 1. 定义文件路径和标签 ---
# ⚠️ 警告：请确保您已将这些路径替换为正确的本地路径
file_paths = {
    "1L": "/data/yuzhiyuan/outputs_LLM-CL/Llama-3.2-1B-Instruct/cl/upcycle/newnew/predictions_single/evaluation_matrix.xlsx",
    "2L": "/data/yuzhiyuan/outputs_LLM-CL/Llama-3.2-1B-Instruct/cl/upcycle/even_new_stable/predictions_single/evaluation_matrix.xlsx",
    "2L_G": "/data/yuzhiyuan/outputs_LLM-CL/Llama-3.2-1B-Instruct/cl/upcycle/grouped_4of8/predictions_single/evaluation_matrix.xlsx",
    "4L": "/data/yuzhiyuan/outputs_LLM-CL/Llama-3.2-1B-Instruct/cl/upcycle/four_new/predictions_single/evaluation_matrix.xlsx",
    "4L_R": "/data/yuzhiyuan/outputs_LLM-CL/Llama-3.2-1B-Instruct/cl/upcycle/four copy?/predictions_single/evaluation_matrix.xlsx",
    "Sub SFT": "/data/yuzhiyuan/outputs_LLM-CL/naive_llama3_1B_500/predictions_single/evaluation_matrix.xlsx",
    "Full SFT": "/data/yuzhiyuan/outputs_LLM-CL/naive_llama3_1B_full/predictions/evaluation_matrix.xlsx",
}

output_excel_path = "merged_evaluation_matrix_aligned.xlsx"
output_image_path = "merged_evaluation_matrix_aligned.png"

# 定义方法在每个 Checkpoint 单元格内的显示顺序
STANDARD_METHODS_ORDER = ["2L", "4L_nodim", "4L_dim", "Sub SFT"]
FULL_SFT_METHOD = "Full SFT"

# 1. 固定任务顺序
TASK_ORDER = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]

# 2. 方法颜色映射（扩展为1L>2L>4L>4L_R）
METHOD_COLORS = {
    "1L": "#FF6666",       # 深红
    "2L": "#FFB266",       # 橙
    "4L": "#90EE90",       # 浅绿
    "4L_R": "#66FF66",     # 另一种浅绿
}

# --- 2. 辅助函数：解析指标字符串 ---
def parse_metrics(metric_str):
    """
    将单元格内的多行指标字符串转换为 {tag: value} 字典。
    """
    metrics = {}
    if pd.isna(metric_str) or not isinstance(metric_str, str):
        return metrics
    
    lines = metric_str.strip().split('\n')
    for line in lines:
        if ':' in line:
            tag, value = line.split(':', 1)
            metrics[tag.strip()] = value.strip()
    return metrics

def extract_sari_number(value):
    """
    从字符串中提取 sari 数字（如 [{'sari': 40.000000}] -> 40.00）
    """
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}"
    if isinstance(value, str):
        # 尝试用正则提取数字
        match = re.search(r"['\"]?sari['\"]?\s*[:=]\s*([0-9.]+)", value)
        if match:
            return f"{float(match.group(1)):.2f}"
        # 尝试直接转为 float
        try:
            return f"{float(value):.2f}"
        except Exception:
            pass
    return value

# --- 3. 数据加载与预处理 ---
def load_and_preprocess_data(file_paths):
    all_data_list = []
    for method_tag, path in file_paths.items():
        try:
            if path.endswith('.csv'):
                df = pd.read_csv(path)
            elif path.endswith('.xlsx'):
                df = pd.read_excel(path, sheet_name=0)
            else:
                continue

            df = df.rename(columns={'task_name': 'Task'})
            df = df.set_index('Task')

            for task_name in df.index:
                for col in df.columns:
                    checkpoint = str(col)
                    original_metric_str = df.loc[task_name, col]
                    metrics = parse_metrics(original_metric_str)
                    if not metrics and pd.notna(original_metric_str) and original_metric_str.strip() != "":
                        continue
                    for metric_tag, metric_value in metrics.items():
                        all_data_list.append({
                            'Method': method_tag,
                            'Task': task_name,
                            'Checkpoint': checkpoint,
                            'Metric_Tag': metric_tag,
                            'Metric_Value': metric_value
                        })
        except FileNotFoundError:
            print(f"Error: File not found at {path}")
        except Exception as e:
            print(f"Error processing file {path} for tag {method_tag}: {e}")

    final_df = pd.DataFrame(all_data_list)
    return final_df

# --- 4. 导出为 Excel (使用 openpyxl 手动构建复杂表格) ---
def export_to_excel_with_merge(df, output_path):
    all_tasks = sorted(df['Task'].unique())
    # 只保留数字 Checkpoint
    all_checkpoints = sorted(
        [cp for cp in df['Checkpoint'].unique() if cp.isdigit() and int(cp) >= 0],
        key=int
    )
    all_methods = [m for m in file_paths.keys()]

    wb = Workbook()
    ws = wb.active
    ws.title = "Merged_Evaluation"

    thin_border = Border(left=Side(style='thin'), right=Side(style='thin'), top=Side(style='thin'), bottom=Side(style='thin'))
    header_font = Font(bold=True, size=10)
    center_alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
    left_alignment = Alignment(horizontal='left', vertical='center', wrap_text=True)

    ws.cell(row=1, column=1, value="Epoch").font = header_font
    ws.cell(row=1, column=1).alignment = center_alignment

    current_col = 2
    for task in all_tasks:
        ws.cell(row=1, column=current_col, value=task).font = header_font
        ws.cell(row=1, column=current_col).alignment = center_alignment
        current_col += 1

    current_row = 2
    for checkpoint in all_checkpoints:
        ws.cell(row=current_row, column=1, value=f"Epoch {checkpoint}")
        ws.cell(row=current_row, column=1).font = header_font
        ws.cell(row=current_row, column=1).alignment = center_alignment

        for t_idx, task in enumerate(all_tasks):
            cell_content_lines = []
            for method in all_methods:
                sub_df = df[
                    (df['Checkpoint'] == str(checkpoint)) &
                    (df['Task'] == task) &
                    (df['Method'] == method)
                ]
                for _, row in sub_df.iterrows():
                    cell_content_lines.append(f"{method}_{row['Metric_Tag']}: {row['Metric_Value']}")
            ws.cell(row=current_row, column=2 + t_idx, value="\n".join(cell_content_lines))
            ws.cell(row=current_row, column=2 + t_idx).alignment = left_alignment

        current_row += 1

    for row in range(1, current_row):
        for col in range(1, ws.max_column + 1):
            cell = ws.cell(row=row, column=col)
            if cell.border == Border():
                cell.border = thin_border

    ws.row_dimensions[1].height = 25
    for i in range(len(all_tasks) + 1):
        ws.column_dimensions[get_column_letter(i + 1)].width = 25

    wb.save(output_path)
    return current_row - 1

# --- 5. 导出为图片 (使用 Styler 模拟表格结构) ---
def is_main_metric(metric_tag, task=None):
    tag = metric_tag.lower()
    # MeetingBank 任务下，rouge-L 视为主指标
    if task == "MeetingBank" and tag == "rouge-l":
        return True
    # 其他任务，排除bleu和rouge
    return not (('bleu' in tag) or ('rouge' in tag))

def export_to_image(df, output_path):
    all_tasks = TASK_ORDER
    all_checkpoints = sorted(
        [cp for cp in df['Checkpoint'].unique() if cp.isdigit() and int(cp) >= 0],
        key=int
    )
    all_methods = [m for m in file_paths.keys()]

    # 1. 构建颜色图例
    legend_labels = list(METHOD_COLORS.keys())
    legend_colors = [METHOD_COLORS[k] for k in legend_labels]
    legend_df = pd.DataFrame([legend_labels], columns=legend_labels)
    legend_color_df = pd.DataFrame([legend_colors], columns=legend_labels)

    # 2. 构建主表格
    data_rows = []
    color_rows = []
    for checkpoint in all_checkpoints:
        row_data = {}
        row_colors = {}
        for task in all_tasks:
            cell_lines = []
            main_metrics = {}
            for method in all_methods:
                sub_df = df[
                    (df['Checkpoint'] == str(checkpoint)) &
                    (df['Task'] == task) &
                    (df['Method'] == method)
                ]
                for _, row in sub_df.iterrows():
                    if is_main_metric(row['Metric_Tag'], task):
                        try:
                            val = float(row['Metric_Value'])
                        except Exception:
                            val = None
                        main_metrics[method] = (row['Metric_Tag'], row['Metric_Value'], val)
            for method in all_methods:
                if method in main_metrics:
                    tag, value, val = main_metrics[method]
                    if tag.lower() == "sari":
                        value_fmt = extract_sari_number(value)
                        cell_lines.append(f"{method} sari: {value_fmt}")
                    else:
                        try:
                            value_fmt = f"{float(value):.2f}"
                        except Exception:
                            value_fmt = value
                        cell_lines.append(f"{method}_{tag}: {value_fmt}")

            color = "#FFFFFF"
            if "Full SFT" in main_metrics:
                _, _, full_val = main_metrics["Full SFT"]
                for m in ["1L", "2L", "4L", "4L_R"]:
                    if m in main_metrics:
                        _, _, m_val = main_metrics[m]
                        if m_val is not None and full_val is not None and m_val > full_val:
                            color = METHOD_COLORS[m]
                            break
            row_data[task] = "\n".join(cell_lines)
            row_colors[task] = color
        data_rows.append(row_data)
        color_rows.append(row_colors)

    df_display = pd.DataFrame(data_rows, index=[f"Epoch {cp}" for cp in all_checkpoints], columns=all_tasks)
    color_df = pd.DataFrame(color_rows, index=df_display.index, columns=all_tasks)

    styles = [
        {'selector': 'td', 'props': [('text-align', 'left'), ('vertical-align', 'middle'), ('padding', '5px'), ('font-size', '10pt')]},
        {'selector': 'th', 'props': [('text-align', 'center'), ('vertical-align', 'middle'), ('font-weight', 'bold'), ('background-color', '#f0f0f0')]},
        {'selector': '.row_heading', 'props': [('text-align', 'left'), ('vertical-align', 'middle')]},
    ]

    # 图例样式
    legend_styler = legend_df.style.set_table_styles(styles)
    legend_styler = legend_styler.apply(lambda x: [f'background-color: {legend_color_df.loc[x.name, col]};' for col in x.index], axis=1)
    legend_styler.set_table_attributes('style="border-collapse: collapse"')

    # 主表格样式
    styler = df_display.style.set_table_styles(styles)
    styler = styler.apply(lambda x: [f'background-color: {color_df.loc[x.name, col]}; white-space: pre-line;' for col in x.index], axis=1)

    # 有内容的格子加黑色边框
    def border_if_not_empty(val):
        if val and str(val).strip():
            return 'border: 1px solid #000000;'
        else:
            return ''
    styler = styler.applymap(border_if_not_empty)
    styler.set_table_attributes('style="border-collapse: collapse"')

    # 导出图例和主表格
    import matplotlib.pyplot as plt
    import io
    import PIL.Image

    # 先导出图例和主表为图片
    import dataframe_image as dfi
    buf_legend = io.BytesIO()
    buf_main = io.BytesIO()
    dfi.export(legend_styler, buf_legend, max_rows=-1, max_cols=-1, table_conversion='matplotlib')
    dfi.export(styler, buf_main, max_rows=-1, max_cols=-1, table_conversion='matplotlib')
    buf_legend.seek(0)
    buf_main.seek(0)
    img_legend = PIL.Image.open(buf_legend)
    img_main = PIL.Image.open(buf_main)

    # 拼接图例和主表
    total_width = max(img_legend.width, img_main.width)
    total_height = img_legend.height + img_main.height
    new_img = PIL.Image.new('RGB', (total_width, total_height), (255, 255, 255))
    new_img.paste(img_legend, (0, 0))
    new_img.paste(img_main, (0, img_legend.height))
    new_img.save(output_path)
    print(f"\n✅ 图片已成功导出到: {output_image_path}")

# --- 7. 主执行流程 ---
if __name__ == "__main__":
    
    print("🚀 开始加载和预处理数据...")
    final_df = load_and_preprocess_data(file_paths)
    
    if final_df is None or final_df.empty:
        print("❌ 数据预处理失败或数据为空，请检查文件路径和内容。")
    else:
        print("✅ 数据预处理完成。")
        
        # 导出 Excel 文件
        print("📝 正在导出 Excel 文件 (复杂结构/最大行对齐) ...")
        export_to_excel_with_merge(final_df, output_excel_path)
        print(f"✅ Excel 文件已成功导出到: {output_excel_path}")
        
        # 导出图片
        print("🖼️ 正在生成图片 (模拟合并结构) ...")
        export_to_image(final_df, output_image_path)