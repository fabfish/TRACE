import torch
import importlib
import re
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.data_utils import load_local_dataset
from tqdm import tqdm

# 任务评估匹配表
TASK_EVAL_MAP = {
    'C-STANCE': 'eval_CStance',
    'FOMC': 'eval_FOMC', 
    'MeetingBank': 'eval_MeetingBank',
    'ScienceQA': 'eval_ScienceQA',
    'NumGLUE-cm': 'eval_NumGLUE_cm',
    'NumGLUE-ds': 'eval_NumGLUE_ds',
    '20Minuten': 'eval_20Minuten'
}

# 任务特定的生成参数配置
TASK_GEN_CONFIG = {
    'C-STANCE': {'max_new_tokens': 20, 'temperature': 0.1, 'do_sample': False},
    'FOMC': {'max_new_tokens': 20, 'temperature': 0.1, 'do_sample': False},
    'MeetingBank': {'max_new_tokens': 150, 'temperature': 0.7, 'do_sample': True},
    'ScienceQA': {'max_new_tokens': 100, 'temperature': 0.5, 'do_sample': True},
    'NumGLUE-cm': {'max_new_tokens': 20, 'temperature': 0.1, 'do_sample': False},
    'NumGLUE-ds': {'max_new_tokens': 20, 'temperature': 0.1, 'do_sample': False},
    '20Minuten': {'max_new_tokens': 150, 'temperature': 0.7, 'do_sample': True}
}

def extract_answer(text, prompt):
    """从生成结果中提取答案部分"""
    # 移除输入提示部分
    if prompt in text:
        answer = text[len(prompt):].strip()
    else:
        answer = text.strip()
    return answer

def extract_first_option(text):
    """提取选择题的选项（A, B, C, D）"""
    if not text:
        return ""
    
    # 尝试查找常见答案模式
    match = re.search(r'([A-D])[.、）\)、\s]', text[:20], re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # 查找答案是X的模式
    match = re.search(r'答案是\s*([A-D])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # 提取第一个出现的选项字母
    match = re.search(r'([A-D])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    return ""

def fallback_evaluate(predicted_sequences, ground_truths, is_multiple_choice=False):
    """
    当无法加载评测模块时的后备评估方法
    """
    # 导入本地metrics模块
    from trace_evaluation.metrics import caculate_accuracy, caculate_rouge
    
    if is_multiple_choice:
        # 选择题评估
        return {"accuracy": caculate_accuracy(predicted_sequences, ground_truths)}
    else:
        # 生成式评估
        return {"rouge-l": caculate_rouge(predicted_sequences, ground_truths)}

def evaluate_model_accuracy(model_path, dataset_name, device="auto", result_dir=None):
    """
    评估模型在指定数据集上的性能
    
    Args:
        model_path: 模型路径
        dataset_name: 数据集名称
        device: 运行设备
        result_dir: 实验结果目录，如果提供则保存到该目录下
        
    Returns:
        dict: 包含评估结果的字典
    """
    import datetime
    import os
    
    print(f"\n评估模型 {model_path} 在 {dataset_name} 数据集上的性能")
    
    # 创建评测结果保存文件夹 - 如果提供了result_dir，则保存到该目录下
    if result_dir:
        save_dir = os.path.join(result_dir, "evaluation_outputs")
    else:
        save_dir = os.path.join("./evaluation_results", "generation_outputs")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成唯一文件名（使用模型名称、数据集名称和时间戳）
    model_name = os.path.basename(model_path)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_file = os.path.join(save_dir, f"{model_name}_{dataset_name}_{timestamp}.json")
    
    # 优先加载eval.json进行评估
    test_dataset = load_local_dataset(dataset_name, split='eval')
    if test_dataset is None:
        # 尝试使用test.json作为替代
        test_dataset = load_local_dataset(dataset_name, split='test')
        if test_dataset is None:
            return {"error": f"无法加载数据集 {dataset_name} 的eval、test或validation分割"}
    
    # 确保数据集有prompt和answer字段
    if "prompt" not in test_dataset.column_names or "answer" not in test_dataset.column_names:
        return {"error": f"数据集 {dataset_name} 不包含必需的prompt和answer字段"}
    
    # # 限制评估样本数量（只提取前100个样本）
    # if len(test_dataset) > 100:
    #     print(f"警告: {dataset_name} 数据集样本数量超过100，评估将仅使用前100个样本")
    #     test_dataset = test_dataset.select(range(100))
    # else:
    print(f"使用 {dataset_name} 数据集的全部样本: {len(test_dataset)} 个")
    
    # 加载模型和tokenizer
    try:
        print(f"加载模型: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=device,
            low_cpu_mem_usage=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        return {"error": f"加载模型失败: {e}"}
    
    # 准备评估
    input_sequences = []
    prompts = []
    ground_truths = []
    
    # 提取测试数据
    for example in test_dataset:
        if "prompt" in example and "answer" in example:
            prompts.append(example["prompt"])
            ground_truths.append(example["answer"])
            # 对于需要输入序列的评测(如20Minuten)
            if "input" in example:
                input_sequences.append(example["input"])
            else:
                input_sequences.append(example.get("prompt", ""))
    
    # 确定任务类型和生成参数
    sample_answers = [test_dataset[i]["answer"] for i in range(min(5, len(test_dataset)))]
    is_multiple_choice = all(len(ans.strip()) <= 3 for ans in sample_answers)
    
    # 获取任务特定的生成参数，如果没有则使用默认值
    gen_kwargs = TASK_GEN_CONFIG.get(dataset_name, {})
    if not gen_kwargs:
        if is_multiple_choice:
            gen_kwargs = {
                "max_new_tokens": 20,
                "temperature": 0.1,
                "top_p": 0.9,
                "do_sample": False
            }
        else:
            gen_kwargs = {
                "max_new_tokens": 150,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True
            }
    
    # 使用模型生成答案
    predicted_sequences = []
    full_outputs = []  # 保存完整输出用于记录
    
    print(f"开始为{dataset_name}生成答案...")
    
    for prompt in tqdm(prompts):
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # 生成回答
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
            
            # 解码
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            full_outputs.append(output_text)  # 保存完整输出
            
            # 提取答案部分
            answer = extract_answer(output_text, prompt)
            predicted_sequences.append(answer)
            
        except Exception as e:
            print(f"生成回答时出错: {e}")
            predicted_sequences.append("")
            full_outputs.append("")  # 出错时保存空字符串
    
    # 清理GPU内存
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 查找任务的评估模块
    eval_module_name = TASK_EVAL_MAP.get(dataset_name)
    eval_results = {}
    
    try:
        # 导入相应的评测模块
        if eval_module_name:
            eval_module_path = f"trace_evaluation.eval_modules.{eval_module_name}" 
            module = importlib.import_module(eval_module_path)
            
            # 根据不同评测模块调用相应的评测函数
            if dataset_name == "20Minuten":
                # 需要输入序列
                eval_results = module.eval(input_sequences, predicted_sequences, ground_truths)
            else:
                # 只需要预测序列和真实标签
                eval_results = module.eval(predicted_sequences, ground_truths)
        else:
            print(f"警告: 找不到数据集 {dataset_name} 的评测模块，使用默认评测")
            eval_results = fallback_evaluate(predicted_sequences, ground_truths, is_multiple_choice)
            
    except Exception as e:
        print(f"评测出错: {e}")
        import traceback
        traceback.print_exc()
        
        # 使用后备评测方法
        print("使用备用评测方法...")
        eval_results = fallback_evaluate(predicted_sequences, ground_truths, is_multiple_choice)
    
    # 组装要保存的数据
    save_data = {
        "metadata": {
            "model_path": model_path,
            "model_name": model_name,
            "dataset_name": dataset_name,
            "timestamp": timestamp,
            "num_samples": len(prompts),
            "generation_parameters": gen_kwargs,
            "is_multiple_choice": is_multiple_choice
        },
        "results": eval_results,
        "samples": []
    }
    
    # 添加样本数据
    for i in range(len(prompts)):
        sample = {
            "prompt": prompts[i],
            "generated_answer": predicted_sequences[i],
            "reference_answer": ground_truths[i],
            "full_output": full_outputs[i]
        }
        save_data["samples"].append(sample)
    
    # 保存评测数据到JSON文件
    try:
        with open(save_file, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        print(f"已保存生成内容和评测结果到: {save_file}")
    except Exception as e:
        print(f"保存评测结果文件时出错: {e}")
    
    print(f"{dataset_name} 评测结果: {json.dumps(eval_results, indent=2)}")
    return eval_results