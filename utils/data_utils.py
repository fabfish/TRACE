import os
import json
from datasets import Dataset

def load_local_dataset(dataset_name, split='train'):
    """
    从本地加载指定的数据集
    
    Args:
        dataset_name: 数据集名称
        split: 数据分割类型 ('train', 'test', 'eval')
        
    Returns:
        dataset: 加载的数据集(Huggingface Dataset格式)
    """
    try:
        # 构建数据集路径
        dataset_path = os.path.join('datasets', dataset_name, f'{split}.json')
        
        if not os.path.exists(dataset_path):
            print(f"找不到数据集文件: {dataset_path}")
            return None
            
        # 加载JSON数据
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 转换为Huggingface Dataset格式
        if isinstance(data, list):
            return Dataset.from_list(data)
        elif isinstance(data, dict) and "examples" in data:
            return Dataset.from_list(data["examples"])
        elif isinstance(data, dict):
            return Dataset.from_dict(data)
                
        print(f"不支持的数据集格式: {dataset_path}")
        return None
        
    except Exception as e:
        print(f"加载数据集 {dataset_name} 失败: {e}")
        return None

def prepare_dataset_for_training(dataset, dataset_name=None):
    """
    准备数据集用于训练，根据统一的prompt-answer格式进行处理
    
    Args:
        dataset: 原始数据集
        dataset_name: 数据集名称
        
    Returns:
        dataset: 处理后的数据集
    """
    if dataset is None:
        return None
    
    # 检查数据集是否已有"prompt"和"answer"字段
    columns = dataset.column_names
    if "prompt" in columns and "answer" in columns:
        # 已有标准字段，将其转换为训练格式
        def convert_to_text_format(example):
            return {"text": f"{example['prompt']}\n{example['answer']}"}
        
        processed_dataset = dataset.map(convert_to_text_format)
        return processed_dataset
    else:
        # 非标准格式，尝试通过其他方式转换
        print(f"警告: 数据集 {dataset_name} 没有标准的prompt-answer格式，尝试其他转换方式")
        
        def fallback_conversion(example):
            text_parts = []
            for key, value in example.items():
                if isinstance(value, str) and value.strip():
                    text_parts.append(f"{key}: {value}")
            
            return {"text": "\n".join(text_parts)}
        
        processed_dataset = dataset.map(fallback_conversion)
        return processed_dataset

def get_available_datasets():
    """
    获取datasets目录中可用的所有数据集名称
    
    Returns:
        datasets_list: 数据集名称列表
    """
    datasets_dir = 'datasets'
    if not os.path.exists(datasets_dir):
        print(f"找不到datasets目录")
        return []
        
    return [name for name in os.listdir(datasets_dir) 
            if os.path.isdir(os.path.join(datasets_dir, name))]