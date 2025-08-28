from trace_evaluation.metrics import caculate_accuracy


def eval(predicted_sequences, ground_truths):
    accuracy = caculate_accuracy(predicted_sequences, ground_truths)
    evaluation_result = {"accuracy": accuracy}
    return evaluation_result
import re

def extract_first_number(text):
    """从文本中提取第一个数字或数值答案"""
    if not text:
        return None
        
    # 尝试提取独立的数字
    match = re.search(r'\b(\d+)\b', text)
    if match:
        return match.group(1)
    
    # 尝试匹配"答案: 123"格式
    match = re.search(r'[Aa]nswer[:\s]+(\d+)', text)
    if match:
        return match.group(1)
        
    # 如果是文本答案(如"June")，则保留原样
    return text.strip()

def eval(predicted_sequences, ground_truths):
    """评估NumGLUE-ds的结果，直接比较数值或文本答案"""
    correct = 0
    valid_count = 0
    
    for i in range(len(predicted_sequences)):
        prediction = predicted_sequences[i]
        target = ground_truths[i]
        
        if not prediction or not target:
            continue
            
        # 提取预测中的第一个数字或答案
        pred_value = extract_first_number(prediction)
        target_value = target.strip()
        
        valid_count += 1
        if pred_value == target_value:
            correct += 1
    
    if valid_count == 0:
        accuracy = 0
    else:
        accuracy = correct / valid_count
    
    print(f"NumGLUE-ds 准确率: {accuracy:.4f} (正确: {correct}/{valid_count})")
    evaluation_result = {"accuracy": accuracy}
    return evaluation_result