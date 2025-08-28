## 默认模型在测试时输出的顺序和测试集顺序相同
## The inputs must not be shuffled during evaluation.


import re
from rouge import Rouge
import evaluate
from nltk.translate.bleu_score import sentence_bleu


########################
## BLEU
########################
def tokenize(text):
    tokens = re.split(r'\s|\.', text)
    tokens = [t for t in tokens if len(t) > 0]
    return tokens


def bleu_score(reference, hypothesis, gram):
    reference_tokens = tokenize(reference)
    hypothesis_tokens = tokenize(hypothesis)

    if gram == 1:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1., ))  # BELU-1
    elif gram == 2:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1. / 2., 1. / 2.))  # BELU-2
    elif gram == 3:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1. / 3., 1. / 3., 1. / 3.))  # BELU-3
    elif gram == 4:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1. / 4., 1. / 4., 1. / 4., 1. / 4.))  # BELU-4

    return bleu


def caculate_bleu(results, data, gram):
    bleus = []
    for output_id in range(len(results)):
        prediction = results[output_id]
        target = data[output_id] 
        if prediction == "" or target == "":
            continue
        bleu = bleu_score(target, prediction, gram)
        bleus.append(bleu)
    avg_bleu = sum(bleus) / len(results)
    return avg_bleu


########################
## Rouge-L
########################
def score_rouge(str1, str2):
    rouge = Rouge(metrics=["rouge-l"])
    scores = rouge.get_scores(str1, str2, avg=True)
    rouge_l = scores['rouge-l']['f']
    return rouge_l


def caculate_rouge(results, data):
    rouges = []
    for output_id in range(len(results)):
        prediction = results[output_id]
        target = data[output_id] 
        if prediction == "" or target == "":
            continue
        rouge = score_rouge(target, prediction)
        rouges.append(rouge)
    avg_rouge = sum(rouges) / len(results)
    return avg_rouge

########################
## Accuracy (EM)
########################

def extract_label(text):
    """
    尝试从文本中提取标准选项标签（A/B/C/D...），返回大写字母或None
    """
    import re
    # 匹配如A、A）、A.、A、A）xxx、A.xxx等
    match = re.search(r'([A-D])[\s）\).、\.:：\-]?', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # 兼容如“答案是A”或“选A”
    match = re.search(r'[答选]案?是?([A-D])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return ""

def caculate_accuracy(results, data):
    scores = 0
    valid_count = 0

    for output_id in range(len(results)):
        prediction = str(results[output_id]).strip()
        target = str(data[output_id]).strip().upper()

        # 提取预测中的选项字母
        pred_label = extract_label(prediction)
        # 目标答案直接取首字母（如"A"）
        target_label = target[0] if target else None

        if not pred_label or not target_label:
            continue

        valid_count += 1
        if pred_label == target_label:
            scores += 1

    if valid_count == 0:
        print("错误: 没有有效的预测结果!")
        return 0.0

    avg_score = scores / valid_count
    print(f"准确率: {avg_score:.4f} (正确数: {scores}, 有效样本: {valid_count})")
    return avg_score

########################
## F1-micro
########################
def f1_score(list1, list2):
    # TP: item in list1 and list2
    # FP: item in list1 but not in list2
    # TN: item not in list1 and list2
    # FN: item in list2 but not in list1
    num_TP = 0
    for item1 in list1:
        for item2 in list2:
            if item1 == item2:
                num_TP += 1
                break
    precision = num_TP / len(list1)
    recall = num_TP / len(list2)
    if precision == 0 or recall == 0:
        return 0
    return 2 * (precision * recall / (precision + recall))


def caculate_f1(results, data):
    scores = []
    for output_id in range(len(results)):
        prediction = results[output_id]
        target = data[output_id] 
        if len(prediction) == 0 or len(target) == 0:
            continue
        score = f1_score(target, prediction)
        scores.append(score)
    avg_score = sum(scores) / len(results)
    return avg_score

########################
## SARI
########################
def caculate_sari(inputs, results, data):
    sari = evaluate.load("sari")
    translation_result = sari.compute(sources=inputs, predictions=results, references=[[label] for label in data]),
    return translation_result
