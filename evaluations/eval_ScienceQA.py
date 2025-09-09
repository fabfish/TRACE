import json
from metrics import caculate_bleu, caculate_rouge, caculate_accuracy


# resolving answer and reasoning
def resolve(dataset: list):
    answers = []
    reasonings = []
    for datium in dataset:
        # answers.append(datium[0]) # the first char is the answer. e.g. A, B,...
        # reasonings.append(datium[2:]) # A/nBecause...

        if datium:
            answers.append(datium[0]) # the first char is the answer. e.g. A, B,...
            reasonings.append(datium[2:]) # A/nBecause...
        else:
            answers.append("")
            reasonings.append("")

    outputs = {"answers": answers, "reasonings": reasonings}
    return outputs


def eval(predicted_sequences, ground_truths):
    outputs = resolve(predicted_sequences)
    gts = resolve(ground_truths)

    bleu_1 = caculate_bleu(outputs["reasonings"], gts["reasonings"], 1)
    bleu_4 = caculate_bleu(outputs["reasonings"], gts["reasonings"], 4)
    rouge = caculate_rouge(outputs["reasonings"], gts["reasonings"])
    accuracy = caculate_accuracy(outputs["answers"], gts["answers"])

    evaluation_result = {"bleu-1": bleu_1, "bleu-4": bleu_4, "rouge-L": rouge, "accuracy": accuracy}
    return evaluation_result

# quick fix on llama3.2-1B-Instruct model no output issue
def eval(predicted_sequences, ground_truths):
    outputs = resolve(predicted_sequences)
    gts = resolve(ground_truths)

    # --- Start of fix ---

    # Pair up the predictions and ground truths for reasoning
    reasoning_pairs = zip(outputs["reasonings"], gts["reasonings"])

    # Filter out pairs where either the prediction or the ground truth is an empty string
    valid_reasonings = [(pred, gt) for pred, gt in reasoning_pairs if pred and gt]

    # If there are any valid pairs left, calculate scores
    if valid_reasonings:
        # Unzip the filtered pairs back into two lists
        valid_outputs_reasonings = [pair[0] for pair in valid_reasonings]
        valid_gts_reasonings = [pair[1] for pair in valid_reasonings]

        bleu_1 = caculate_bleu(valid_outputs_reasonings, valid_gts_reasonings, 1)
        bleu_4 = caculate_bleu(valid_outputs_reasonings, valid_gts_reasonings, 4)
        rouge = caculate_rouge(valid_outputs_reasonings, valid_gts_reasonings)
    else:
        # If all reasonings were empty, set scores to 0 to avoid errors
        bleu_1 = 0.0
        bleu_4 = 0.0
        rouge = 0.0

    # --- End of fix ---

    # Accuracy calculation can remain the same
    accuracy = caculate_accuracy(outputs["answers"], gts["answers"])

    evaluation_result = {"bleu-1": bleu_1, "bleu-4": bleu_4, "rouge-L": rouge, "accuracy": accuracy}
    return evaluation_result