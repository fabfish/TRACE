"""
This script performs baseline inference on a set of tasks using a
pre-trained model, without any continual learning methods.
"""
#!/usr/bin/env python

import argparse
import os
import sys
from tqdm import tqdm
import torch
import torch_npu
from torch.utils.data import DataLoader, SequentialSampler
import json

from transformers import AutoModelForCausalLM
import deepspeed

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data.data_collator import DataCollator
from utils.data.data_utils import create_prompt_dataset
from utils.utils import print_rank_0, to_device, set_random_seed, load_hf_tokenizer
from evaluations import eval_ScienceQA, eval_MeetingBank, eval_PapyrusF, eval_CStance, eval_Py150, eval_FOMC, eval_NumGLUE_cm, eval_NumGLUE_ds, eval_20Minuten


def parse_args():
    def list_of_strings(arg):
        return arg.split(',')
    parser = argparse.ArgumentParser(
        description="Run baseline inference on a pretrained model.")
    parser.add_argument('--data_path',
                        type=str,
                        default='Dahoas/rm-static',
                        help='Path to the training dataset. A single data path.')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help=
        'Where to store the data-related files.'
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--inference_model_path",
        type=str,
        help="Path to inference model. For baseline, this can be the same as model_name_or_path.",
        required=True,
    )
    parser.add_argument(
        "--max_prompt_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--max_ans_len",
        type=int,
        default=256,
        help="The maximum answer length.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Generate temperature params.",
    )
    parser.add_argument(
        "--inference_batch",
        type=int,
        default=4,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--inference_tasks",
        type=list_of_strings,
        default='all',
        help='Datasets to be used.'
    )
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="A seed for reproducible training.")

    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--inference_output_path',
                        type=str,
                        default=None,
                        help="Where to store inference results.")
    
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    set_random_seed(args.seed)
    device = torch.device("npu")


    def prediction(model, infer_dataloader, tokenizer, inference_task=None):
        predicted_sequences = []
        sources_sequences = []
        ground_truths = []
        model.eval()

        # Set generation parameters based on the task
        max_new_tokens = args.max_ans_len
        temperature = args.temperature
        do_sample = True
        
        if inference_task == "C-STANCE" or inference_task == "FOMC" or inference_task == "NumGLUE-cm" or inference_task == "NumGLUE-ds":
            max_new_tokens = 20
            temperature = 0.1
            do_sample = False
        elif inference_task == "MeetingBank" or inference_task == "20Minuten":
            max_new_tokens = 150
            temperature = 0.7
            do_sample = True
        elif inference_task == "ScienceQA":
            max_new_tokens = 100
            temperature = 0.5
            do_sample = True


        print(f"Model dtype: {next(model.parameters()).dtype}")

        for step, batch in enumerate(tqdm(infer_dataloader, desc="Inference Progress")):
            sources_sequences += batch['sources']
            ground_truths += batch['gts']
            del batch['gts']
            del batch['sources']
            
            batch = to_device(batch, device)
            prompt_len = batch['input_ids'].shape[1]

            with torch.no_grad():
                generate_ids = model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_new_tokens=max_new_tokens,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.unk_token_id,
                    temperature=temperature,
                    do_sample=do_sample,
                    num_return_sequences=1,
                    use_cache=True
                )
            sequences = tokenizer.batch_decode(generate_ids[:, prompt_len:], skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)
            predicted_sequences += sequences

        return sources_sequences, predicted_sequences, ground_truths

    def save_inference_results(evaluation_result: dict, sources_sequences: list, predicted_sequences: list,
                                ground_truths: list, task: str):
        df = {"eval": evaluation_result, 'prompts': sources_sequences, 'results': predicted_sequences,
                'labels': ground_truths}
        if not os.path.exists(args.inference_output_path):
            os.makedirs(args.inference_output_path)
        with open(os.path.join(args.inference_output_path, f"results-baseline-{task}.json"), "w+", encoding='utf-8') as file:
            json.dump(df, file, ensure_ascii=False)


    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    assert tokenizer.padding_side == 'left'
    assert tokenizer.truncation_side == "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    # Place model on the NPU device
    model = model.to(device)

    inference_tasks = args.inference_tasks
    if isinstance(inference_tasks, str):
        inference_tasks = inference_tasks.split(',')
    
    for inference_task in inference_tasks:
        dataset_path = os.path.join(args.data_path, inference_task)
        _, _, infer_dataset = create_prompt_dataset(
            args.local_rank,
            dataset_path,
            args.data_output_path,
            args.seed,
            distributed=False
        )

        inf_data_collator = DataCollator(
            tokenizer,
            model=model,
            padding="longest",
            max_prompt_len=args.max_prompt_len,
            max_ans_len=args.max_ans_len,
            pad_to_multiple_of=8,
            inference=True
        )
        infer_sampler = SequentialSampler(infer_dataset)
        infer_dataloader = DataLoader(infer_dataset,
                                      collate_fn=inf_data_collator,
                                      sampler=infer_sampler,
                                      batch_size=args.inference_batch)

        print_rank_0(f"***** Starting baseline inference for task: {inference_task} *****", args.local_rank)
        sources_sequences, predicted_sequences, ground_truths = prediction(model, infer_dataloader, tokenizer, inference_task)
        
        # Evaluate based on the task
        if inference_task == "ScienceQA":
            evaluation_result = eval_ScienceQA.eval(predicted_sequences, ground_truths)
        elif inference_task == "MeetingBank":
            evaluation_result = eval_MeetingBank.eval(predicted_sequences, ground_truths)
        elif inference_task == "C-STANCE":
            evaluation_result = eval_CStance.eval(predicted_sequences, ground_truths)
        elif inference_task == "Papyrus-f":
            evaluation_result = eval_PapyrusF.eval(predicted_sequences, ground_truths)
        elif inference_task == "Py150":
            evaluation_result = eval_Py150.eval(predicted_sequences, ground_truths)
        elif inference_task == "FOMC":
            evaluation_result = eval_FOMC.eval(predicted_sequences, ground_truths)
        elif inference_task == "NumGLUE-cm":
            evaluation_result = eval_NumGLUE_cm.eval(predicted_sequences, ground_truths)
        elif inference_task == "NumGLUE-ds":
            evaluation_result = eval_NumGLUE_ds.eval(predicted_sequences, ground_truths)
        elif inference_task == "20Minuten":
            evaluation_result = eval_20Minuten.eval(sources_sequences, predicted_sequences, ground_truths)
        else:
            evaluation_result = {}

        print("***** Saving baseline inference results *****")
        save_inference_results(evaluation_result, sources_sequences, predicted_sequences, ground_truths, inference_task)
    
    # Release model memory
    del model
    torch_npu.npu.empty_cache()


if __name__ == "__main__":
    main()