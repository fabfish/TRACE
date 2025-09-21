#!/bin/bash

# Generates a random port number between 25000 and 30000
port=$(shuf -i25000-30000 -n1)

mkdir -p /data/yuzhiyuan/outputs_LLM-CL/Llama-3.2-1B-Instruct/baseline/predictions

# Run the inference script for the baseline model
deepspeed --include=localhost:7 --master_port $port inference/infer_base.py  \
    --data_path /data/datasets/TRACE-Benchmark/LLM-CL-Benchmark_500 \
    --inference_tasks C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
    --model_name_or_path /data/models/Llama-3.2-1B-Instruct \
    --inference_model_path data/models/Llama-3.2-1B-Instruct \
    --inference_batch 1 \
    --max_prompt_len 1024 \
    --max_ans_len 512 \
    --seed 1234 \
    --deepspeed \
    --inference_output_path /data/yuzhiyuan/outputs_LLM-CL/Llama-3.2-1B-Instruct/baseline/predictions 2>&1 | tee /data/yuzhiyuan/outputs_LLM-CL/Llama-3.2-1B-Instruct/baseline/infer_baseline.log &