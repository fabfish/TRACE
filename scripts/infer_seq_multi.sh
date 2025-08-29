#!bin/bash
port=$(shuf -i25000-30000 -n1)
deepspeed --include=localhost:0,1 --master_port $port inference/infer_multi.py  \
    --data_path /data/datasets/TRACE-Benchmark/LLM-CL-Benchmark_500 \
    --dataset_name C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
    --model_name_or_path /data/models/Llama-2-7b-chat-hf \
    --inference_model_path /data/yuzhiyuan/outputs_LLM-CL/naive \
    --inference_batch 1 \
    --max_prompt_len 1024 \
    --max_ans_len 512 \
    --seed 1234 \
    --deepspeed \
    --CL_method base \
    --inference_output_path /data/yuzhiyuan/outputs_LLM-CL/naive/predictions_ 2>&1 | tee /data/yuzhiyuan/outputs_LLM-CL/naive/infer_.log &
    