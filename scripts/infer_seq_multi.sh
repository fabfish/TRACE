#!bin/bash
port=$(shuf -i25000-30000 -n1)
# deepspeed --include=localhost:0,1,2,3 --master_port $port inference/infer_multi.py  \
#     --data_path /data/datasets/TRACE-Benchmark/LLM-CL-Benchmark_500 \
#     --inference_tasks C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
#     --model_name_or_path /data/models/Llama-2-7b-chat-hf \
#     --inference_model_path /data/yuzhiyuan/outputs_LLM-CL/naive \
#     --inference_batch 64 \
#     --max_prompt_len 1024 \
#     --max_ans_len 512 \
#     --seed 1234 \
#     --deepspeed \
#     --CL_method base \
#     --fixed_generation_config \
#     --inference_output_path /data/yuzhiyuan/outputs_LLM-CL/naive/predictions_multi_fixed_config 2>&1 | tee /data/yuzhiyuan/outputs_LLM-CL/naive/predictions_multi_fixed_config/infer.log &

# deepspeed --include=localhost:4,5,6,7 --master_port $port inference/infer_multi.py  \
#     --data_path /data/datasets/TRACE-Benchmark/LLM-CL-Benchmark_500 \
#     --inference_tasks C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
#     --model_name_or_path /data/models/Qwen1.5-MoE-A2.7B-Chat \
#     --inference_model_path /data/yuzhiyuan/outputs_LLM-CL/naive_qwen_500 \
#     --inference_batch 64 \
#     --max_prompt_len 1024 \
#     --max_ans_len 512 \
#     --seed 1234 \
#     --deepspeed \
#     --CL_method base \
#     --inference_output_path /data/yuzhiyuan/outputs_LLM-CL/naive_qwen_500/predictions 2>&1 | tee /data/yuzhiyuan/outputs_LLM-CL/naive_qwen_500/infer.log &

deepspeed --include=localhost:4,5,6,7 --master_port $port inference/infer_multi.py  \
    --data_path /data/datasets/TRACE-Benchmark/LLM-CL-Benchmark_5000 \
    --inference_tasks C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
    --model_name_or_path /data/models/Llama-3.2-1B-Instruct \
    --inference_model_path /data/yuzhiyuan/outputs_LLM-CL/naive_llama3_1B_full \
    --inference_batch 64 \
    --max_prompt_len 1024 \
    --max_ans_len 512 \
    --seed 1234 \
    --deepspeed \
    --CL_method base \
    --inference_output_path /data/yuzhiyuan/outputs_LLM-CL/naive_llama3_1B_full/predictions_multi_ 2>&1 | tee /data/yuzhiyuan/outputs_LLM-CL/naive_llama3_1B_full/predictions_multi_/infer.log &
    