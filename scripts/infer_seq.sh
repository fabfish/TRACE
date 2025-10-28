#!bin/bash
port=$(shuf -i25000-30000 -n1)
# deepspeed --include=localhost:7 --master_port $port inference/infer_single.py  \
#     --data_path /data/datasets/TRACE-Benchmark/LLM-CL-Benchmark_500 \
#     --inference_tasks C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
#     --model_name_or_path /data/models/Llama-2-7b-chat-hf \
#     --inference_model_path /data/yuzhiyuan/outputs_LLM-CL/naive_full \
#     --inference_batch 1 \
#     --max_prompt_len 1024 \
#     --max_ans_len 512 \
#     --seed 1234 \
#     --deepspeed \
#     --CL_method base \
#     --inference_output_path /data/yuzhiyuan/outputs_LLM-CL/naive_full/predictions_half 2>&1 | tee /data/yuzhiyuan/outputs_LLM-CL/naive_full/infer_half.log &
    
# deepspeed --include=localhost:7 --master_port $port inference/infer_single.py  \
#     --data_path /data/datasets/TRACE-Benchmark/LLM-CL-Benchmark_5000 \
#     --inference_tasks C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
#     --model_name_or_path /data/models/Llama-3.2-1B-Instruct \
#     --inference_model_path /data/yuzhiyuan/outputs_LLM-CL/naive_llama3_1B_full \
#     --inference_batch 4 \
#     --max_prompt_len 1024 \
#     --max_ans_len 512 \
#     --seed 1234 \
#     --deepspeed \
#     --CL_method base \
#     --inference_output_path /data/yuzhiyuan/outputs_LLM-CL/naive_llama3_1B_full/predictions 2>&1 | tee /data/yuzhiyuan/outputs_LLM-CL/naive_llama3_1B_full/predictions/infer.log &
    
# mkdir /data/yuzhiyuan/outputs_LLM-CL/Llama-2-7b-chat-hf/cl/L2P/predictions_single

# deepspeed --include=localhost:7 --master_port $port inference/infer_single.py  \
#     --data_path /data/datasets/TRACE-Benchmark/LLM-CL-Benchmark_5000 \
#     --inference_tasks C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
#     --model_name_or_path /data/models/Llama-2-7b-chat-hf \
#     --inference_model_path /data/yuzhiyuan/outputs_LLM-CL/Llama-2-7b-chat-hf/cl/L2P \
#     --inference_batch 4 \
#     --max_prompt_len 1024 \
#     --max_ans_len 512 \
#     --seed 1234 \
#     --deepspeed \
#     --CL_method L2P \
#     --inference_output_path /data/yuzhiyuan/outputs_LLM-CL/Llama-2-7b-chat-hf/cl/L2P/predictions_single 2>&1 | tee /data/yuzhiyuan/outputs_LLM-CL/Llama-2-7b-chat-hf/cl/L2P/predictions_single/infer.log &


# /data/yuzhiyuan/outputs_LLM-CL/Llama-3.2-1B-Instruct/cl/upcycle

# mkdir -p /data/yuzhiyuan/outputs_LLM-CL/Llama-3.2-1B-Instruct/cl/upcycle/predictions_single

# deepspeed --include=localhost:7 --master_port $port inference/infer_single.py  \
#     --data_path /data/datasets/TRACE-Benchmark/LLM-CL-Benchmark_500 \
#     --inference_tasks C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
#     --model_name_or_path /data/models/Llama-3.2-1B-Instruct \
#     --inference_model_path /data/yuzhiyuan/outputs_LLM-CL/Llama-3.2-1B-Instruct/cl/upcycle \
#     --inference_batch 1 \
#     --max_prompt_len 1024 \
#     --max_ans_len 512 \
#     --seed 1234 \
#     --deepspeed \
#     --CL_method upcycle \
#     --inference_output_path /data/yuzhiyuan/outputs_LLM-CL/Llama-3.2-1B-Instruct/cl/upcycle/predictions_single 2>&1 | tee /data/yuzhiyuan/outputs_LLM-CL/Llama-3.2-1B-Instruct/cl/upcycle/predictions_single/infer.log &

# mkdir -p /data/yuzhiyuan/outputs_LLM-CL/naive_llama3_1B_500/predictions_single

# deepspeed --include=localhost:7 --master_port $port inference/infer_single.py  \
#     --data_path /data/datasets/TRACE-Benchmark/LLM-CL-Benchmark_500 \
#     --inference_tasks C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
#     --model_name_or_path /data/models/Llama-3.2-1B-Instruct \
#     --inference_model_path /data/yuzhiyuan/outputs_LLM-CL/naive_llama3_1B_500/ \
#     --inference_batch 4 \
#     --max_prompt_len 1024 \
#     --max_ans_len 512 \
#     --seed 1234 \
#     --deepspeed \
#     --CL_method base \
#     --inference_output_path /data/yuzhiyuan/outputs_LLM-CL/naive_llama3_1B_500/predictions_single 2>&1 | tee /data/yuzhiyuan/outputs_LLM-CL/naive_llama3_1B_500/predictions_single/infer.log &

mkdir -p /data/yuzhiyuan/outputs_LLM-CL/Llama-3.2-1B-Instruct/cl/upcycle/even_full/predictions_single

deepspeed --include=localhost:7 --master_port $port inference/infer_single.py  \
    --data_path /data/datasets/TRACE-Benchmark/LLM-CL-Benchmark_500 \
    --inference_tasks C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
    --model_name_or_path /data/models/Llama-3.2-1B-Instruct \
    --inference_model_path /data/yuzhiyuan/outputs_LLM-CL/Llama-3.2-1B-Instruct/cl/upcycle/even_full/ \
    --inference_batch 4 \
    --max_prompt_len 1024 \
    --max_ans_len 512 \
    --seed 1234 \
    --deepspeed \
    --CL_method upcycle \
    --inference_output_path /data/yuzhiyuan/outputs_LLM-CL/Llama-3.2-1B-Instruct/cl/upcycle/even_full/predictions_single 2>&1 | tee /data/yuzhiyuan/outputs_LLM-CL/Llama-3.2-1B-Instruct/cl/upcycle/even_full/predictions_single/infer.log &