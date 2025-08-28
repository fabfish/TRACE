#!bin/bash
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:64,garbage_collection_threshold:0.5
port=$(shuf -i25000-30000 -n1)

MODEL_PATH="/data/models/Qwen1.5-MoE-A2.7B-Chat" 
DATASET_LIST="C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten"
EPOCH_LIST="5,3,7,5,3,5,5,7"

# --- OOM 修复配置 ---
MAX_LEN=1024
GRAD_CKPT="--gradient_checkpointing"

# --- Offload 配置 (按需启用) ---
# 如果上面的配置仍然OOM，取消下面一行的注释
OFFLOAD_FLAG="--offload"
deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port $port training/main.py \
    --data_path /data/datasets/TRACE-Benchmark/LLM-CL-Benchmark_500 \
    --dataset_name $DATASET_LIST \
    --model_name_or_path $MODEL_PATH \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --max_prompt_len $MAX_LEN \
    --max_ans_len 512 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --num_train_epochs $EPOCH_LIST \
    --gradient_accumulation_steps 32 \
    $GRAD_CKPT \
    $OFFLOAD_FLAG \
    --lr_scheduler_type cosine \
    --num_warmup_steps 0 \
    --seed 1234 \
    --zero_stage 3 \
    --deepspeed \
    --print_loss \
    --CL_method base \
    --output_dir /data/yuzhiyuan/outputs_LLM-CL/naive_qwen_500 2>&1 | tee /data/yuzhiyuan/outputs_LLM-CL/naive_qwen_500/train.log &



# # for slurm

# # for slurm running
# srun --partition=xai --mpi=pmi2 --gres=gpu:8 -n1 --ntasks-per-node=1 --job-name=asb-llama --kill-on-bad-exit=1 /mnt/petrelfs/wangxiao/miniconda3/envs/cl/bin/deepspeed --master_port 51459 training/main.py  \
#     --data_path /mnt/petrelfs/wangxiao/DATA/LLM-CL-Benchmark/LLM-CL-Benchmark_5000 \
#     --dataset_name C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
#     --model_name_or_path /mnt/petrelfs/wangxiao/MODELS/llama2HF/7B-Chat \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 16 \
#     --max_prompt_len 1024 \
#     --max_ans_len 512 \
#     --learning_rate 1e-5 \
#     --weight_decay 0. \
#     --num_train_epochs 5,3,7,5,3,5,5,7 \
#     --gradient_accumulation_steps 16 \
#     --lr_scheduler_type cosine \
#     --num_warmup_steps 0 \
#     --seed 1234 \
#     --zero_stage 3 \
#     --deepspeed \
#     --print_loss \
#     --CL_method base \
#     --output_dir /mnt/petrelfs/wangxiao/LLM-Continual-Learning/CKPT/naive/llama7b-chat > llama2-7b-naive.log 2>&1 &



# srun --partition=xai --mpi=pmi2 --gres=gpu:8 -n1 --ntasks-per-node=1 --job-name=asb-llama --kill-on-bad-exit=1 /mnt/petrelfs/wangxiao/miniconda3/envs/cl/bin/deepspeed --master_port 51459 training/main.py  \
#     --data_path /mnt/petrelfs/wangxiao/DATA/LLM-CL-Benchmark/LLM-CL-Benchmark_5000 \
#     --dataset_name C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
#     --model_name_or_path /mnt/petrelfs/wangxiao/MODELS/vicuna-7b-v1.5 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 16 \
#     --max_prompt_len 1024 \
#     --max_ans_len 512 \
#     --learning_rate 1e-5 \
#     --weight_decay 0. \
#     --num_train_epochs 5,3,7,5,3,5,5,7 \
#     --gradient_accumulation_steps 16 \
#     --lr_scheduler_type cosine \
#     --num_warmup_steps 0 \
#     --seed 1234 \
#     --zero_stage 3 \
#     --deepspeed \
#     --print_loss \
#     --CL_method base \
#     --output_dir /mnt/petrelfs/wangxiao/LLM-Continual-Learning/CKPT/naive/vicuna7b > vicuna-7b-naive.log 2>&1 &
