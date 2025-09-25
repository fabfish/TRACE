#!bin/bash
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:64,garbage_collection_threshold:0.5
port=$(shuf -i25000-30000 -n1)

# CL_METHOD="L2P"
CL_METHOD="upcycle"

# MODEL_PATH="/data/models/Llama-2-7b-chat-hf"
MODEL_PATH="/data/models/Llama-3.2-1B-Instruct"

DATASET_LIST="C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten"
EPOCH_LIST="5,3,7,5,3,5,5,7"

# --- OOM 修复配置 ---
MAX_LEN=1024
GRAD_CKPT="--gradient_checkpointing"

# --- Offload 配置 (按需启用) ---
# 如果上面的配置仍然OOM，取消下面一行的注释
OFFLOAD_FLAG="--offload"

# mkdir /data/yuzhiyuan/outputs_LLM-CL/qwen_full/cl/$cl_method
# mk all the parent directories if not exist

mkdir -p /data/yuzhiyuan/outputs_LLM-CL/Llama-3.2-1B-Instruct/cl/$CL_METHOD/even

deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port $port training/main.py \
    --data_path /data/datasets/TRACE-Benchmark/LLM-CL-Benchmark_500 \
    --dataset_name $DATASET_LIST \
    --model_name_or_path $MODEL_PATH \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --max_prompt_len $MAX_LEN \
    --max_ans_len 512 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --num_train_epochs $EPOCH_LIST \
    --gradient_accumulation_steps 1 \
    $GRAD_CKPT \
    $OFFLOAD_FLAG \
    --lr_scheduler_type cosine \
    --num_warmup_steps 0 \
    --seed 1234 \
    --zero_stage 2 \
    --deepspeed \
    --print_loss \
    --CL_method $CL_METHOD \
    --output_dir /data/yuzhiyuan/outputs_LLM-CL/Llama-3.2-1B-Instruct/cl/$CL_METHOD/even 2>&1 | tee /data/yuzhiyuan/outputs_LLM-CL/Llama-3.2-1B-Instruct/cl/$CL_METHOD/even/train.log &
