#!bin/bash
export TOKENIZERS_PARALLELISM=false
port=$(shuf -i25000-30000 -n1)
deepspeed --include=localhost:0 --master_port $port inference/ICL.py  \
    --data_path /data/datasets/TRACE-Benchmark/LLM-CL-Benchmark_5000 \
    --dataset_name FOMC,C-STANCE,NumGLUE-cm,NumGLUE-ds,ScienceQA,MeetingBank,Py150,20Minuten \
    --model_name_or_path /data/models/Qwen1.5-MoE-A2.7B-Chat \
    --inference_batch 4 \
    --max_prompt_len 3584 \
    --max_ans_len 512 \
    --seed 1234 \
    --deepspeed \
    --demonstrations_num 6 \
    --inference_output_path /home/fish/GitHub/TRACE/outputs-LLM-CL/ICL > /home/fish/GitHub/TRACE/outputs-LLM-CL/ICL/infer.log 2>&1 &

# for slurm, single gpu
# srun --partition=xai --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=asb-llama --kill-on-bad-exit=1 /home/yzy/.local/bin/deepspeed --master_port 51417 inference/ICL.py  \
#     --data_path /data/yzy/datasets/TRACE-Benchmark/LLM-CL-Benchmark_500 \
#     --dataset_name C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
#     --model_name_or_path /data/models/Qwen1.5-MoE-A2.7B-Chat \
#     --inference_batch 4 \
#     --max_prompt_len 3584 \
#     --max_ans_len 512 \
#     --seed 1234 \
#     --deepspeed \
#     --demonstrations_num 6 \
#     --inference_output_path /data/yzy/outputs-LLM-CL/ICL/qwen > llama2_7b_ICL_infer.log 2>&1 &




# srun --partition=xai --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=asb-llama --kill-on-bad-exit=1 /mnt/petrelfs/wangxiao/miniconda3/envs/cl/bin/deepspeed --master_port 51408 inference/ICL.py  \
#     --data_path /mnt/petrelfs/wangxiao/DATA/LLM-CL-Benchmark/LLM-CL-Benchmark_5000 \
#     --dataset_name C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
#     --model_name_or_path /mnt/petrelfs/wangxiao/MODELS/vicuna-7b-v1.5 \
#     --inference_batch 4 \
#     --max_prompt_len 3584 \
#     --max_ans_len 512 \
#     --seed 1234 \
#     --deepspeed \
#     --demonstrations_num 6 \
#     --inference_output_path /mnt/petrelfs/wangxiao/LLM-Continual-Learning/CKPT/ICL/vicuna-7b > vicuna-7b_ICL_infer.log 2>&1 &

