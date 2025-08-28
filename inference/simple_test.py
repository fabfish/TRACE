import torch
import torch_npu  # <-- Added the necessary import
import deepspeed
import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to the fine-tuned model directory")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    args = parser.parse_args()

    # 1. Initialize DeepSpeed Distributed Backend
    deepspeed.init_distributed()
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    local_rank = args.local_rank
    
    # --- THIS IS THE CORRECTED LINE ---
    torch_npu.npu.set_device(local_rank)
    
    print(f"--> [Rank {local_rank}] Initialized. World size: {world_size}")

    # 2. Load the model directly using Hugging Face's from_pretrained.
    #    Using torch_dtype=torch.bfloat16 is crucial for memory.
    print(f"--> [Rank {local_rank}] Loading model from: {args.path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.path,
        torch_dtype=torch.bfloat16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.path)

    print(f"--> [Rank {local_rank}] Model loaded successfully on CPU.")

    # 3. Initialize the DeepSpeed Inference Engine.
    #    This will shard the model across all available NPUs.
    print(f"--> [Rank {local_rank}] Initializing DeepSpeed-Inference...")
    ds_engine = deepspeed.init_inference(
        model,
        mp_size=world_size,
        dtype=torch.bfloat16,
        replace_with_kernel_inject=True,
    )
    model = ds_engine.module
    print(f"--> [Rank {local_rank}] DeepSpeed-Inference engine ready.")

    # 4. Run a simple test generation on Rank 0
    if local_rank == 0:
        print("\n--> [Rank 0] Running a test generation...")
        
        # Manually set the device for the input tensors
        device = torch.device("npu", local_rank)
        prompt = "Hey, are you conscious? Can you talk to me?"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        output = model.generate(**inputs, max_new_tokens=20)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        
        print("\n" + "="*50)
        print("Test Prompt:")
        print(prompt)
        print("\nModel Response:")
        print(response)
        print("="*50)
        print("\n✅ Test successful!")

if __name__ == "__main__":
    main()