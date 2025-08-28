# chat_interactive.py

import argparse
import logging
import torch
from transformers import pipeline, set_seed, AutoTokenizer, AutoModelForCausalLM

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="与NPU上的语言模型进行交互式聊天。")
    parser.add_argument(
        "--path", 
        type=str, 
        required=True, 
        help="包含已训练模型和tokenizer的目录路径。"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=256, 
        help="每次回复生成的最大新词元(token)数量。"
    )
    return parser.parse_args()

def setup_pipeline(path, device):
    """设置并返回一个transformers pipeline"""
    print(f"🚀 正在从 '{path}' 加载模型和tokenizer...")
    try:
        # 使用 AutoClass 自动识别并加载对应的模型和tokenizer，更具通用性
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,  # 在NPU和现代GPU上，bfloat16性能和稳定性更佳
            trust_remote_code=True
        )
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        print("👉 请确保提供的路径正确，并且目录中包含所有必要的文件（如config.json, pytorch_model.bin等）。")
        exit()

    # 如果tokenizer没有pad_token，通常可以将其设置为eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    print(f"🧠 模型已加载，正在部署到设备: {device}...")
    
    # 创建pipeline，这是与模型交互的推荐方式
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        # 关键参数：设置后，pipeline将只返回新生成的文本，无需我们手动处理
        return_full_text=False
    )
    print("✅ Pipeline准备就绪！")
    return generator

def main(args):
    # 自动检测NPU，如果不可用则使用CPU
    if torch.npu.is_available():
        device = "npu:0"
    else:
        device = "cpu"
        print("⚠️ 警告：未检测到NPU，将使用CPU运行，速度可能较慢。")
    
    generator = setup_pipeline(args.path, device)
    set_seed(42)

    # 使用一个列表来优雅地维护对话历史
    history = []
    
    print("\n" + "="*30)
    print("      欢迎来到NPU模型聊天室      ")
    print("="*30)
    print("👉 输入 'quit' 退出, 输入 'clear' 清空对话历史。")

    while True:
        user_prompt = input("You: ")
        
        if user_prompt.lower() in ["quit", "exit"]:
            print("👋 再见！")
            break
        if user_prompt.lower() == "clear":
            history = []
            print("\n✨ 对话历史已清空 ✨\n")
            continue

        # 1. 格式化包含历史记录的完整提示
        # 模板: "Human: ...\nAssistant: ...\nHuman: ...\nAssistant:"
        full_prompt = ""
        for h, a in history:
            full_prompt += f"Human: {h}\nAssistant: {a}\n"
        full_prompt += f"Human: {user_prompt}\nAssistant:"
        
        # 2. 模型生成回复
        print("Bot: ...✍️")
        response = generator(full_prompt, max_new_tokens=args.max_new_tokens)
        
        # 因为设置了 return_full_text=False，这里直接就是干净的新回复
        model_output = response[0]['generated_text'].strip()
        
        # 3. 打印回复并更新历史记录
        # 使用 \r 和 end="" 可以将 "...✍️" 覆盖掉，显得更流畅
        print(f"\rBot: {model_output}", end="\n\n") 
        history.append((user_prompt, model_output))

if __name__ == "__main__":
    # 屏蔽transformers的一些不必要的日志输出
    logging.getLogger("transformers").setLevel(logging.ERROR)
    args = parse_args()
    main(args)