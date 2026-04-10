"""
Qwen3.5 Claude-4.6-Opus-Reasoning-Distilled Chat
支持多轮上下文对话
对话记录保存到JSON文件
"""
import argparse
import torch
import time
import json
import os
from datetime import datetime
from transformers import Qwen3_5ForConditionalGeneration, AutoTokenizer, AutoProcessor, TextStreamer, BitsAndBytesConfig
from PIL import Image
import pynvml


MODEL_PATH = "/root/models"
MAX_CONTEXT_TOKENS = 10000

MODEL_SELECTION = {
    "4B": "/root/models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-v2",
    "9B": "/root/models/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled",
}

# PROMPT = "You are a helpful assistant."
PROMPT = '''你是一位专注于**思想与语言特征双重采集**的提问者。本次对话的核心目标有两个，且同等重要：
1. 完整梳理我的思想体系、认知、价值观、逻辑与立场；
2. **精准捕捉我的全部语言表达习惯**（包括但不限于：口头禅、常用连接词、句式结构、语气风格、口语/书面偏好、表达节奏、情绪表达方式、常用措辞、习惯性省略与重复等），在重要性上，语言表达特征甚至更为关键。

工作规则：
1. 全程只做一件事：**提问**。不解答、不评判、不总结、不安慰、不模仿我的说话方式。
2. 提问风格自然、流畅、不生硬，避免封闭式是非题，让我能够以最真实、最放松的状态完整表达，自然流露语言习惯。
3. 每轮仅提出1～2个递进式问题，围绕我上一轮表达的内容与风格，从多角度延伸，不重复、不跳脱。
4. 不主动结束对话，持续引导我充分表达，直到我明确停止。
5. 所有对话记录将用于完整还原我的思想与个人说话特征，你只需专注于高质量提问。'''


class GPUMonitor:
    def __init__(self):
        self.peak_memory_allocated = 0
        self.peak_memory_reserved = 0
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.has_nvml = True
        except:
            self.has_nvml = False
    
    def update_peak(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.max_memory_allocated() / 1024**3
            reserved = torch.cuda.max_memory_reserved() / 1024**3
            self.peak_memory_allocated = max(self.peak_memory_allocated, allocated)
            self.peak_memory_reserved = max(self.peak_memory_reserved, reserved)
    
    def get_stats(self):
        stats = {}
        if torch.cuda.is_available():
            stats["current_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
            stats["current_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
            stats["peak_allocated_gb"] = self.peak_memory_allocated
            stats["peak_reserved_gb"] = self.peak_memory_reserved
        if self.has_nvml:
            try:
                info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                stats["gpu_total_gb"] = info.total / 1024**3
                stats["gpu_used_gb"] = info.used / 1024**3
                stats["gpu_free_gb"] = info.free / 1024**3
            except:
                pass
        return stats
    
    def print_stats(self, prefix=""):
        stats = self.get_stats()
        print(f"{prefix}=== GPU Memory Stats ===")
        for k, v in stats.items():
            print(f"{prefix}  {k}: {v:.2f}")
    
    def reset_peak(self):
        self.peak_memory_allocated = 0
        self.peak_memory_reserved = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


monitor = GPUMonitor()


def load_model(model_path: str):
    print(f"Loading model from {model_path} ...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
        quantization_config=quantization_config,
    )
    model.eval()
    print("Model loaded.")
    return model, processor


def chat(model, processor, messages: list, max_new_tokens: int = 2048):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, return_tensors="pt").to("cuda:0")
    input_len = inputs.input_ids.shape[1]
    remaining_pct = (MAX_CONTEXT_TOKENS - input_len) / MAX_CONTEXT_TOKENS * 100
    print(f"[Input: {input_len} tokens, {remaining_pct:.1f}% context remaining]")
    
    streamer = TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
    stop_token_ids = [processor.tokenizer.eos_token_id]
    for stop_str in ["<|im_end|>", "<|endoftext|>"]:
        if stop_str in processor.tokenizer.vocab:
            stop_token_ids.append(processor.tokenizer.vocab[stop_str])
    stop_token_ids = list(set(stop_token_ids))
    
    start_time = time.time()
    monitor.reset_peak()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=stop_token_ids,
            streamer=streamer,
            repetition_penalty=1.1,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
    monitor.update_peak()
    elapsed = time.time() - start_time
    
    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    output_tokens = len(generated_ids[0])
    tokens_per_sec = output_tokens / elapsed if elapsed > 0 else 0
    
    print(f"\n[Output: {output_tokens} tokens, Total: {input_len + output_tokens} tokens, {remaining_pct:.1f}% remaining]")
    print(f"[Speed: {tokens_per_sec:.2f} tok/s]")
    monitor.print_stats(prefix="  ")
    
    return response


def save_conversation(messages: list, session_dir: str, session_id: str):
    os.makedirs(session_dir, exist_ok=True)
    filepath = os.path.join(session_dir, f"{session_id}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "messages": messages
        }, f, ensure_ascii=False, indent=2)
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Qwen3.5 Claude-4.6-Opus-Reasoning-Distilled Chat with Multi-turn Context")
    parser.add_argument("--model", type=str, default="9B", choices=["4B", "9B"], help="Model size (default: 9B)")
    parser.add_argument("--model_path", type=str, default=None, help="Model path (takes precedence over --model)")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--session_dir", type=str, default="./data", help="Directory to save conversation sessions")
    args = parser.parse_args()

    model_path = args.model_path if args.model_path else MODEL_SELECTION[args.model]
    print(f"Using model: {args.model} -> {model_path}")
    model, processor = load_model(model_path)
    
    messages = [
        {"role": "system", "content": PROMPT}
    ]
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("\n=== Multi-turn Chat Mode ===")
    print("输入 'quit' 退出，'/new' 开始新一轮对话\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            if messages:
                save_conversation(messages, args.session_dir, session_id)
                print(f"\nConversation saved to {args.session_dir}/{session_id}.json")
            print("\nBye!")
            break
        
        if not user_input:
            continue
        if user_input.lower() == "quit" or user_input.lower() == "q":
            if messages:
                save_conversation(messages, args.session_dir, session_id)
                print(f"Conversation saved to {args.session_dir}/{session_id}.json")
            print("Bye!")
            break
        if user_input == "/new":
            if len(messages) > 1:
                save_conversation(messages, args.session_dir, session_id)
                print(f"Conversation saved to {args.session_dir}/{session_id}.json")
            messages = [
                {"role": "system", "content": PROMPT}
            ]
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            print("Started new conversation.\n")
            continue
        
        messages.append({"role": "user", "content": user_input})
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, return_tensors="pt")
        current_tokens = inputs.input_ids.shape[1]
        max_new = max(1, MAX_CONTEXT_TOKENS - current_tokens)
        
        response = chat(model, processor, messages, max_new)
        messages.append({"role": "assistant", "content": response})
        print()


if __name__ == "__main__":
    main()
