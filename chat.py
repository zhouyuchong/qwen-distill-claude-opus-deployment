"""
Qwen3.5-27B Claude-4.6-Opus-Reasoning-Distilled Chat
支持多轮上下文对话
"""
import argparse
import torch
import time
from transformers import Qwen3_5ForConditionalGeneration, AutoTokenizer, AutoProcessor, TextStreamer, BitsAndBytesConfig
from PIL import Image
import pynvml


MODEL_PATH = "/workspace/models"
MAX_CONTEXT_TOKENS = 10000


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


def main():
    parser = argparse.ArgumentParser(description="Qwen3.5 Chat with Multi-turn Context")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    args = parser.parse_args()

    model, processor = load_model(args.model_path)
    
    messages = []
    print("\n=== Multi-turn Chat Mode ===")
    print("输入 'quit' 退出，'clear' 清空上下文\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Bye!")
            break
        if user_input.lower() == "clear":
            messages = []
            print("Context cleared.\n")
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
