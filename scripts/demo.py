"""
Qwen3.5 Claude-4.6-Opus-Reasoning-Distilled Demo
支持纯文本对话和图片理解（多模态）
"""
import argparse
import torch
import time
from transformers import Qwen3_5ForConditionalGeneration, AutoTokenizer, AutoProcessor, TextStreamer, BitsAndBytesConfig
from PIL import Image
from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo


MODEL_PATH = "/root/models"

MODEL_SELECTION = {
    "4B": "/root/models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-v2",
    "9B": "/root/models/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled",
}


class GPUMonitor:
    def __init__(self, gpu_index: int = 0):
        self.gpu_index = gpu_index
        self.peak_memory_allocated = 0
        self.peak_memory_reserved = 0
        self.has_nvml = False
        try:
            nvmlInit()
            self.handle = nvmlDeviceGetHandleByIndex(gpu_index)
            self.has_nvml = True
        except Exception:
            pass

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
                info = nvmlDeviceGetMemoryInfo(self.handle)
                stats["gpu_total_gb"] = info.total / 1024**3
                stats["gpu_used_gb"] = info.used / 1024**3
                stats["gpu_free_gb"] = info.free / 1024**3
            except Exception:
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


monitor = GPUMonitor(gpu_index=0)


def load_model(model_path: str):
    """加载模型和处理器"""
    print(f"Loading model from {model_path} ...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    
    # 2080Ti 每张 11GB，分配 9GB 给模型权重，剩余给 KV cache 和激活值
    # max_memory = {0: "8GB", 1: "8GB"}
    
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        # device_map="balanced",
        # device_map="auto",
        device_map={"": "cuda:0"},
        # max_memory=max_memory,
        trust_remote_code=True,
        quantization_config=quantization_config,
    )
    model.eval()
    print("Model loaded.")
    return model, processor


def text_chat(model, processor, prompt: str, max_new_tokens: int = 2048):
    """纯文本对话"""
    messages = [
        {"role": "system", "content": "你是一个严谨的AI助手, 请一步步进行逻辑推理, 并将你的思考过程包裹在<think>和</think>标签中"},
        {"role": "user", "content": prompt}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, return_tensors="pt").to("cuda:0")
    print(f"Generating (max {max_new_tokens} tokens)...")
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
    tokens_per_sec = len(generated_ids[0]) / elapsed if elapsed > 0 else 0
    print(f"\n[Generated {len(generated_ids[0])} tokens in {elapsed:.2f}s ({tokens_per_sec:.2f} tok/s)]")
    monitor.print_stats(prefix="  ")
    return response


def image_chat(model, processor, image_path: str, prompt: str, max_new_tokens: int = 2048):
    """图片理解对话"""
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def interactive_mode(model, processor):
    """交互式对话模式"""
    print("\n=== Interactive Mode ===")
    print("输入文本直接对话，输入 'img:<path> <prompt>' 进行图片理解")
    print("输入 'quit' 退出\n")
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.startswith("img:"):
            parts = user_input[4:].split(" ", 1)
            if len(parts) < 2:
                print("格式: img:<image_path> <prompt>")
                continue
            image_path, prompt = parts[0], parts[1]
            response = image_chat(model, processor, image_path, prompt)
        else:
            response = text_chat(model, processor, user_input)
        print(f"Assistant: {response}\n")


def main():
    parser = argparse.ArgumentParser(description="Qwen3.5 Claude-4.6-Opus-Reasoning-Distilled Demo")
    parser.add_argument("--model", type=str, default="9B", choices=["4B", "9B"], help="模型大小 (默认 9B)")
    parser.add_argument("--model_path", type=str, default=None, help="模型路径（优先于 --model）")
    parser.add_argument("--prompt", type=str, default=None, help="单次对话 prompt")
    parser.add_argument("--image", type=str, default=None, help="图片路径（可选）")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--interactive", action="store_true", help="交互式模式")
    args = parser.parse_args()

    model_path = args.model_path if args.model_path else MODEL_SELECTION[args.model]
    print(f"Using model: {args.model} -> {model_path}")
    model, processor = load_model(model_path)

    if args.interactive:
        interactive_mode(model, processor)
    elif args.prompt:
        if args.image:
            response = image_chat(model, processor, args.image, args.prompt, args.max_new_tokens)
        else:
            response = text_chat(model, processor, args.prompt, args.max_new_tokens)
        print(f"\nResponse:\n{response}")
    else:
        # 默认测试
        test_prompt = "请用中文解释什么是思维链推理(Chain-of-Thought Reasoning)，并给出一个简单的数学推理示例。"
        print(f"\n[Test] Prompt: {test_prompt}")
        response = text_chat(model, processor, test_prompt, args.max_new_tokens)
        print(f"\nResponse:\n{response}")


if __name__ == "__main__":
    main()
