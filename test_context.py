"""
RTX 3060 + Qwen3.5-9B Context Length & Performance Test
"""
import argparse
import torch
import time
import sys
from datetime import datetime
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from PIL import Image
import pynvml


class OutputLogger:
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, "w")
        self.stdout = sys.stdout
        sys.stdout = self
    
    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)
    
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        sys.stdout = self.stdout
        self.file.close()


MODEL_PATH = "/workspace/models"


def get_memory_stats():
    stats = {}
    if torch.cuda.is_available():
        stats["allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
        stats["reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        stats["gpu_total_gb"] = info.total / 1024**3
        stats["gpu_used_gb"] = info.used / 1024**3
        stats["gpu_free_gb"] = info.free / 1024**3
    except:
        pass
    return stats


def load_model():
    print("Loading model...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
        quantization_config=quantization_config,
    )
    model.eval()
    
    mem = get_memory_stats()
    print(f"Model loaded. Memory: {mem['allocated_gb']:.2f}GB allocated, {mem['reserved_gb']:.2f}GB reserved")
    print(f"GPU: {mem.get('gpu_total_gb', 0):.2f}GB total, {mem.get('gpu_used_gb', 0):.2f}GB used, {mem.get('gpu_free_gb', 0):.2f}GB free\n")
    
    return model, processor


def test_context_length(model, processor, num_turns=5, tokens_per_turn=256):
    print(f"=== Context Length Test: {num_turns} turns, ~{tokens_per_turn} tokens/turn ===")
    
    messages = [{"role": "user", "content": "Hello"}]
    for i in range(num_turns):
        messages.append({"role": "assistant", "content": f"Response {i+1} " * 50})
        messages.append({"role": "user", "content": f"Turn {i+2}"})
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, return_tensors="pt").to("cuda:0")
    input_len = inputs.input_ids.shape[1]
    
    print(f"Input length: {input_len} tokens ({input_len/1024:.1f}K)")
    
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    
    try:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=tokens_per_turn,
                use_cache=True,
                pad_token_id=processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                do_sample=False,
            )
        elapsed = time.time() - start_time
        output_len = output_ids.shape[1] - input_len
        throughput = output_len / elapsed if elapsed > 0 else 0
        
        mem = get_memory_stats()
        peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
        peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
        
        print(f"Output: {output_len} tokens in {elapsed:.2f}s ({throughput:.2f} tok/s)")
        print(f"Peak: {peak_allocated:.2f}GB allocated, {peak_reserved:.2f}GB reserved")
        print(f"GPU free: {mem.get('gpu_free_gb', 0):.2f}GB")
        print(f"Status: {'OK' if mem.get('gpu_free_gb', 0) > 0.5 else 'LOW MEMORY'}\n")
        
        return True, input_len, throughput
        
    except torch.OutOfMemoryError:
        print(f"OOM at input length {input_len}\n")
        return False, input_len, 0


def find_max_context(model, processor, max_turns=20):
    print("=== Finding Max Context Length ===\n")
    
    test_cases = [
        (5, 512),
        (10, 512),
        (15, 512),
        (20, 512),
        (25, 512),
        (30, 512),
        (35, 512),
        (40, 512),
        (50, 512),
        (60, 256),
        (80, 256),
        (100, 128),
    ]
    
    for turns, tokens in test_cases:
        print(f">>> Testing {turns} turns, {tokens} tokens/turn")
        success, input_len, throughput = test_context_length(model, processor, turns, tokens)
        if not success:
            print(f"\nMAX REACHED: ~{turns-5} turns, ~{input_len} tokens context")
            return turns - 5, input_len
        time.sleep(0.5)
    
    return max_turns, 0


def benchmark_throughput_by_context(model, processor):
    print("=== Throughput vs Context Length ===\n")
    
    test_cases = [
        (0, 512),
        (1, 512),
        (3, 512),
        (5, 512),
        (10, 256),
        (15, 128),
    ]
    
    results = []
    for prev_turns, output_tokens in test_cases:
        messages = [{"role": "user", "content": "Hello"}]
        for i in range(prev_turns):
            messages.append({"role": "assistant", "content": "Response " * 100})
            messages.append({"role": "user", "content": f"Turn {i+2}"})
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, return_tensors="pt").to("cuda:0")
        input_len = inputs.input_ids.shape[1]
        
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        
        try:
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=output_tokens,
                    use_cache=True,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    do_sample=False,
                )
            elapsed = time.time() - start_time
            throughput = output_tokens / elapsed
            
            mem = get_memory_stats()
            print(f"Context: {input_len:5d} tokens ({input_len/1024:.1f}K) | Throughput: {throughput:.2f} tok/s | Free: {mem.get('gpu_free_gb', 0):.2f}GB")
            results.append((input_len, throughput))
            
        except torch.OutOfMemoryError:
            print(f"Context: {input_len:5d} tokens - OOM")
            break
        
        time.sleep(0.5)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Context Length & Performance Test")
    parser.add_argument("--mode", type=str, default="both", choices=["max_context", "throughput", "both"])
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output or f"/root/results/test_context_{timestamp}.log"
    
    print(f"Logging to: {output_file}\n")
    
    with OutputLogger(output_file):
        print(f"=== Context Test Log {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        
        model, processor = load_model()
        
        if args.mode in ["max_context", "both"]:
            find_max_context(model, processor)
        
        if args.mode in ["throughput", "both"]:
            benchmark_throughput_by_context(model, processor)
        
        print("\n=== Summary ===")
        print("Recommend keeping context under ~16K tokens for stable performance on RTX 3060 12GB")


if __name__ == "__main__":
    main()
