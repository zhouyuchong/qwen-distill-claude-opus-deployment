'''
Author: zhouyuchong
Date: 2026-04-08 16:23:22
Description: vLLM Context Length & Performance Test (vllm==0.17.0)
'''
from vllm import LLM, SamplingParams

import os

# ===================== 核心修复：禁用torch.compile（省显存，0.18.0必开）=====================
os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
os.environ["VLLM_USE_V1"] = "1"

MODEL_SELECTION = {
    "4B": "/root/workspace/models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-v2",
    "9B": "/root/workspace/models/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled",
}

def main():
    import argparse
    parser = argparse.ArgumentParser(description="vLLM Context Test")
    parser.add_argument("--model", type=str, default="4B", choices=["4B", "9B"], help="模型大小")
    parser.add_argument("--max_tokens", type=int, default=512, help="最大生成token数")
    parser.add_argument("--test_tokens", type=int, default=10000, help="测试用token数")
    args = parser.parse_args()

    model_path = MODEL_SELECTION[args.model]
    print(f"Using model: {args.model} -> {model_path}")

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        # max_model_len=262144,
        max_model_len=65536,
        quantization="bitsandbytes",
        kv_cache_dtype="fp8",
        gpu_memory_utilization=0.75,
        language_model_only=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=4096,
        enforce_eager=True
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.1,
    )

    print("\n=== vLLM Throughput Test ===")
    
    for tokens in [1000, 5000, 10000, 20000, 50000, 100000]:
        if tokens > args.test_tokens:
            break
        long_prompt = "hello " * (tokens // 2)
        import time
        start = time.time()
        outputs = llm.generate(long_prompt, sampling_params)
        elapsed = time.time() - start
        throughput = (outputs[0].outputs[0].token_ids.__len__() if hasattr(outputs[0].outputs[0], 'token_ids') else 0) / elapsed
        print(f"Input: ~{tokens} tokens | Output: {len(outputs[0].outputs[0].token_ids) if hasattr(outputs[0].outputs[0], 'token_ids') else 0} tokens | Time: {elapsed:.2f}s | Throughput: {throughput:.2f} tok/s")

if __name__ == "__main__":
    main()
