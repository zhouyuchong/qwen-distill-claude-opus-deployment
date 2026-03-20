"""
Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled Demo
支持纯文本对话和图片理解（多模态）
"""
import argparse
import torch
from transformers import Qwen3_5ForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image


MODEL_PATH = "/path-to-weight/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled"


def load_model(model_path: str):
    """加载模型和处理器"""
    print(f"Loading model from {model_path} ...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.")
    return model, processor


def text_chat(model, processor, prompt: str, max_new_tokens: int = 2048):
    """纯文本对话"""
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    # 只取新生成的部分
    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
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
    parser = argparse.ArgumentParser(description="Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled Demo")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="模型路径")
    parser.add_argument("--prompt", type=str, default=None, help="单次对话 prompt")
    parser.add_argument("--image", type=str, default=None, help="图片路径（可选）")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--interactive", action="store_true", help="交互式模式")
    args = parser.parse_args()

    model, processor = load_model(args.model_path)

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
