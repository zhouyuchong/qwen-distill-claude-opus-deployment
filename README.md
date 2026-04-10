# qwen-distill-claude-opus-deployment

Scripts for deploying [Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled](https://huggingface.co/Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled)

## Environment

- GPU: RTX 3060 (12GB)
- Model: Qwen3.5-9B (4bit quantized)
- CUDA: 12.8
- PyTorch: 2.10.0

## Performance

| Context Length | Throughput |
|----------------|------------|
| ~350 tokens    | ~10 tok/s  |
| ~1700 tokens   | ~9 tok/s   |

RTX 3060 建议上下文控制在 10000 tokens 以内。

## Scripts

所有脚本位于 `scripts/` 目录下。

### chat.py - 多轮对话

```bash
python scripts/chat.py
```

功能：
- 多轮上下文对话
- 输入/输出 token 数量显示
- 上下文剩余百分比显示
- 动态 max_new_tokens (根据剩余上下文空间调整)
- GPU 显存监控

命令：
- `quit` - 退出
- `/new` - 开始新一轮对话

### demo.py - 单次对话

```bash
python scripts/demo.py --prompt "your prompt"
```

### finetune.py - NekoQA-10K 微调

```bash
# 安装额外依赖
pip install datasets peft

# LoRA 微调 (推荐，显存占用小)
python scripts/finetune.py --model 4B --use_lora

# 全参数微调 (需要更多显存)
python scripts/finetune.py --model 4B --bf16
```

参数：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | 4B | 模型大小 (4B/9B) |
| `--model_path` | - | 自定义模型路径 |
| `--data_dir` | data/NekoQA-10K | 数据集目录 |
| `--output_dir` | ./output/nekoqa-finetune | 输出目录 |
| `--max_seq_length` | 1024 | 最大序列长度 |
| `--num_train_epochs` | 3 | 训练轮数 |
| `--per_device_train_batch_size` | 1 | 每设备 batch size |
| `--gradient_accumulation_steps` | 8 | 梯度累积步数 |
| `--learning_rate` | 1e-4 | 学习率 |
| `--use_lora` | False | 使用 LoRA 微调 |
| `--lora_rank` | 8 | LoRA rank |
| `--lora_alpha` | 16 | LoRA alpha |
| `--lora_target_modules` | all | LoRA 目标模块 |

### test_context.py - 上下文长度测试

```bash
cd /workspace/project
python scripts/test_context.py --mode both --output test.log
```

参数：
- `--mode throughput` - 吞吐量测试
- `--mode max_context` - 最大上下文测试
- `--mode both` - 两者都运行
- `--output <file>` - 日志输出路径

## Model Path

模型路径：`/root/models`

需要修改脚本中的 `MODEL_PATH` 或通过 `--model_path` 参数指定。

## Dependencies

```bash
pip install -r requirements.txt
```

使用 vllm 时，需要 apt install build-essential

## Dataset

[NekoQA-10K](https://huggingface.co/datasets/MindsRiverPonder/NekoQA-10K) 猫娘对话数据集用于微调。

**数据集需要自行下载：**
```bash
# 方法1: 使用 huggingface-cli
huggingface-cli download --repo-type dataset MindsRiverPonder/NekoQA-10K --local-dir data/NekoQA-10K

# 方法2: 手动下载
# https://huggingface.co/datasets/MindsRiverPonder/NekoQA-10K
```

下载后将数据集放在 `data/NekoQA-10K/` 目录。
