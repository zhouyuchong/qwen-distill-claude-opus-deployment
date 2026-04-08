<!--
 * @Author: zhouyuchong
 * @Date: 2026-03-30 17:17:23
 * @Description: 
 * @LastEditors: zhouyuchong
 * @LastEditTime: 2026-04-08 15:10:00
 -->
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

### chat.py - 多轮对话

```bash
python chat.py
```

功能：
- 多轮上下文对话
- 输入/输出 token 数量显示
- 上下文剩余百分比显示
- 动态 max_new_tokens (根据剩余上下文空间调整)
- GPU 显存监控

命令：
- `quit` - 退出
- `clear` - 清空上下文

### test_context.py - 上下文长度测试

```bash
cd /workspace/project
python test_context.py --mode both --output test.log
```

参数：
- `--mode throughput` - 吞吐量测试
- `--mode max_context` - 最大上下文测试
- `--mode both` - 两者都运行
- `--output <file>` - 日志输出路径

### demo.py - 单次对话

```bash
python demo.py --prompt "your prompt"
```

## Model Path

模型路径：`/workspace/models`

需要修改脚本中的 `MODEL_PATH` 或通过 `--model_path` 参数指定。
