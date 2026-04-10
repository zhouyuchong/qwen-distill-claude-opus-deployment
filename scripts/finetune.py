#!/usr/bin/env python3
"""
NekoQA-10K 猫娘数据集微调脚本
对 Qwen3.5-4B 模型进行微调
"""
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    logging,
)
from transformers.integrations import is_ray_available


try:
    from ray import train as ray_train
    from ray.tune import is_session_enabled
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    is_session_enabled = lambda: False

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    LoraConfig = None
    get_peft_model = None
    TaskType = None


CURSOR_UP_ONE = "\x1b[1A"
ERASE_LINE = "\x1b[2K"


def erase_lines(n=1):
    for _ in range(n):
        sys.stdout.write(CURSOR_UP_ONE)
        sys.stdout.write(ERASE_LINE)
        sys.stdout.write("\r")


@dataclass
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def colorize(text: str, color: str) -> str:
    return f"{color}{text}{Colors.RESET}"


def print_header(text: str):
    print()
    print(colorize("=" * 70, Colors.CYAN))
    print(colorize(f"  {text}", Colors.BOLD + Colors.CYAN))
    print(colorize("=" * 70, Colors.CYAN))
    print()


def print_config(args, dataset_size: int, model_info: dict):
    print_header("训练配置")
    
    lora_info = ""
    if args.use_lora:
        lora_info = f" + LoRA (rank={args.lora_rank}, alpha={args.lora_alpha}, target={args.lora_target_modules})"
    
    cfg_items = [
        ("模型", f"{args.model} ({args.model_path}){lora_info}"),
        ("数据集", f"NekoQA-10K ({dataset_size} samples)"),
        ("输出目录", args.output_dir),
        ("", ""),
        ("训练参数", ""),
        ("  最大序列长度", str(args.max_seq_length)),
        ("  训练轮数 (epochs)", str(args.num_train_epochs)),
        ("  Batch Size", f"{args.per_device_train_batch_size} x {args.gradient_accumulation_steps} = {args.per_device_train_batch_size * args.gradient_accumulation_steps}"),
        ("  学习率", f"{args.learning_rate}"),
        ("  Warmup Steps", str(args.warmup_steps)),
        ("  Weight Decay", str(args.weight_decay)),
        ("  Max Grad Norm", str(args.max_grad_norm)),
        ("  LR Scheduler", args.lr_scheduler_type),
        ("", ""),
        ("模型信息", ""),
        ("  量化模式", "4-bit NF4 (bitsandbytes)"),
        ("  精度", "bf16" if args.bf16 else "fp16"),
        ("  Gradient Checkpointing", "启用" if args.gradient_checkpointing else "禁用"),
        ("  总参数量", model_info.get("total_params", "N/A")),
        ("  可训练参数量", model_info.get("trainable_params", "N/A")),
        ("  可训练比例", model_info.get("trainable_pct", "N/A")),
    ]
    
    for i, (key, val) in enumerate(cfg_items):
        if val == "":
            print()
            continue
        if key == "":
            continue
        print(f"  {colorize(key, Colors.YELLOW)}: {val}")
    
    print()
    print(colorize("-" * 70, Colors.DIM))
    print()


def print_stage(stage: str, status: str = "进行中"):
    icons = {"进行中": "⏳", "完成": "✓", "跳过": "○"}
    icon = icons.get(status, "○")
    status_color = {
        "进行中": Colors.CYAN,
        "完成": Colors.GREEN,
        "跳过": Colors.DIM,
    }.get(status, Colors.RESET)
    print(f"  {colorize(icon, status_color)} {stage}")


MODEL_SELECTION = {
    "4B": "/root/models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-v2",
    "9B": "/root/models/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled",
}


def load_nekoqa_dataset(data_path: str) -> Dataset:
    print_stage("加载数据集", "进行中")
    start = time.time()
    
    print(f"    路径: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    def format_example(item):
        instruction = item.get("instruction", "")
        output = item.get("output", "")
        return {
            "text": f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
        }
    
    formatted_data = [format_example(item) for item in data]
    dataset = Dataset.from_list(formatted_data)
    
    elapsed = time.time() - start
    print_stage("加载数据集", "完成")
    print(f"    {colorize(f'Loaded {len(dataset):,} samples', Colors.GREEN)} in {elapsed:.1f}s")
    print()
    return dataset


def tokenize_function(examples, tokenizer, max_seq_length: int):
    texts = examples["text"]
    model_inputs = tokenizer(
        texts,
        max_length=max_seq_length,
        truncation=True,
        padding="max_length",
        return_tensors=None,
    )
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs


def get_model_info(model) -> dict:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_size_gb = total_params * 2 / (1024**3)
    
    return {
        "total_params": f"{total_params:,} ({total_size_gb:.2f} GB)",
        "trainable_params": f"{trainable_params:,}",
        "trainable_pct": f"{trainable_params / total_params * 100:.2f}%",
    }


def load_model_and_tokenizer(model_path: str, model_size: str = "4B"):
    print_stage("加载模型", "进行中")
    start = time.time()
    
    print(f"    模型: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quantization_config,
    )
    
    model.config.use_cache = False
    model_info = get_model_info(model)
    
    elapsed = time.time() - start
    print_stage("加载模型", "完成")
    print(f"    {colorize('Model loaded', Colors.GREEN)} in {elapsed:.1f}s")
    print(f"    总参数: {model_info['total_params']} | 可训练: {model_info['trainable_params']} ({model_info['trainable_pct']})")
    print()
    return model, tokenizer, model_info


class ProgressTracker(TrainerCallback):
    def __init__(
        self,
        trainer,
        total_steps: int,
        num_epochs: int,
        log_interval: int = 1,
    ):
        self.trainer = trainer
        self.total_steps = total_steps
        self.num_epochs = num_epochs
        self.log_interval = log_interval
        
        self.start_time = None
        self.epoch_start_time = None
        self.current_epoch = 0
        self.step_in_epoch = 0
        self.last_log_step = 0
        
        self.best_loss = float('inf')
        self.loss_history = []
        
    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self.start_time = time.time()
        self.epoch_start_time = time.time()
        self._print_train_start(state)
        
    def on_epoch_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self.current_epoch = state.epoch
        self.step_in_epoch = 0
        self.epoch_start_time = time.time()
        erase_lines(6)
        self._print_epoch_header(state)
        
    def on_step(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self.step_in_epoch += 1
        
        if state.global_step % self.log_interval == 0 and state.global_step > 0:
            self._print_progress(state)
            
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is None:
            return
        self.last_log_step = state.global_step
        
        if "loss" in logs:
            self.loss_history.append(logs["loss"])
            if logs["loss"] < self.best_loss:
                self.best_loss = logs["loss"]
                
    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        epoch_time = time.time() - self.epoch_start_time
        avg_loss = sum(self.loss_history[-self.step_in_epoch:]) / len(self.loss_history[-self.step_in_epoch:]) if self.loss_history else 0
        erase_lines(8)
        self._print_epoch_summary(state, epoch_time, avg_loss)
        
    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        total_time = time.time() - self.start_time
        erase_lines(10)
        self._print_train_summary(total_time)
        
    def _print_train_start(self, state: TrainerState):
        print_header("开始训练")
        print(f"  {colorize('⏱  总步数:', Colors.YELLOW)} {state.max_steps:,}")
        print(f"  {colorize('📅 训练轮数:', Colors.YELLOW)} {self.num_epochs}")
        print(f"  {colorize('🔄 日志间隔:', Colors.YELLOW)} {self.log_interval} steps")
        print(f"  {colorize('💾 保存间隔:', Colors.YELLOW)} {self.trainer.args.save_steps} steps")
        print()
        print(colorize("-" * 70, Colors.DIM))
        print()
        
    def _print_epoch_header(self, state: TrainerState):
        epoch_pct = state.epoch / self.num_epochs * 100
        bar_len = 40
        filled = int(bar_len * state.epoch / self.num_epochs)
        bar = "█" * filled + "░" * (bar_len - filled)
        
        print(f"  {colorize('Epoch', Colors.CYAN)} {state.epoch:.1f}/{self.num_epochs} {colorize('│', Colors.DIM)} {colorize(bar, Colors.CYAN)} {epoch_pct:.1f}%")
        print()
        
    def _print_progress(self, state: TrainerState):
        if not self.loss_history:
            return
            
        elapsed = time.time() - self.start_time
        current_loss = self.loss_history[-1] if self.loss_history else 0
        avg_loss = sum(self.loss_history[-100:]) / min(len(self.loss_history), 100)
        
        eta_seconds = (elapsed / state.global_step) * (self.total_steps - state.global_step)
        eta_str = self._format_time(eta_seconds)
        
        steps_per_sec = state.global_step / elapsed if elapsed > 0 else 0
        
        lr = self.trainer.lr_scheduler.get_last_lr()[0] if self.trainer.lr_scheduler else 0
        
        erase_lines(5)
        
        metrics = [
            ("Step", f"{state.global_step:,}/{self.total_steps:,}"),
            ("Loss", f"{current_loss:.4f}"),
            ("Avg Loss", f"{avg_loss:.4f}"),
            ("Best", f"{self.best_loss:.4f}"),
            ("LR", f"{lr:.2e}"),
            ("Speed", f"{steps_per_sec:.2f} s/s"),
            ("ETA", eta_str),
        ]
        
        print(f"  {colorize('进度:', Colors.CYAN)}", end=" ")
        for i, (k, v) in enumerate(metrics):
            if i > 0:
                print(colorize(" │ ", Colors.DIM), end="")
            print(f"{colorize(k+':', Colors.YELLOW)} {v}", end="")
        print()
        print()
        
    def _print_epoch_summary(self, state: TrainerState, epoch_time: float, avg_loss: float):
        epoch_loss = self.loss_history[-1] if self.loss_history else avg_loss
        
        print(f"  {colorize('✓ Epoch', Colors.GREEN)} {state.epoch:.1f} {colorize('完成', Colors.GREEN)} | {colorize(f'时间: {self._format_time(epoch_time)}', Colors.DIM)} | {colorize(f'Loss: {epoch_loss:.4f}', Colors.YELLOW)}")
        print()
        
    def _print_train_summary(self, total_time: float):
        print_header("训练完成")
        print(f"  {colorize('⏱  总耗时:', Colors.YELLOW)} {self._format_time(total_time)}")
        print(f"  {colorize('📊 最终 Loss:', Colors.YELLOW)} {self.best_loss:.6f}")
        print(f"  {colorize('📝 步骤数:', Colors.YELLOW)} {self.total_steps}")
        print(f"  {colorize('🐱 猫娘微调完成!', Colors.CYAN)}")
        print()
        
    @staticmethod
    def _format_time(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            m, s = divmod(int(seconds), 60)
            return f"{m}m {s}s"
        else:
            h, m = divmod(int(seconds), 3600)
            return f"{h}h {m}m"


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.progress_callback = kwargs.pop("progress_callback", None)
        super().__init__(*args, **kwargs)
        if self.progress_callback:
            self.add_callback(self.progress_callback)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3.5 on NekoQA-10K dataset")
    parser.add_argument("--model", type=str, default="4B", choices=["4B", "9B"], help="Model size (default: 4B)")
    parser.add_argument("--model_path", type=str, default=None, help="Model path (takes precedence over --model)")
    parser.add_argument("--data_dir", type=str, default="data/NekoQA-10K", help="Dataset directory")
    parser.add_argument("--output_dir", type=str, default="./output/nekoqa-finetune", help="Output directory")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save steps")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Max number of checkpoints to keep")
    parser.add_argument("--bf16", action="store_true", default=True, help="Use bf16 precision")
    parser.add_argument("--no_bf16", action="store_false", dest="bf16", help="Disable bf16 precision")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Use gradient checkpointing")
    parser.add_argument("--max_grad_norm", type=float, default=0.3, help="Max gradient norm")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="LR scheduler type")
    parser.add_argument("--use_lora", action="store_true", default=False, help="使用 LoRA 微调 (需要安装 peft)")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str, default="all", help="LoRA target modules (comma-separated or 'all')")
    
    args = parser.parse_args()
    
    print(colorize("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║     🐱 NekoQA-10K Fine-tuning for Qwen3.5                    ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """, Colors.CYAN))
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    model_path = args.model_path if args.model_path else MODEL_SELECTION[args.model]
    data_path = os.path.join(args.data_dir, "NekoQA-10K.json")
    args.model_path = model_path
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    print_stage("阶段 1: 加载数据")
    print()
    dataset = load_nekoqa_dataset(data_path)
    
    print_stage("阶段 2: 加载模型")
    print()
    model, tokenizer, model_info = load_model_and_tokenizer(model_path, args.model)
    
    if args.use_lora:
        if not PEFT_AVAILABLE:
            raise ImportError("需要安装 peft 库: pip install peft")
        print_stage("应用 LoRA 配置")
        print()
        if args.lora_target_modules == "all":
            target_modules = "all-linear"
        else:
            target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        model_info = get_model_info(model)
        print()
    
    print_stage("阶段 3: Tokenize 数据")
    print()
    def tokenize(examples):
        return tokenize_function(examples, tokenizer, args.max_seq_length)
    
    print(f"    最大长度: {args.max_seq_length}")
    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    print(f"    {colorize('Tokenize 完成!', Colors.GREEN)}")
    print()
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt",
    )
    
    effective_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    total_steps = (len(tokenized_dataset) // effective_batch_size) * args.num_train_epochs
    
    print_stage("阶段 4: 配置训练参数")
    print()
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=not args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        prediction_loss_only=True,
        dataloader_drop_last=True,
    )
    
    progress_callback = ProgressTracker(
        trainer=None,
        total_steps=total_steps,
        num_epochs=args.num_train_epochs,
        log_interval=args.logging_steps,
    )
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        progress_callback=progress_callback,
    )
    progress_callback.trainer = trainer
    
    print_config(args, len(dataset), model_info)
    
    print_stage("阶段 5: 开始训练")
    print()
    print(colorize("  " + "─" * 66, Colors.DIM))
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print(colorize("\n⚠ 训练被用户中断", Colors.YELLOW))
        print(colorize("  正在保存当前检查点...", Colors.YELLOW))
    
    final_output_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_output_dir, exist_ok=True)
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    
    print()
    print(colorize(f"  ✓ 模型已保存到: {final_output_dir}", Colors.GREEN))
    print(colorize(f"  ✓ Tokenizer 已保存", Colors.GREEN))
    print()


if __name__ == "__main__":
    main()