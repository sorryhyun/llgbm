#!/usr/bin/env python3
"""
Train LoRA adapters for ablation studies.

Creates real LoRA adapters by fine-tuning Qwen2.5-0.5B on various tasks.
Each adapter is saved with its training prompts for conditioning.
"""

import json
import gc
import argparse
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm


@dataclass
class TaskConfig:
    """Configuration for a training task."""
    name: str
    data_file: str
    num_samples: int = 500  # samples per adapter
    num_adapters: int = 3   # adapters to train per task


TASKS = {
    "arc_e": TaskConfig("arc_e", "ARC-e_train.json", num_samples=450, num_adapters=5),
    "boolq": TaskConfig("boolq", "BoolQ_train.json", num_samples=1800, num_adapters=5),
    "gsm8k": TaskConfig("gsm8k", "GSM8K_train.json", num_samples=1400, num_adapters=5),
    "hellaswag": TaskConfig("hellaswag", "HellaSwag_train.json", num_samples=7900, num_adapters=5),
}


class SFTDataset(Dataset):
    """Simple SFT dataset for instruction tuning."""

    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 512,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Build chat format
        system = item.get("system", "You are a helpful assistant.")
        prompt = item["prompt"]
        response = item["response"]

        # Simple format: <|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>
        text = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Labels: same as input_ids, with padding tokens set to -100
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_task_data(data_dir: Path, task_config: TaskConfig) -> List[Dict]:
    """Load and prepare data for a task."""
    data_file = data_dir / task_config.data_file
    with open(data_file, "r") as f:
        data = json.load(f)
    return data


def train_adapter(
    model,
    tokenizer,
    train_data: List[Dict],
    output_dir: Path,
    adapter_name: str,
    num_epochs: int = 2,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_length: int = 384,
    device: str = "cuda",
):
    """Train a single LoRA adapter and save it."""

    print(f"\n  Training adapter: {adapter_name}")
    print(f"  Samples: {len(train_data)}")

    # Create dataset and dataloader
    dataset = SFTDataset(train_data, tokenizer, max_length=max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    num_training_steps = len(dataloader) * num_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(100, num_training_steps // 10),
        num_training_steps=num_training_steps,
    )

    # Training loop
    model.train()
    global_step = 0
    total_loss = 0

    progress = tqdm(total=num_training_steps, desc=f"  {adapter_name}")

    for epoch in range(num_epochs):
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            global_step += 1

            if global_step % 10 == 0:
                avg_loss = total_loss / global_step
                progress.set_postfix(loss=f"{avg_loss:.4f}")
            progress.update(1)

    progress.close()

    # Save adapter
    adapter_dir = output_dir / adapter_name
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Save LoRA weights
    model.save_pretrained(adapter_dir)

    # Save training prompts (for conditioning)
    prompts = [item["prompt"] for item in train_data[:128]]  # Save first 128 prompts
    with open(adapter_dir / "prompts.json", "w") as f:
        json.dump({"prompts": prompts, "task": adapter_name}, f, indent=2)

    avg_loss = total_loss / global_step
    print(f"  Final loss: {avg_loss:.4f}")

    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Train LoRA adapters for ablation studies")
    parser.add_argument("--data_dir", type=str, default="dnd_repo/prepare/data", help="Directory with training data")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory for adapters")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B", help="Base model to fine-tune")
    parser.add_argument("--tasks", type=str, nargs="+", default=["arc_e", "boolq", "gsm8k"], help="Tasks to train on")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of training epochs per adapter")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_batches", type=int, default=5, help="Number of non-overlapping batches per task")
    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Base model: {args.model_name}")
    print(f"Tasks: {args.tasks}")
    print(f"LoRA config: rank={args.lora_rank}, alpha={args.lora_alpha}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    # Track all trained adapters
    all_adapters = []

    for task_name in args.tasks:
        if task_name not in TASKS:
            print(f"Unknown task: {task_name}, skipping")
            continue

        task_config = TASKS[task_name]
        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"{'='*60}")

        # Load task data and shuffle with fixed seed for reproducibility
        task_data = load_task_data(data_dir, task_config)
        random.seed(args.seed)
        random.shuffle(task_data)
        print(f"Loaded {len(task_data)} samples (shuffled with seed={args.seed})")

        # Determine number of batches for this task
        num_batches = args.num_batches

        # Train multiple adapters with different data subsets
        for adapter_idx in range(num_batches):
            # Load fresh base model for each adapter
            print(f"\nLoading base model for adapter {adapter_idx + 1}/{num_batches}...")
            base_model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.bfloat16,
                device_map=device,
                trust_remote_code=True,
            )

            # Apply LoRA
            model = get_peft_model(base_model, lora_config)
            model.print_trainable_parameters()

            # Select data subset for this adapter (strict non-overlapping batches)
            # Divide data into num_batches equal parts
            total_samples = len(task_data)
            samples_per_batch = total_samples // num_batches

            start_idx = adapter_idx * samples_per_batch
            end_idx = start_idx + samples_per_batch

            # Last batch gets remaining samples
            if adapter_idx == num_batches - 1:
                end_idx = total_samples

            subset = task_data[start_idx:end_idx]
            print(f"  Batch {adapter_idx}: samples [{start_idx}:{end_idx}] ({len(subset)} samples)")

            adapter_name = f"{task_name}_{adapter_idx:03d}"

            # Train
            loss = train_adapter(
                model=model,
                tokenizer=tokenizer,
                train_data=subset,
                output_dir=output_dir / task_name,
                adapter_name=adapter_name,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                device=device,
            )

            all_adapters.append({
                "name": adapter_name,
                "task": task_name,
                "path": str(output_dir / task_name / adapter_name),
                "final_loss": loss,
                "num_samples": len(subset),
                "batch_idx": adapter_idx,
                "batch_range": [start_idx, end_idx],
            })

            # Cleanup
            del model, base_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Save manifest
    manifest = {
        "model_name": args.model_name,
        "lora_config": {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
        },
        "training_config": {
            "non_overlapping_batches": True,
            "num_batches_per_task": args.num_batches,
            "shuffle_seed": args.seed,
        },
        "adapters": all_adapters,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Trained {len(all_adapters)} adapters")
    print(f"Saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
