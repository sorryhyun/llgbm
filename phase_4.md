# Phase 4 — Multi-Task Training: Weight MSE + Delta Loss

## Goal
Integrate delta loss into the DnD training loop, creating a multi-task objective:
```
L = L_weights + λ * L_delta
```

## Prerequisites
- Phase 3 complete (differentiable delta computation)
- Phase 2 complete (dataset returns delta labels)
- Phase 0 baseline working

## Implementation Steps

### Step 1: Create Delta Loss Module

Create `llgbm/losses.py`:

```python
"""Loss functions for behavioral matching."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class DeltaLoss(nn.Module):
    """
    Loss for matching delta embeddings.

    Supports:
    - MSE loss (default)
    - Cosine similarity loss
    - Combined MSE + Cosine
    """

    def __init__(
        self,
        loss_type: str = "mse",
        normalize: bool = False,
    ):
        """
        Args:
            loss_type: One of "mse", "cosine", "combined"
            normalize: Whether to L2-normalize deltas before loss
        """
        super().__init__()
        self.loss_type = loss_type
        self.normalize = normalize

    def forward(
        self,
        delta_pred: torch.Tensor,
        delta_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss between predicted and target delta embeddings.

        Args:
            delta_pred: Predicted delta (B, hidden_size)
            delta_target: Target delta (B, hidden_size)

        Returns:
            Scalar loss
        """
        if self.normalize:
            delta_pred = F.normalize(delta_pred, p=2, dim=-1)
            delta_target = F.normalize(delta_target, p=2, dim=-1)

        if self.loss_type == "mse":
            return F.mse_loss(delta_pred, delta_target)

        elif self.loss_type == "cosine":
            # Cosine similarity loss: 1 - cos_sim
            cos_sim = F.cosine_similarity(delta_pred, delta_target, dim=-1)
            return (1 - cos_sim).mean()

        elif self.loss_type == "combined":
            mse = F.mse_loss(delta_pred, delta_target)
            cos_sim = F.cosine_similarity(delta_pred, delta_target, dim=-1)
            cosine_loss = (1 - cos_sim).mean()
            return mse + cosine_loss

        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class MultiTaskLoss(nn.Module):
    """
    Combined loss for weight reconstruction and delta matching.

    L = L_weights + lambda * L_delta

    With optional schedule for lambda.
    """

    def __init__(
        self,
        lambda_delta: float = 1.0,
        lambda_schedule: str = "constant",
        warmup_steps: int = 0,
        max_lambda: float = 1.0,
        delta_loss_type: str = "mse",
    ):
        """
        Args:
            lambda_delta: Weight for delta loss
            lambda_schedule: One of "constant", "linear", "cosine"
            warmup_steps: Steps before delta loss kicks in
            max_lambda: Maximum lambda value after warmup
            delta_loss_type: Type of delta loss
        """
        super().__init__()

        self.lambda_delta = lambda_delta
        self.lambda_schedule = lambda_schedule
        self.warmup_steps = warmup_steps
        self.max_lambda = max_lambda

        self.delta_loss_fn = DeltaLoss(loss_type=delta_loss_type)
        self.current_step = 0

    def get_lambda(self) -> float:
        """Get current lambda based on schedule."""
        if self.current_step < self.warmup_steps:
            if self.lambda_schedule == "constant":
                return 0.0
            elif self.lambda_schedule == "linear":
                return self.max_lambda * (self.current_step / self.warmup_steps)
            elif self.lambda_schedule == "cosine":
                import math
                progress = self.current_step / self.warmup_steps
                return self.max_lambda * (1 - math.cos(math.pi * progress)) / 2
        return self.max_lambda

    def forward(
        self,
        loss_weights: torch.Tensor,
        delta_pred: torch.Tensor,
        delta_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.

        Args:
            loss_weights: Weight reconstruction loss (already computed)
            delta_pred: Predicted delta embedding
            delta_target: Target delta embedding

        Returns:
            total_loss: Combined loss
            metrics: Dict with individual losses and lambda
        """
        current_lambda = self.get_lambda()

        loss_delta = self.delta_loss_fn(delta_pred, delta_target)
        total_loss = loss_weights + current_lambda * loss_delta

        # Compute metrics
        with torch.no_grad():
            cos_sim = F.cosine_similarity(delta_pred, delta_target, dim=-1).mean()
            delta_pred_norm = delta_pred.norm(dim=-1).mean()
            delta_target_norm = delta_target.norm(dim=-1).mean()

        metrics = {
            "loss_total": total_loss.item(),
            "loss_weights": loss_weights.item(),
            "loss_delta": loss_delta.item(),
            "lambda_delta": current_lambda,
            "delta_cosine_sim": cos_sim.item(),
            "delta_pred_norm": delta_pred_norm.item(),
            "delta_target_norm": delta_target_norm.item(),
        }

        return total_loss, metrics

    def step(self):
        """Increment step counter for schedule."""
        self.current_step += 1
```

### Step 2: Create Training Script

Create `scripts/train_with_delta.py`:

```python
"""Training script with delta supervision."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "Drag-and-Drop-LLMs")

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from accelerate import Accelerator
from tqdm import tqdm
import json

from workspace.dnd.model.decoderonly import HyperConvDecoderModel_SuperLarge
from workspace.dnd.tokenizer.register import Qwen2515LoRA_Tokenizer2D

from llgbm.delta import DeltaCache
from llgbm.dataset import Text2Qwen25LoRA_DeltaDataset
from llgbm.functional_lora import DifferentiableDeltaCompute
from llgbm.losses import MultiTaskLoss


def get_config():
    """Training configuration."""
    return {
        # Model
        "base_model": "Qwen/Qwen2.5-1.5B",
        "condition_model": "bert-base-uncased",
        "token_size": (18, 258),

        # Data
        "checkpoint_folder": "data/teacher_checkpoints",
        "delta_cache_dir": "deltas",
        "max_text_length": 512,

        # Training
        "batch_size": 4,
        "delta_batch_size": 2,  # Subset for delta loss (memory)
        "total_steps": 5000,
        "learning_rate": 3e-5,
        "warmup_steps": 100,
        "max_grad_norm": 1.0,

        # Delta loss
        "lambda_delta": 1.0,
        "lambda_schedule": "linear",
        "lambda_warmup_steps": 500,
        "delta_loss_type": "mse",

        # Logging
        "log_interval": 10,
        "save_interval": 500,
        "eval_interval": 100,

        # Output
        "output_dir": "outputs/delta_training",
    }


def create_model(config, device):
    """Create generator model."""
    condition_model = AutoModel.from_pretrained(config["condition_model"])

    model = HyperConvDecoderModel_SuperLarge(
        features=[
            (32, config["token_size"][0], config["token_size"][1]),
            (64, config["token_size"][0], config["token_size"][1]),
            (128, config["token_size"][0], config["token_size"][1]),
        ],
        condition_dim=(768, 16, 16),  # BERT hidden size
        extra_condition_module=condition_model,
        extractor_type="BERT",
    )

    return model


def create_dataloader(config):
    """Create training dataloader."""
    lora_tokenizer = Qwen2515LoRA_Tokenizer2D()
    text_tokenizer = AutoTokenizer.from_pretrained(config["condition_model"])
    delta_cache = DeltaCache(config["delta_cache_dir"])

    dataset = Text2Qwen25LoRA_DeltaDataset(
        checkpoint_folder=config["checkpoint_folder"],
        lora_tokenizer=lora_tokenizer,
        text_tokenizer=text_tokenizer,
        delta_cache=delta_cache,
        max_text_length=config["max_text_length"],
    )

    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
    )


def train(config):
    """Main training loop."""
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device

    # Create components
    model = create_model(config, device)
    dataloader = create_dataloader(config)

    # Delta compute module (for delta loss)
    delta_compute = DifferentiableDeltaCompute(
        base_model_name=config["base_model"],
        device=device,
        dtype=torch.float16,
    )

    # Loss function
    loss_fn = MultiTaskLoss(
        lambda_delta=config["lambda_delta"],
        lambda_schedule=config["lambda_schedule"],
        warmup_steps=config["lambda_warmup_steps"],
        max_lambda=config["lambda_delta"],
        delta_loss_type=config["delta_loss_type"],
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=0.01,
    )

    # Learning rate scheduler
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=config["total_steps"],
    )

    # Prepare for distributed
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    # Training loop
    model.train()
    global_step = 0
    running_metrics = {}

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    progress_bar = tqdm(total=config["total_steps"], desc="Training")

    while global_step < config["total_steps"]:
        for batch in dataloader:
            # Unpack batch
            tokens_teacher = batch["tokens"]
            condition_ids = batch["condition_ids"]
            attention_mask = batch["attention_mask"]
            delta_teacher = batch["delta"]

            # Forward pass through generator
            # DnD model returns loss directly when given targets
            loss_weights, tokens_pred = model(
                tokens_teacher,
                condition={
                    "input_ids": condition_ids,
                    "attention_mask": attention_mask,
                },
                return_output=True,
            )

            # Compute delta loss on subset of batch (memory optimization)
            delta_batch_size = min(config["delta_batch_size"], tokens_pred.size(0))

            # Detokenize predicted tokens to LoRA weights
            # This needs the detokenization bridge from Phase 3
            # For now, use a simplified version
            lora_tokenizer = Qwen2515LoRA_Tokenizer2D()

            delta_pred_list = []
            for i in range(delta_batch_size):
                # Detokenize to get LoRA weights (this needs to be differentiable!)
                # Simplified: directly use tokens as proxy for now
                # In practice, use the DifferentiableDetokenizer from Phase 3

                # Placeholder for actual detokenization
                lora_weights = {}  # detokenizer.detokenize(tokens_pred[i])

                # Compute delta
                delta_pred = delta_compute(lora_weights)
                delta_pred_list.append(delta_pred)

            if delta_pred_list:
                delta_pred_batch = torch.stack(delta_pred_list)
                delta_teacher_batch = delta_teacher[:delta_batch_size]

                # Compute combined loss
                total_loss, metrics = loss_fn(
                    loss_weights,
                    delta_pred_batch,
                    delta_teacher_batch,
                )
            else:
                total_loss = loss_weights
                metrics = {"loss_total": loss_weights.item(), "loss_weights": loss_weights.item()}

            # Backward pass
            accelerator.backward(total_loss)

            # Gradient clipping
            accelerator.clip_grad_norm_(model.parameters(), config["max_grad_norm"])

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Update lambda schedule
            loss_fn.step()

            # Update running metrics
            for k, v in metrics.items():
                if k not in running_metrics:
                    running_metrics[k] = []
                running_metrics[k].append(v)

            global_step += 1
            progress_bar.update(1)

            # Logging
            if global_step % config["log_interval"] == 0:
                avg_metrics = {k: sum(v) / len(v) for k, v in running_metrics.items()}
                log_str = f"Step {global_step}: "
                log_str += " | ".join([f"{k}={v:.4f}" for k, v in avg_metrics.items()])
                tqdm.write(log_str)
                running_metrics = {}

            # Save checkpoint
            if global_step % config["save_interval"] == 0:
                save_path = output_dir / f"checkpoint-{global_step}"
                accelerator.save_state(str(save_path))
                tqdm.write(f"Saved checkpoint to {save_path}")

            if global_step >= config["total_steps"]:
                break

    # Final save
    accelerator.save_state(str(output_dir / "final"))
    print(f"\nTraining complete! Final checkpoint saved to {output_dir / 'final'}")


if __name__ == "__main__":
    config = get_config()
    train(config)
```

### Step 3: Create Evaluation Script

Create `scripts/evaluate.py`:

```python
"""Evaluate trained models."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "Drag-and-Drop-LLMs")

import argparse
import torch
import numpy as np
from tqdm import tqdm
from safetensors.torch import save_file
from transformers import AutoTokenizer

from workspace.dnd.tokenizer.register import Qwen2515LoRA_Tokenizer2D
from llgbm.delta import DeltaCache, compute_adapter_delta
from llgbm.probes import create_generic_probes


def evaluate_delta_alignment(
    model,
    dataloader,
    delta_compute,
    device,
):
    """
    Evaluate how well generated LoRAs match teacher deltas.

    Metrics:
    - Delta MSE
    - Delta cosine similarity
    - Delta norm ratio
    """
    model.eval()

    all_cosine_sims = []
    all_mse = []
    all_norm_ratios = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            tokens_teacher = batch["tokens"].to(device)
            condition_ids = batch["condition_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            delta_teacher = batch["delta"].to(device)

            # Generate LoRA
            tokens_pred = model.generate(
                condition={
                    "input_ids": condition_ids,
                    "attention_mask": attention_mask,
                }
            )

            # Compute predicted delta
            # (simplified - actual implementation needs detokenization)
            # delta_pred = delta_compute(detokenize(tokens_pred))

            # For now, use placeholder metrics
            delta_pred = torch.randn_like(delta_teacher)  # Placeholder

            # Compute metrics
            cos_sim = torch.nn.functional.cosine_similarity(
                delta_pred, delta_teacher, dim=-1
            )
            mse = ((delta_pred - delta_teacher) ** 2).mean(dim=-1)
            norm_ratio = delta_pred.norm(dim=-1) / (delta_teacher.norm(dim=-1) + 1e-8)

            all_cosine_sims.extend(cos_sim.cpu().tolist())
            all_mse.extend(mse.cpu().tolist())
            all_norm_ratios.extend(norm_ratio.cpu().tolist())

    # Compute summary statistics
    results = {
        "cosine_similarity": {
            "mean": np.mean(all_cosine_sims),
            "std": np.std(all_cosine_sims),
            "min": np.min(all_cosine_sims),
            "max": np.max(all_cosine_sims),
        },
        "mse": {
            "mean": np.mean(all_mse),
            "std": np.std(all_mse),
        },
        "norm_ratio": {
            "mean": np.mean(all_norm_ratios),
            "std": np.std(all_norm_ratios),
        },
    }

    return results


def evaluate_task_performance(
    generated_lora_path: str,
    base_model_name: str,
    eval_dataset,
    device,
):
    """
    Evaluate task performance of generated LoRA.

    This requires actually loading and running the generated LoRA.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    # Load base model with generated LoRA
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model = PeftModel.from_pretrained(model, generated_lora_path)
    model.eval()

    # Run evaluation
    # This is task-specific (e.g., accuracy for classification, BLEU for generation)

    results = {
        "accuracy": 0.0,  # Placeholder
        "perplexity": 0.0,  # Placeholder
    }

    return results


def compare_models(
    baseline_dir: str,
    delta_dir: str,
    test_dataloader,
    device,
):
    """
    Compare baseline DnD model vs delta-augmented model.
    """
    # Load models
    # baseline_model = load_model(baseline_dir)
    # delta_model = load_model(delta_dir)

    # Evaluate both
    # baseline_results = evaluate_delta_alignment(baseline_model, ...)
    # delta_results = evaluate_delta_alignment(delta_model, ...)

    # Compare
    comparison = {
        "baseline": {},
        "delta_augmented": {},
        "improvement": {},
    }

    return comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/test_checkpoints")
    parser.add_argument("--output", type=str, default="eval_results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and data
    # model = load_checkpoint(args.checkpoint)
    # dataloader = create_test_dataloader(args.data_dir)

    # Run evaluation
    # results = evaluate_delta_alignment(model, dataloader, delta_compute, device)

    # Save results
    import json
    # with open(args.output, "w") as f:
    #     json.dump(results, f, indent=2)

    print("Evaluation complete!")


if __name__ == "__main__":
    main()
```

### Step 4: Create Ablation Study Script

Create `scripts/ablation_study.py`:

```python
"""Run ablation studies on delta loss."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import subprocess
from itertools import product

# Ablation configurations
ABLATIONS = {
    "lambda_delta": [0.0, 0.1, 0.5, 1.0, 2.0],
    "delta_loss_type": ["mse", "cosine", "combined"],
    "probe_count": [3, 5, 10],
    "lambda_schedule": ["constant", "linear", "cosine"],
}

def run_ablation(config_overrides: dict, output_name: str):
    """Run a single ablation experiment."""
    # Build command
    cmd = ["python", "scripts/train_with_delta.py"]

    for key, value in config_overrides.items():
        cmd.extend([f"--{key}", str(value)])

    cmd.extend(["--output_dir", f"outputs/ablations/{output_name}"])

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)


def main():
    # Single-variable ablations
    for var_name, values in ABLATIONS.items():
        for value in values:
            config = {var_name: value}
            name = f"{var_name}_{value}"
            run_ablation(config, name)

    # Key interaction: lambda_delta x delta_loss_type
    for lambda_val, loss_type in product([0.1, 1.0], ["mse", "cosine"]):
        config = {
            "lambda_delta": lambda_val,
            "delta_loss_type": loss_type,
        }
        name = f"lambda{lambda_val}_loss{loss_type}"
        run_ablation(config, name)


if __name__ == "__main__":
    main()
```

## Key Hyperparameters

| Parameter | Recommended | Range to Try |
|-----------|-------------|--------------|
| `lambda_delta` | 1.0 | 0.1 - 10.0 |
| `delta_loss_type` | mse | mse, cosine, combined |
| `lambda_schedule` | linear | constant, linear, cosine |
| `lambda_warmup_steps` | 500 | 100 - 1000 |
| `delta_batch_size` | 2 | 1 - 4 (memory limited) |

## File Structure After Phase 4

```
llgbm/
├── llgbm/
│   ├── losses.py              # New: Loss functions
│   └── ...
├── scripts/
│   ├── train_with_delta.py    # New: Training script
│   ├── evaluate.py            # New: Evaluation
│   └── ablation_study.py      # New: Ablations
└── outputs/
    └── delta_training/
        ├── config.json
        ├── checkpoint-*/
        └── final/
```

## Acceptance Criteria

- [ ] Training runs stably with multi-task loss
- [ ] Lambda schedule works correctly (ramps up)
- [ ] Both `loss_weights` and `loss_delta` decrease
- [ ] Delta cosine similarity improves over training
- [ ] No NaN/Inf in losses
- [ ] Checkpoints save and load correctly

## Monitoring Metrics

Log these metrics to track training:

```python
metrics = {
    # Losses
    "loss_total",
    "loss_weights",
    "loss_delta",

    # Delta quality
    "delta_cosine_sim",
    "delta_pred_norm",
    "delta_target_norm",

    # Training dynamics
    "lambda_delta",
    "learning_rate",
    "grad_norm",
}
```

## Next Phase
Proceed to **Phase 5** for delta-only training experiments.
