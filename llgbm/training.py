"""Training utilities for LLGBM multi-task learning.

This module provides reusable components for training LoRA generators
with both weight supervision and behavioral (delta) supervision.
"""

import gc
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from llgbm.functional import FunctionalLoRA, compute_delta_differentiable


@dataclass
class TrainingConfig:
    """Configuration for multi-task training."""

    # Model settings
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    use_small_model: bool = True
    dtype: str = "bfloat16"

    # LoRA settings
    lora_rank: int = 8
    lora_alpha: int = 16

    # Training settings
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 10
    max_steps: int = 1000
    eval_steps: int = 100
    save_steps: int = 200
    log_steps: int = 10
    max_grad_norm: float = 1.0

    # Loss weighting
    lambda_delta: float = 0.1
    lambda_weight: float = 1.0

    # Delta computation
    num_probes: int = 3
    max_probe_length: int = 128
    compute_delta_every_n_steps: int = 1

    # Paths
    checkpoint_dir: str = "data/teacher_checkpoints"
    delta_cache_dir: str = "deltas"
    output_dir: str = "outputs/training"
    generator_checkpoint: Optional[str] = None

    # Logging
    use_wandb: bool = False
    wandb_project: str = "llgbm"
    wandb_run_name: Optional[str] = None

    # Architecture (filled based on model choice)
    hidden_size: int = 896
    intermediate_size: int = 4864
    num_layers: int = 24
    num_heads: int = 14
    num_kv_heads: int = 2

    def __post_init__(self):
        if not self.use_small_model:
            self.base_model = "Qwen/Qwen2.5-1.5B"
            self.hidden_size = 1536
            self.intermediate_size = 8960
            self.num_layers = 28
            self.num_heads = 12
            self.num_kv_heads = 2
            self.lora_rank = 16
            self.lora_alpha = 32

    def save(self, path: str):
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        """Load config from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class TrainingState:
    """Track training progress."""

    step: int = 0
    epoch: int = 0
    best_loss: float = float("inf")
    loss_history: List[float] = field(default_factory=list)
    loss_delta_history: List[float] = field(default_factory=list)
    loss_weight_history: List[float] = field(default_factory=list)
    lr_history: List[float] = field(default_factory=list)
    grad_norm_history: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingState":
        return cls(**data)


class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task training.

    Loss = lambda_weight * L_weight + lambda_delta * L_delta

    Where:
    - L_weight: MSE between generated and teacher LoRA tokens
    - L_delta: MSE between predicted and teacher delta activations
    """

    def __init__(
        self,
        lambda_weight: float = 1.0,
        lambda_delta: float = 0.1,
        normalize_delta: bool = True,
    ):
        """
        Args:
            lambda_weight: Weight for token/weight MSE loss
            lambda_delta: Weight for delta MSE loss
            normalize_delta: If True, normalize deltas before computing loss
        """
        super().__init__()
        self.lambda_weight = lambda_weight
        self.lambda_delta = lambda_delta
        self.normalize_delta = normalize_delta

    def forward(
        self,
        delta_pred: torch.Tensor,
        delta_teacher: torch.Tensor,
        tokens_pred: Optional[torch.Tensor] = None,
        tokens_teacher: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.

        Args:
            delta_pred: Predicted delta activations (B, hidden_size)
            delta_teacher: Teacher delta activations (B, hidden_size)
            tokens_pred: Predicted LoRA tokens (optional)
            tokens_teacher: Teacher LoRA tokens (optional)

        Returns:
            Dict with 'loss', 'loss_weight', 'loss_delta'
        """
        losses = {}

        # Delta loss
        if self.normalize_delta:
            delta_pred_norm = F.normalize(delta_pred, dim=-1)
            delta_teacher_norm = F.normalize(delta_teacher, dim=-1)
            loss_delta = F.mse_loss(delta_pred_norm, delta_teacher_norm)
        else:
            loss_delta = F.mse_loss(delta_pred, delta_teacher)

        losses["loss_delta"] = loss_delta

        # Weight loss (if provided)
        if tokens_pred is not None and tokens_teacher is not None:
            mask = ~torch.isnan(tokens_teacher)
            if mask.any():
                loss_weight = F.mse_loss(tokens_pred[mask], tokens_teacher[mask])
            else:
                loss_weight = torch.tensor(0.0, device=delta_pred.device)
            losses["loss_weight"] = loss_weight
        else:
            loss_weight = torch.tensor(0.0, device=delta_pred.device)
            losses["loss_weight"] = loss_weight

        # Combined loss
        total_loss = self.lambda_weight * loss_weight + self.lambda_delta * loss_delta
        losses["loss"] = total_loss

        return losses


class DeltaOnlyLoss(nn.Module):
    """
    Delta-only loss for Phase 5 experiments.

    Loss = L_delta(delta_pred, delta_teacher)

    No weight supervision - purely behavioral matching.
    """

    def __init__(self, normalize: bool = True, loss_type: str = "mse"):
        """
        Args:
            normalize: If True, normalize deltas before loss
            loss_type: "mse" or "cosine"
        """
        super().__init__()
        self.normalize = normalize
        self.loss_type = loss_type

    def forward(
        self,
        delta_pred: torch.Tensor,
        delta_teacher: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute delta-only loss."""
        if self.loss_type == "cosine":
            # Cosine similarity loss: 1 - cos_sim
            cos_sim = F.cosine_similarity(delta_pred, delta_teacher, dim=-1)
            loss = (1 - cos_sim).mean()
        else:
            # MSE loss
            if self.normalize:
                delta_pred = F.normalize(delta_pred, dim=-1)
                delta_teacher = F.normalize(delta_teacher, dim=-1)
            loss = F.mse_loss(delta_pred, delta_teacher)

        return {"loss": loss, "loss_delta": loss}


def compute_delta_for_batch(
    generator: nn.Module,
    functional_lora: FunctionalLoRA,
    base_activation: torch.Tensor,
    probe_tokens: List[torch.Tensor],
    probe_masks: List[torch.Tensor],
    condition_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Compute delta activations for a batch of generated LoRAs.

    Args:
        generator: Generator model that outputs LoRA weights
        functional_lora: FunctionalLoRA wrapper for differentiable application
        base_activation: Pre-computed base model activation
        probe_tokens: Tokenized probe inputs
        probe_masks: Attention masks for probes
        condition_ids: Condition token IDs (B, seq_len)
        attention_mask: Condition attention mask (B, seq_len)
        compute_dtype: Dtype for computation

    Returns:
        Delta activations (B, hidden_size)
    """
    batch_size = condition_ids.shape[0]
    device = condition_ids.device

    # Generate LoRA weights
    lora_weights_batch = generator(condition_ids, attention_mask)

    # Compute delta for each sample with memory cleanup
    deltas = []
    for i in range(batch_size):
        lora_weights = lora_weights_batch[i]
        delta = compute_delta_differentiable(
            functional_lora=functional_lora,
            lora_weights=lora_weights,
            base_activation=base_activation,
            probe_tokens=probe_tokens,
            probe_masks=probe_masks,
        )
        deltas.append(delta)

        # Clear intermediate activations from functional_call
        if device.type == "cuda" and i < batch_size - 1:
            torch.cuda.empty_cache()

    return torch.stack(deltas)


def save_checkpoint(
    generator: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    state: TrainingState,
    config: TrainingConfig,
    suffix: str = "",
) -> Path:
    """Save training checkpoint."""
    checkpoint_path = Path(config.output_dir) / f"checkpoint{suffix}.pt"

    torch.save(
        {
            "generator_state_dict": generator.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "training_state": state.to_dict(),
            "config": asdict(config),
        },
        checkpoint_path,
    )

    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    generator: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> TrainingState:
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    generator.load_state_dict(checkpoint["generator_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    state = TrainingState.from_dict(checkpoint["training_state"])
    return state


def train_step(
    batch: Dict[str, torch.Tensor],
    generator: nn.Module,
    functional_lora: FunctionalLoRA,
    base_activation: torch.Tensor,
    probe_tokens: List[torch.Tensor],
    probe_masks: List[torch.Tensor],
    criterion: nn.Module,
    config: TrainingConfig,
    compute_dtype: torch.dtype,
) -> Dict[str, float]:
    """
    Single training step.

    Returns dict of loss values (not scaled by accumulation).
    """
    device = next(generator.parameters()).device

    # Move batch to device
    condition_ids = batch["condition_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    delta_teacher = batch["delta_teacher"].to(device)

    # Get teacher tokens if available
    tokens_teacher = batch.get("tokens")
    if tokens_teacher is not None:
        tokens_teacher = tokens_teacher.to(device)

    # Forward pass
    device_type = "cuda" if device.type == "cuda" else "cpu"
    with torch.autocast(device_type=device_type, dtype=compute_dtype):
        delta_pred = compute_delta_for_batch(
            generator=generator,
            functional_lora=functional_lora,
            base_activation=base_activation,
            probe_tokens=probe_tokens,
            probe_masks=probe_masks,
            condition_ids=condition_ids,
            attention_mask=attention_mask,
            compute_dtype=compute_dtype,
        )

        # Compute loss
        if isinstance(criterion, MultiTaskLoss) and tokens_teacher is not None:
            # Get predicted tokens if generator outputs them
            tokens_pred = getattr(generator, "last_tokens", None)
            losses = criterion(
                delta_pred=delta_pred.float(),
                delta_teacher=delta_teacher.float(),
                tokens_pred=tokens_pred,
                tokens_teacher=tokens_teacher.float() if tokens_teacher is not None else None,
            )
        else:
            losses = criterion(
                delta_pred=delta_pred.float(),
                delta_teacher=delta_teacher.float(),
            )

    # Scale loss for gradient accumulation
    scaled_loss = losses["loss"] / config.gradient_accumulation_steps
    scaled_loss.backward()

    return {k: v.item() for k, v in losses.items()}


def train(
    generator: nn.Module,
    dataloader: DataLoader,
    functional_lora: FunctionalLoRA,
    base_activation: torch.Tensor,
    probe_tokens: List[torch.Tensor],
    probe_masks: List[torch.Tensor],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: TrainingConfig,
    compute_dtype: torch.dtype = torch.bfloat16,
    callback: Optional[Callable[[TrainingState], None]] = None,
) -> TrainingState:
    """
    Main training loop.

    Args:
        generator: Generator model
        dataloader: Training data
        functional_lora: FunctionalLoRA for delta computation
        base_activation: Pre-computed base activation
        probe_tokens: Tokenized probes
        probe_masks: Probe attention masks
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        config: Training configuration
        compute_dtype: Dtype for mixed precision
        callback: Optional callback called after each step

    Returns:
        Final TrainingState
    """
    state = TrainingState()
    generator.train()

    pbar = tqdm(total=config.max_steps, desc="Training")

    accumulation_step = 0
    running_loss = 0.0
    running_loss_delta = 0.0
    running_loss_weight = 0.0

    while state.step < config.max_steps:
        for batch in dataloader:
            if state.step >= config.max_steps:
                break

            # Training step
            losses = train_step(
                batch=batch,
                generator=generator,
                functional_lora=functional_lora,
                base_activation=base_activation,
                probe_tokens=probe_tokens,
                probe_masks=probe_masks,
                criterion=criterion,
                config=config,
                compute_dtype=compute_dtype,
            )

            running_loss += losses["loss"]
            running_loss_delta += losses.get("loss_delta", 0.0)
            running_loss_weight += losses.get("loss_weight", 0.0)

            accumulation_step += 1

            # Gradient update
            if accumulation_step >= config.gradient_accumulation_steps:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    generator.parameters(),
                    config.max_grad_norm,
                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Record metrics
                avg_loss = running_loss / accumulation_step
                avg_loss_delta = running_loss_delta / accumulation_step
                avg_loss_weight = running_loss_weight / accumulation_step

                state.loss_history.append(avg_loss)
                state.loss_delta_history.append(avg_loss_delta)
                state.loss_weight_history.append(avg_loss_weight)
                state.lr_history.append(scheduler.get_last_lr()[0])
                state.grad_norm_history.append(
                    grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                )

                # Logging
                if state.step % config.log_steps == 0:
                    pbar.set_postfix(
                        {
                            "loss": f"{avg_loss:.4f}",
                            "L_d": f"{avg_loss_delta:.4f}",
                            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                        }
                    )

                # Save checkpoint
                if state.step > 0 and state.step % config.save_steps == 0:
                    save_checkpoint(
                        generator, optimizer, scheduler, state, config,
                        suffix=f"_step{state.step}"
                    )

                # Save best
                if avg_loss < state.best_loss:
                    state.best_loss = avg_loss
                    save_checkpoint(
                        generator, optimizer, scheduler, state, config,
                        suffix="_best"
                    )

                # Callback
                if callback is not None:
                    callback(state)

                # Reset
                accumulation_step = 0
                running_loss = 0.0
                running_loss_delta = 0.0
                running_loss_weight = 0.0

                state.step += 1
                pbar.update(1)

                # Memory cleanup
                if state.step % 50 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        state.epoch += 1

    pbar.close()
    save_checkpoint(generator, optimizer, scheduler, state, config, suffix="_final")

    return state


@torch.no_grad()
def evaluate(
    generator: nn.Module,
    dataloader: DataLoader,
    functional_lora: FunctionalLoRA,
    base_activation: torch.Tensor,
    probe_tokens: List[torch.Tensor],
    probe_masks: List[torch.Tensor],
    criterion: nn.Module,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, float]:
    """
    Evaluate generator.

    Returns dict of metrics.
    """
    generator.eval()
    device = next(generator.parameters()).device

    total_loss = 0.0
    total_loss_delta = 0.0
    total_samples = 0
    cosines = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        condition_ids = batch["condition_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        delta_teacher = batch["delta_teacher"].to(device)

        device_type = "cuda" if device.type == "cuda" else "cpu"
        with torch.autocast(device_type=device_type, dtype=compute_dtype):
            delta_pred = compute_delta_for_batch(
                generator=generator,
                functional_lora=functional_lora,
                base_activation=base_activation,
                probe_tokens=probe_tokens,
                probe_masks=probe_masks,
                condition_ids=condition_ids,
                attention_mask=attention_mask,
            )

        losses = criterion(
            delta_pred=delta_pred.float(),
            delta_teacher=delta_teacher.float(),
        )

        batch_size = condition_ids.shape[0]
        total_loss += losses["loss"].item() * batch_size
        total_loss_delta += losses.get("loss_delta", losses["loss"]).item() * batch_size
        total_samples += batch_size

        cos = F.cosine_similarity(delta_pred.float(), delta_teacher.float(), dim=-1)
        cosines.extend(cos.cpu().tolist())

    generator.train()

    import numpy as np

    return {
        "eval_loss": total_loss / total_samples,
        "eval_loss_delta": total_loss_delta / total_samples,
        "mean_cosine": float(np.mean(cosines)),
        "std_cosine": float(np.std(cosines)),
        "min_cosine": float(np.min(cosines)),
        "max_cosine": float(np.max(cosines)),
    }
