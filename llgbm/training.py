"""Training utilities for LLGBM multi-task learning.

This module provides reusable components for training LoRA generators
with both weight supervision and behavioral (delta) supervision.
"""

import gc
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from llgbm.functional import FunctionalLoRA, compute_delta_differentiable
from llgbm.losses import (
    WeightLoss,
    DeltaWLoss,
    MultiTaskLoss,
    DeltaGuidedLoss,
)

# Re-export for backward compatibility
__all__ = ["WeightLoss", "DeltaWLoss", "MultiTaskLoss", "DeltaGuidedLoss"]


@dataclass
class TrainingConfig:
    """Configuration for multi-task training."""

    # Model settings
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    use_small_model: bool = True
    dtype: str = "bfloat16"

    # LoRA settings
    lora_rank: int = 16
    lora_alpha: int = 32

    # Training settings
    batch_size: int = 2
    num_workers: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 10
    num_steps: int = 500  # Total optimizer steps
    eval_every_n_steps: int = 50
    save_every_n_steps: int = 100
    log_steps: int = 10
    max_grad_norm: float = 1.0

    # Loss weighting
    lambda_delta: float = 0.1
    lambda_weight: float = 1.0

    # Delta computation
    num_probes: int = 10
    max_probe_length: int = 256
    compute_delta_every_n_steps: int = 1
    delta_batch_probes: bool = True

    # Paths
    checkpoint_dir: str = "data/teacher_checkpoints"
    delta_cache_dir: str = "deltas"
    output_dir: str = "outputs/training"
    generator_checkpoint: Optional[str] = None

    # Logging
    use_wandb: bool = False
    wandb_project: str = "llgbm"
    wandb_run_name: Optional[str] = None

    # Text encoder settings
    text_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    freeze_text_encoder: bool = True
    text_encoder_pooling: str = "mean"
    num_prompts_per_adapter: int = 8
    embedding_cache_dir: str = "embeddings"

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


def compute_delta_for_batch(
    generator: nn.Module,
    functional_lora: FunctionalLoRA,
    base_activation: torch.Tensor,
    probe_tokens: List[torch.Tensor],
    probe_masks: List[torch.Tensor],
    condition_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    compute_dtype: torch.dtype = torch.bfloat16,
    return_weights: bool = False,
    batch_probes: bool = True,
) -> Tuple[torch.Tensor, Optional[List[Dict[str, torch.Tensor]]]]:
    """
    Compute delta activations for a batch of generated LoRAs.

    Args:
        generator: Generator model that outputs LoRA weights
        functional_lora: FunctionalLoRA wrapper for differentiable application
        base_activation: Pre-computed base model activation
        probe_tokens: Tokenized probe inputs
        probe_masks: Attention masks for probes
        condition_ids: Condition token IDs (B, seq_len) or (B, N, seq_len) for batched prompts
        attention_mask: Condition attention mask, same shape as condition_ids
        compute_dtype: Dtype for computation
        return_weights: If True, also return the generated LoRA weights

    Returns:
        Tuple of:
            - Delta activations (B, hidden_size)
            - Generated LoRA weights (if return_weights=True), else None
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
            batch_probes=batch_probes,
        )
        deltas.append(delta)

    if return_weights:
        return torch.stack(deltas), lora_weights_batch
    return torch.stack(deltas), None


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
    global_step: int = 0,
    weight_criterion: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """
    Single training step.

    Args:
        batch: Batch dict with condition_ids, attention_mask, delta_teacher, lora_weights
        generator: Generator model
        functional_lora: FunctionalLoRA for delta computation
        base_activation: Pre-computed base activation
        probe_tokens: Tokenized probes
        probe_masks: Probe attention masks
        criterion: Loss function (MultiTaskLoss or DeltaOnlyLoss)
        config: Training configuration
        compute_dtype: Dtype for mixed precision
        weight_criterion: Optional WeightLoss for direct weight supervision

    Returns:
        Dict of loss values (not scaled by accumulation).
    """
    device = next(generator.parameters()).device

    # Move batch to device
    delta_teacher = batch["delta_teacher"].to(device)

    # Handle both cached embeddings and tokenized prompts
    if "condition_embedding" in batch:
        # Using cached embeddings - pass directly to generator
        # Note: generator needs to handle this case
        condition_ids = batch["condition_embedding"].to(device)
        attention_mask = None
    else:
        condition_ids = batch["condition_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

    # Get teacher LoRA weights if available (for weight supervision)
    lora_weights_teacher = batch.get("lora_weights")  # List[Dict[str, Tensor]]

    # Forward pass
    device_type = "cuda" if device.type == "cuda" else "cpu"
    with torch.autocast(device_type=device_type, dtype=compute_dtype):
        need_weight = (
            weight_criterion is not None
            and lora_weights_teacher is not None
            and config.lambda_weight > 0
        )

        if isinstance(criterion, DeltaGuidedLoss):
            # Delta-guided training:
            # - Inner delta head steps are handled in train() before this function
            # - This function always does the full forward (delta_pred + delta_computed)
            # - delta_computed requires expensive LoRA application
            need_computed = (
                float(getattr(criterion, "lambda_computed", 0.0)) > 0.0
                or float(getattr(criterion, "lambda_consistency", 0.0)) > 0.0
            )

            # Only request LoRA weights when needed (computed delta and/or weight supervision).
            return_lora = need_computed or need_weight
            out = generator(
                condition_ids,
                attention_mask,
                return_delta=True,
                return_lora=return_lora,
            )
            delta_predicted = out["delta_pred"]

            delta_computed = None
            lora_weights_pred = out.get("lora_weights") if return_lora else None
            if need_computed and lora_weights_pred is not None:
                deltas = []
                for i in range(condition_ids.shape[0]):
                    delta_i = compute_delta_differentiable(
                        functional_lora=functional_lora,
                        lora_weights=lora_weights_pred[i],
                        base_activation=base_activation,
                        probe_tokens=probe_tokens,
                        probe_masks=probe_masks,
                        batch_probes=config.delta_batch_probes,
                    )
                    deltas.append(delta_i)
                delta_computed = torch.stack(deltas, dim=0)

            losses = criterion(
                delta_predicted=delta_predicted.float(),
                delta_computed=delta_computed.float() if delta_computed is not None else None,
                delta_teacher=delta_teacher.float(),
            )

            # Optional direct weight supervision (does not require delta computation).
            loss_weight = torch.tensor(0.0, device=device)
            if need_weight:
                if lora_weights_pred is None:
                    # We didn't request LoRA weights above; generate them now.
                    out_lora = generator(
                        condition_ids,
                        attention_mask,
                        return_delta=False,
                        return_lora=True,
                    )
                    lora_weights_pred = out_lora.get("lora_weights")
                if lora_weights_pred is not None:
                    loss_weight = weight_criterion(lora_weights_pred, lora_weights_teacher)
                    losses["loss_weight"] = loss_weight
                    losses["loss"] = losses["loss"] + config.lambda_weight * loss_weight
            losses.setdefault("loss_weight", loss_weight)

        # Weight-only path: skip delta forward entirely to save VRAM/compute.
        elif config.lambda_delta <= 0:
            lora_weights_pred = generator(condition_ids, attention_mask)
            loss_weight = (
                weight_criterion(lora_weights_pred, lora_weights_teacher)
                if need_weight
                else torch.tensor(0.0, device=device)
            )
            losses = {
                "loss_weight": loss_weight,
                "loss_delta": torch.tensor(0.0, device=device),
                "loss": config.lambda_weight * loss_weight,
            }
        else:
            # Compute delta and (optionally) generated weights
            delta_pred, lora_weights_pred = compute_delta_for_batch(
                generator=generator,
                functional_lora=functional_lora,
                base_activation=base_activation,
                probe_tokens=probe_tokens,
                probe_masks=probe_masks,
                condition_ids=condition_ids,
                attention_mask=attention_mask,
                compute_dtype=compute_dtype,
                return_weights=need_weight and isinstance(criterion, MultiTaskLoss),
                batch_probes=config.delta_batch_probes,
            )

            # Compute delta loss
            if isinstance(criterion, MultiTaskLoss):
                losses = criterion(
                    delta_pred=delta_pred.float(),
                    delta_teacher=delta_teacher.float(),
                    tokens_pred=None,  # We no longer use token-based weight loss
                    tokens_teacher=None,
                )

                # Compute weight loss separately if we have teacher weights
                if need_weight and lora_weights_pred is not None:
                    loss_weight = weight_criterion(lora_weights_pred, lora_weights_teacher)
                    losses["loss_weight"] = loss_weight
                    # Surface key-matching stats for debugging (helpful when teacher keys have PEFT prefixes).
                    dbg = getattr(weight_criterion, "last_debug", None)
                    if isinstance(dbg, dict):
                        losses["weight_key_overlap"] = float(dbg.get("key_overlap", 0))
                        losses["weight_matched_tensors"] = float(dbg.get("matched_tensors", 0))

                        if (
                            not getattr(train_step, "_warned_no_weight_match", False)
                            and float(dbg.get("matched_tensors", 0)) == 0
                        ):
                            print(
                                "[WARN] Weight supervision matched 0 tensors. "
                                "Teacher keys may be prefixed (e.g. 'base_model.model.'). "
                                "Check WeightLoss.last_debug for details."
                            )
                            train_step._warned_no_weight_match = True
                    # Recompute total loss with actual weight loss
                    losses["loss"] = (
                        config.lambda_weight * loss_weight
                        + config.lambda_delta * losses["loss_delta"]
                    )
            else:
                # DeltaOnlyLoss or other
                losses = criterion(
                    delta_pred=delta_pred.float(),
                    delta_teacher=delta_teacher.float(),
                )

    # Scale loss for gradient accumulation
    scaled_loss = losses["loss"] / config.gradient_accumulation_steps
    scaled_loss.backward()

    return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}


def _train_delta_head_inner_steps(
    generator: nn.Module,
    batch: Dict[str, torch.Tensor],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_inner_steps: int,
    compute_dtype: torch.dtype,
    max_grad_norm: float = 1.0,
) -> List[float]:
    """
    Train only the delta head for N inner steps (cheap, no LoRA application).

    This is used in delta_guided mode to train the delta predictor more frequently
    than the expensive LoRA computation.

    Args:
        generator: LoRAGeneratorWithDeltaHead model
        batch: Batch with condition_ids, attention_mask, delta_teacher
        criterion: DeltaGuidedLoss (only loss_pred is used)
        optimizer: Optimizer for all parameters
        num_inner_steps: Number of inner delta head updates
        compute_dtype: Dtype for mixed precision
        max_grad_norm: Gradient clipping threshold

    Returns:
        List of loss values for each inner step
    """
    device = next(generator.parameters()).device
    delta_teacher = batch["delta_teacher"].to(device)

    if "condition_embedding" in batch:
        condition_ids = batch["condition_embedding"].to(device)
        attention_mask = None
    else:
        condition_ids = batch["condition_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

    # Get delta head parameters to train
    delta_head_params = set(generator.delta_head.parameters())
    shared_proj_params = set(generator.shared_proj.parameters())
    trainable_params = delta_head_params | shared_proj_params

    # Freeze non-delta-head parameters temporarily
    frozen_params = []
    for p in generator.parameters():
        if p not in trainable_params and p.requires_grad:
            p.requires_grad = False
            frozen_params.append(p)

    inner_losses = []
    device_type = "cuda" if device.type == "cuda" else "cpu"

    try:
        for _ in range(num_inner_steps):
            optimizer.zero_grad()

            with torch.autocast(device_type=device_type, dtype=compute_dtype):
                # Only compute delta_pred (cheap, no LoRA generation)
                out = generator(
                    condition_ids,
                    attention_mask,
                    return_delta=True,
                    return_lora=False,
                )
                delta_predicted = out["delta_pred"]

                # Compute only the prediction loss (no computed delta needed)
                loss_pred = criterion._compute_loss(
                    delta_predicted.float(),
                    delta_teacher.float(),
                )
                loss = criterion.lambda_pred * loss_pred

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
            optimizer.step()

            inner_losses.append(loss.item())
    finally:
        # Restore frozen parameters
        for p in frozen_params:
            p.requires_grad = True

    return inner_losses


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
    num_steps: Optional[int] = None,
    weight_criterion: Optional[nn.Module] = None,
) -> TrainingState:
    """
    Main training loop (step-based).

    Args:
        generator: Generator model
        dataloader: Training data (will be cycled through)
        functional_lora: FunctionalLoRA for delta computation
        base_activation: Pre-computed base activation
        probe_tokens: Tokenized probes
        probe_masks: Probe attention masks
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        config: Training configuration
        compute_dtype: Dtype for mixed precision
        callback: Optional callback called after each update
        num_steps: Total optimizer steps. If None, uses num_epochs * len(dataloader) / grad_accum.
        weight_criterion: Optional WeightLoss for direct weight supervision

    Returns:
        Final TrainingState
    """
    from itertools import cycle

    state = TrainingState()
    generator.train()

    # Calculate total steps if not provided
    if num_steps is None:
        num_steps = config.num_steps

    pbar = tqdm(total=num_steps, desc="Training")
    data_iter = cycle(dataloader)

    accumulation_step = 0
    running_loss = 0.0
    running_loss_delta = 0.0
    running_loss_weight = 0.0
    update_count = 0

    # Check if using delta-guided training with inner steps
    is_delta_guided = isinstance(criterion, DeltaGuidedLoss)
    inner_steps = max(0, config.compute_delta_every_n_steps - 1) if is_delta_guided else 0

    while update_count < num_steps:
        batch = next(data_iter)

        # For delta-guided mode: train delta head for N-1 inner steps first
        # Then the main train_step will do the full forward (delta head + LoRA)
        if inner_steps > 0:
            _train_delta_head_inner_steps(
                generator=generator,
                batch=batch,
                criterion=criterion,
                optimizer=optimizer,
                num_inner_steps=inner_steps,
                compute_dtype=compute_dtype,
                max_grad_norm=config.max_grad_norm,
            )

        # Training step (for delta-guided, this is the 1 full step with LoRA computation)
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
            global_step=state.step,
            weight_criterion=weight_criterion,
        )

        running_loss += losses["loss"]
        running_loss_delta += losses.get("loss_delta", 0.0)
        running_loss_weight += losses.get("loss_weight", 0.0)
        accumulation_step += 1
        state.step += 1

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

            # Track best
            if avg_loss < state.best_loss:
                state.best_loss = avg_loss
                save_checkpoint(
                    generator, optimizer, scheduler, state, config,
                    suffix="_best"
                )

            # Reset accumulation
            accumulation_step = 0
            running_loss = 0.0
            running_loss_delta = 0.0
            running_loss_weight = 0.0
            update_count += 1

            # Logging
            pbar.set_postfix({
                "L": f"{avg_loss:.4f}",
                "L_d": f"{avg_loss_delta:.4f}",
                "L_w": f"{avg_loss_weight:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })
            pbar.update(1)

            # Callback
            if callback is not None:
                callback(state)

            # Periodic checkpoint
            if update_count % config.save_every_n_steps == 0:
                save_checkpoint(
                    generator, optimizer, scheduler, state, config,
                    suffix=f"_step{update_count}"
                )

            # Memory cleanup
            if update_count % 100 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

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
    total_loss_pred = 0.0
    total_loss_computed = 0.0
    total_samples = 0
    cosines = []
    cosines_pred = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        delta_teacher = batch["delta_teacher"].to(device)

        if "condition_embedding" in batch:
            condition_ids = batch["condition_embedding"].to(device)
            attention_mask = None
        else:
            condition_ids = batch["condition_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

        device_type = "cuda" if device.type == "cuda" else "cpu"
        with torch.autocast(device_type=device_type, dtype=compute_dtype):
            if isinstance(criterion, DeltaGuidedLoss):
                out = generator(
                    condition_ids,
                    attention_mask,
                    return_delta=True,
                    return_lora=True,
                )
                delta_predicted = out["delta_pred"]
                lora_weights_pred = out.get("lora_weights")

                deltas = []
                for i in range(condition_ids.shape[0]):
                    delta_i = compute_delta_differentiable(
                        functional_lora=functional_lora,
                        lora_weights=lora_weights_pred[i],
                        base_activation=base_activation,
                        probe_tokens=probe_tokens,
                        probe_masks=probe_masks,
                        batch_probes=getattr(getattr(generator, "cfg", None), "delta_batch_probes", True),
                    )
                    deltas.append(delta_i)
                delta_computed = torch.stack(deltas, dim=0)

                losses = criterion(
                    delta_predicted=delta_predicted.float(),
                    delta_computed=delta_computed.float(),
                    delta_teacher=delta_teacher.float(),
                )

                cos = F.cosine_similarity(delta_computed.float(), delta_teacher.float(), dim=-1)
                cos_pred = F.cosine_similarity(delta_predicted.float(), delta_teacher.float(), dim=-1)
            else:
                delta_pred, _ = compute_delta_for_batch(
                    generator=generator,
                    functional_lora=functional_lora,
                    base_activation=base_activation,
                    probe_tokens=probe_tokens,
                    probe_masks=probe_masks,
                    condition_ids=condition_ids,
                    attention_mask=attention_mask,
                    return_weights=False,
                    batch_probes=getattr(getattr(generator, "cfg", None), "delta_batch_probes", True),
                )

                losses = criterion(
                    delta_pred=delta_pred.float(),
                    delta_teacher=delta_teacher.float(),
                )

                cos = F.cosine_similarity(delta_pred.float(), delta_teacher.float(), dim=-1)
                cos_pred = None

        batch_size = condition_ids.shape[0]
        total_loss += losses["loss"].item() * batch_size
        total_loss_delta += losses.get("loss_delta", losses["loss"]).item() * batch_size
        total_loss_pred += float(losses.get("loss_pred", 0.0)) * batch_size
        total_loss_computed += float(losses.get("loss_computed", 0.0)) * batch_size
        total_samples += batch_size

        cosines.extend(cos.cpu().tolist())
        if cos_pred is not None:
            cosines_pred.extend(cos_pred.cpu().tolist())

    generator.train()

    import numpy as np

    result = {
        "eval_loss": total_loss / total_samples,
        "eval_loss_delta": total_loss_delta / total_samples,
        "mean_cosine": float(np.mean(cosines)),
        "std_cosine": float(np.std(cosines)),
        "min_cosine": float(np.min(cosines)),
        "max_cosine": float(np.max(cosines)),
    }

    # Extra metrics for delta-guided models.
    if cosines_pred:
        result.update(
            {
                "eval_loss_pred": total_loss_pred / total_samples,
                "eval_loss_computed": total_loss_computed / total_samples,
                "mean_cosine_pred": float(np.mean(cosines_pred)),
                "std_cosine_pred": float(np.std(cosines_pred)),
            }
        )

    return result
