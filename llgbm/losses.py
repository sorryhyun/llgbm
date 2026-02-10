"""Loss functions for LLGBM training.

This module provides loss functions for training LoRA generators with:
- Weight supervision (MSE between generated and teacher weights)
- Delta supervision (behavioral matching via hidden state deltas)
- Multi-task combinations of both
"""

from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _canonicalize_lora_key(key: str) -> str:
    """Normalize LoRA state_dict keys across save formats (e.g., PEFT prefixes)."""
    if key.startswith("base_model.model."):
        return key[len("base_model.model.") :]
    if key.startswith("base_model."):
        return key[len("base_model.") :]
    return key


class WeightLoss(nn.Module):
    """
    Direct MSE loss between predicted and teacher LoRA weights.

    This computes MSE between the generated LoRA A/B matrices and the
    teacher adapter weights, without requiring tokenization.
    """

    def __init__(self, reduction: str = "mean"):
        """
        Args:
            reduction: "mean" or "sum" for loss reduction
        """
        super().__init__()
        self.reduction = reduction
        self.last_debug: Dict[str, Any] = {}

    def forward(
        self,
        weights_pred: List[Dict[str, torch.Tensor]],
        weights_teacher: List[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Compute MSE between predicted and teacher LoRA weights.

        Args:
            weights_pred: List of weight dicts (one per batch item)
            weights_teacher: List of teacher weight dicts (one per batch item)

        Returns:
            Scalar MSE loss
        """
        total_loss = 0.0
        count = 0
        matched_tensors = 0
        matched_keys = 0
        key_overlap = 0

        pred_total_keys = 0
        teacher_total_keys = 0

        for pred, teacher in zip(weights_pred, weights_teacher):
            pred_canon = {_canonicalize_lora_key(k): v for k, v in pred.items()}
            teacher_canon = {_canonicalize_lora_key(k): v for k, v in teacher.items()}

            pred_keys = set(pred_canon.keys())
            teacher_keys = set(teacher_canon.keys())
            pred_total_keys += len(pred_keys)
            teacher_total_keys += len(teacher_keys)
            key_overlap += len(pred_keys & teacher_keys)

            for key in pred_canon.keys():
                if key in teacher_canon:
                    matched_keys += 1
                    pred_w = pred_canon[key].float()
                    teach_w = teacher_canon[key].float().to(pred_w.device)

                    # Only compute loss if shapes match
                    if pred_w.shape == teach_w.shape:
                        matched_tensors += 1
                        if self.reduction == "mean":
                            total_loss += F.mse_loss(pred_w, teach_w, reduction="sum")
                            count += pred_w.numel()
                        else:
                            total_loss += F.mse_loss(pred_w, teach_w, reduction="sum")
                            count += 1

        # Store lightweight debug stats for sanity-checking weight supervision.
        self.last_debug = {
            "pred_total_keys": pred_total_keys,
            "teacher_total_keys": teacher_total_keys,
            "key_overlap": key_overlap,
            "matched_keys": matched_keys,
            "matched_tensors": matched_tensors,
            "matched_numel": count if self.reduction == "mean" else None,
        }

        if count == 0:
            # Return a tensor that requires grad for backward compatibility
            device = "cpu"
            if weights_pred and weights_pred[0]:
                device = next(iter(weights_pred[0].values())).device
            return torch.zeros((), device=device, requires_grad=True)

        if self.reduction == "mean":
            return total_loss / count
        return total_loss


class DeltaWLoss(nn.Module):
    """
    MSE loss on effective weight updates DW = B @ A * scaling.

    Instead of comparing raw (A, B) matrices (which suffer from gauge ambiguity—
    many A,B pairs produce the same function), this loss compares the actual
    weight perturbation applied to the base model. This is both canonical
    (gauge-invariant) and cheap (matrix multiply, no forward pass required).
    """

    def __init__(
        self,
        lora_alpha: int = 16,
        lora_rank: int = 8,
        reduction: str = "mean",
    ):
        """
        Args:
            lora_alpha: LoRA alpha for scaling
            lora_rank: LoRA rank for scaling
            reduction: "mean" or "sum" for loss reduction
        """
        super().__init__()
        self.scaling = lora_alpha / lora_rank
        self.reduction = reduction
        self.last_debug: Dict[str, Any] = {}

    def forward(
        self,
        weights_pred: List[Dict[str, torch.Tensor]],
        weights_teacher: List[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Compute MSE between predicted and teacher effective weight updates.

        For each batch item, pairs lora_A and lora_B matrices by module prefix,
        computes DW = B @ A * scaling, then takes MSE(DW_pred, DW_teacher).

        Args:
            weights_pred: List of weight dicts (one per batch item)
            weights_teacher: List of teacher weight dicts (one per batch item)

        Returns:
            Scalar MSE loss
        """
        total_loss = 0.0
        count = 0
        matched_modules = 0

        for pred, teacher in zip(weights_pred, weights_teacher):
            pred_canon = {_canonicalize_lora_key(k): v for k, v in pred.items()}
            teacher_canon = {_canonicalize_lora_key(k): v for k, v in teacher.items()}

            # Group keys by module prefix: split on ".lora_A." or ".lora_B."
            pred_pairs = self._group_ab_pairs(pred_canon)
            teacher_pairs = self._group_ab_pairs(teacher_canon)

            for prefix in pred_pairs:
                if prefix not in teacher_pairs:
                    continue
                p_A, p_B = pred_pairs[prefix]
                t_A, t_B = teacher_pairs[prefix]
                if p_A is None or p_B is None or t_A is None or t_B is None:
                    continue

                p_A, p_B = p_A.float(), p_B.float()
                t_A, t_B = t_A.float(), t_B.float().to(p_A.device)

                # DW = B @ A * scaling
                dw_pred = p_B @ p_A * self.scaling
                dw_teacher = t_B @ t_A * self.scaling

                if dw_pred.shape != dw_teacher.shape:
                    continue

                matched_modules += 1
                if self.reduction == "mean":
                    total_loss += F.mse_loss(dw_pred, dw_teacher, reduction="sum")
                    count += dw_pred.numel()
                else:
                    total_loss += F.mse_loss(dw_pred, dw_teacher, reduction="sum")
                    count += 1

        self.last_debug = {"matched_modules": matched_modules, "matched_numel": count}

        if count == 0:
            device = "cpu"
            if weights_pred and weights_pred[0]:
                device = next(iter(weights_pred[0].values())).device
            return torch.zeros((), device=device, requires_grad=True)

        if self.reduction == "mean":
            return total_loss / count
        return total_loss

    @staticmethod
    def _group_ab_pairs(
        canon: Dict[str, torch.Tensor],
    ) -> Dict[str, tuple]:
        """Group canonicalized keys into (A, B) pairs by module prefix."""
        pairs: Dict[str, list] = {}  # prefix -> [A, B]
        for key, tensor in canon.items():
            if ".lora_A." in key:
                prefix = key.split(".lora_A.")[0]
                pairs.setdefault(prefix, [None, None])
                pairs[prefix][0] = tensor
            elif ".lora_B." in key:
                prefix = key.split(".lora_B.")[0]
                pairs.setdefault(prefix, [None, None])
                pairs[prefix][1] = tensor
        return {k: tuple(v) for k, v in pairs.items()}


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


class DeltaGuidedLoss(nn.Module):
    """
    Loss for training LoRA generator with delta prediction head.

    Loss = λ_pred * L(δ_predicted, δ_teacher)
         + λ_computed * L(δ_computed, δ_teacher)
         + λ_consistency * L(δ_computed, δ_predicted)

    Where:
    - δ_predicted: Fast prediction from delta head (no LoRA application)
    - δ_computed: Actual delta from applying generated LoRA weights
    - δ_teacher: Target delta from teacher adapters
    """

    def __init__(
        self,
        lambda_pred: float = 1.0,
        lambda_computed: float = 1.0,
        lambda_consistency: float = 0.5,
        normalize: bool = True,
        loss_type: str = "mse",
    ):
        """
        Args:
            lambda_pred: Weight for predicted delta vs teacher loss
            lambda_computed: Weight for computed delta vs teacher loss
            lambda_consistency: Weight for consistency between predicted and computed
            normalize: If True, normalize deltas before loss
            loss_type: "mse" or "cosine"
        """
        super().__init__()
        self.lambda_pred = lambda_pred
        self.lambda_computed = lambda_computed
        self.lambda_consistency = lambda_consistency
        self.normalize = normalize
        self.loss_type = loss_type

    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss between two delta tensors."""
        if self.loss_type == "cosine":
            cos_sim = F.cosine_similarity(pred, target, dim=-1)
            return (1 - cos_sim).mean()
        else:
            if self.normalize:
                pred = F.normalize(pred, dim=-1)
                target = F.normalize(target, dim=-1)
            return F.mse_loss(pred, target)

    def forward(
        self,
        delta_predicted: torch.Tensor,
        delta_computed: Optional[torch.Tensor],
        delta_teacher: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute delta-guided loss.

        Args:
            delta_predicted: (B, hidden_size) from delta prediction head
            delta_computed: (B, hidden_size) from applying LoRA, or None to skip
            delta_teacher: (B, hidden_size) target from teacher adapters

        Returns:
            Dict with loss components
        """
        losses = {}

        # Loss 1: Predicted delta vs teacher
        loss_pred = self._compute_loss(delta_predicted, delta_teacher)
        losses["loss_pred"] = loss_pred

        total_loss = self.lambda_pred * loss_pred

        # Loss 2: Computed delta vs teacher (if provided)
        if delta_computed is not None and self.lambda_computed > 0:
            loss_computed = self._compute_loss(delta_computed, delta_teacher)
            losses["loss_computed"] = loss_computed
            total_loss = total_loss + self.lambda_computed * loss_computed

            # Loss 3: Consistency between predicted and computed
            if self.lambda_consistency > 0:
                loss_consistency = self._compute_loss(delta_computed, delta_predicted.detach())
                losses["loss_consistency"] = loss_consistency
                total_loss = total_loss + self.lambda_consistency * loss_consistency

        losses["loss"] = total_loss
        losses["loss_delta"] = total_loss  # For compatibility

        return losses
