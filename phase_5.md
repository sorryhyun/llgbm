# Phase 5 — Delta-Only Training (After Multi-Loss Works)

## Goal
Train models using only delta supervision, without MSE on LoRA weights. This tests whether behavioral signals alone are sufficient for learning useful LoRA generators.

## Prerequisites
- Phase 4 complete (multi-task training working)
- Evidence that delta loss provides useful signal

## Variants

### Variant B1: Generator Outputs LoRA Weights (Delta-Only Loss)

The generator still produces LoRA weights, but we only supervise via delta:
```
min_G || Δ(B + θ̂, B) - Δ* ||
```

**Risk:** Degenerate solutions (e.g., all-zero LoRA, or extreme weights that happen to produce similar delta)

**Mitigation:** Add regularizers

### Variant B2: Predict Delta Directly + Retrieval

Train a small model to predict delta embeddings, then retrieve/merge existing adapters at inference:
```
min_g || g(condition) - Δ* ||
```

At inference: retrieve top-k adapters by cosine similarity in delta space, merge them.

## Implementation Steps

### Step 1: Create Delta-Only Training

Create `llgbm/training/delta_only.py`:

```python
"""Delta-only training variants."""
import sys
sys.path.insert(0, "Drag-and-Drop-LLMs")

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class DeltaOnlyLoss(nn.Module):
    """
    Delta-only loss with regularization to prevent degenerate solutions.

    L = L_delta + α * L_reg

    Regularizers:
    - LoRA norm penalty (prevent extreme weights)
    - LoRA variance penalty (prevent collapsed solutions)
    - Optional: KL tether on probe logits
    """

    def __init__(
        self,
        delta_loss_type: str = "mse",
        lora_norm_weight: float = 0.01,
        lora_variance_weight: float = 0.001,
        logit_kl_weight: float = 0.0,
    ):
        super().__init__()

        self.delta_loss_type = delta_loss_type
        self.lora_norm_weight = lora_norm_weight
        self.lora_variance_weight = lora_variance_weight
        self.logit_kl_weight = logit_kl_weight

    def compute_delta_loss(
        self,
        delta_pred: torch.Tensor,
        delta_target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute delta matching loss."""
        if self.delta_loss_type == "mse":
            return F.mse_loss(delta_pred, delta_target)
        elif self.delta_loss_type == "cosine":
            cos_sim = F.cosine_similarity(delta_pred, delta_target, dim=-1)
            return (1 - cos_sim).mean()
        elif self.delta_loss_type == "combined":
            mse = F.mse_loss(delta_pred, delta_target)
            cos_sim = F.cosine_similarity(delta_pred, delta_target, dim=-1)
            return mse + (1 - cos_sim).mean()
        else:
            raise ValueError(f"Unknown loss type: {self.delta_loss_type}")

    def compute_lora_regularization(
        self,
        lora_weights: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute regularization terms on LoRA weights.

        Returns:
            reg_loss: Total regularization loss
            metrics: Individual regularization values
        """
        # Collect all LoRA parameters
        all_params = torch.cat([p.flatten() for p in lora_weights.values()])

        # Norm penalty (L2 regularization)
        norm_penalty = all_params.pow(2).mean()

        # Variance penalty (encourage diverse weights)
        variance_penalty = -all_params.var()  # Negative because we want high variance

        reg_loss = (
            self.lora_norm_weight * norm_penalty +
            self.lora_variance_weight * variance_penalty
        )

        metrics = {
            "lora_norm": norm_penalty.item(),
            "lora_variance": (-variance_penalty).item(),
        }

        return reg_loss, metrics

    def forward(
        self,
        delta_pred: torch.Tensor,
        delta_target: torch.Tensor,
        lora_weights: Optional[Dict[str, torch.Tensor]] = None,
        logits_pred: Optional[torch.Tensor] = None,
        logits_base: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute delta-only loss with regularization.

        Args:
            delta_pred: Predicted delta embedding
            delta_target: Target delta embedding
            lora_weights: Generated LoRA weights (for regularization)
            logits_pred: Logits from adapted model (for KL tether)
            logits_base: Logits from base model (for KL tether)

        Returns:
            total_loss: Combined loss
            metrics: Individual loss components
        """
        # Main delta loss
        loss_delta = self.compute_delta_loss(delta_pred, delta_target)

        metrics = {
            "loss_delta": loss_delta.item(),
        }

        total_loss = loss_delta

        # LoRA regularization
        if lora_weights is not None and self.lora_norm_weight > 0:
            reg_loss, reg_metrics = self.compute_lora_regularization(lora_weights)
            total_loss = total_loss + reg_loss
            metrics.update(reg_metrics)
            metrics["loss_reg"] = reg_loss.item()

        # KL tether on logits (keep behavior consistent beyond hidden states)
        if (logits_pred is not None and logits_base is not None
            and self.logit_kl_weight > 0):
            kl_loss = F.kl_div(
                F.log_softmax(logits_pred, dim=-1),
                F.softmax(logits_base, dim=-1),
                reduction="batchmean",
            )
            total_loss = total_loss + self.logit_kl_weight * kl_loss
            metrics["loss_kl"] = kl_loss.item()

        # Additional metrics
        with torch.no_grad():
            cos_sim = F.cosine_similarity(delta_pred, delta_target, dim=-1).mean()
            metrics["delta_cosine_sim"] = cos_sim.item()

        metrics["loss_total"] = total_loss.item()

        return total_loss, metrics


class DeltaOnlyTrainer:
    """
    Trainer for delta-only experiments.
    """

    def __init__(
        self,
        model,
        delta_compute,
        loss_fn: DeltaOnlyLoss,
        optimizer,
        device,
    ):
        self.model = model
        self.delta_compute = delta_compute
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Single training step."""
        self.model.train()

        condition_ids = batch["condition_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        delta_target = batch["delta"].to(self.device)

        # Generate LoRA weights
        tokens_pred = self.model.generate(
            condition={
                "input_ids": condition_ids,
                "attention_mask": attention_mask,
            }
        )

        # Detokenize to LoRA weights (differentiable)
        lora_weights = self.detokenize(tokens_pred)

        # Compute predicted delta
        delta_pred = self.delta_compute(lora_weights)

        # Compute loss
        loss, metrics = self.loss_fn(
            delta_pred=delta_pred,
            delta_target=delta_target,
            lora_weights=lora_weights,
        )

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return metrics
```

### Step 2: Create Delta Predictor for Retrieval (Variant B2)

Create `llgbm/models/delta_predictor.py`:

```python
"""Delta predictor model for retrieval-based approach."""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Optional


class DeltaPredictor(nn.Module):
    """
    Predict delta embedding directly from condition text.

    Architecture: Text Encoder -> MLP -> Delta Embedding

    Much smaller and faster than full LoRA generation.
    """

    def __init__(
        self,
        encoder_name: str = "bert-base-uncased",
        hidden_dim: int = 1536,  # Qwen2.5-1.5B hidden size
        mlp_hidden: int = 2048,
        dropout: float = 0.1,
        freeze_encoder: bool = True,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(encoder_name)
        encoder_dim = self.encoder.config.hidden_size

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(encoder_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
        )

        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)

    def encode_text(
        self,
        texts: List[str],
        max_length: int = 512,
    ) -> torch.Tensor:
        """Encode text to embeddings."""
        inputs = self.tokenizer(
            texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.encoder.device) for k, v in inputs.items()}

        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.encoder(**inputs)

        # Use [CLS] token or mean pooling
        embeddings = outputs.last_hidden_state[:, 0]  # [CLS]
        return embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        texts: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Predict delta embedding.

        Args:
            input_ids: Pre-tokenized input IDs
            attention_mask: Attention mask
            texts: Raw text strings (alternative to input_ids)

        Returns:
            Predicted delta embedding (B, hidden_dim)
        """
        if texts is not None:
            text_embeddings = self.encode_text(texts)
        else:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            text_embeddings = outputs.last_hidden_state[:, 0]

        delta_pred = self.projection(text_embeddings)
        return delta_pred


class DeltaPredictorTrainer:
    """Train the delta predictor."""

    def __init__(
        self,
        model: DeltaPredictor,
        optimizer,
        device,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def train_step(self, batch: dict) -> dict:
        """Single training step."""
        self.model.train()

        condition_ids = batch["condition_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        delta_target = batch["delta"].to(self.device)

        # Forward
        delta_pred = self.model(
            input_ids=condition_ids,
            attention_mask=attention_mask,
        )

        # Loss
        loss = nn.functional.mse_loss(delta_pred, delta_target)

        # Metrics
        with torch.no_grad():
            cos_sim = nn.functional.cosine_similarity(
                delta_pred, delta_target, dim=-1
            ).mean()

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "cosine_sim": cos_sim.item(),
        }
```

### Step 3: Create Adapter Retrieval System

Create `llgbm/retrieval.py`:

```python
"""Adapter retrieval and merging based on delta similarity."""
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from safetensors.torch import load_file, save_file
import faiss

from llgbm.delta import DeltaCache


class AdapterIndex:
    """
    Index of adapter delta embeddings for fast retrieval.

    Uses FAISS for efficient similarity search.
    """

    def __init__(
        self,
        delta_cache: DeltaCache,
        use_gpu: bool = True,
    ):
        self.delta_cache = delta_cache
        self.use_gpu = use_gpu

        # Load all deltas
        self.deltas = delta_cache.get_all_deltas()
        self.adapter_paths = list(self.deltas.keys())

        # Build index
        self._build_index()

    def _build_index(self):
        """Build FAISS index for delta embeddings."""
        embeddings = np.stack([
            self.deltas[path] for path in self.adapter_paths
        ]).astype(np.float32)

        # L2 normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Create index
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner product = cosine after normalization

        if self.use_gpu and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(),
                0,
                self.index,
            )

        self.index.add(embeddings)

    def search(
        self,
        query_delta: np.ndarray,
        k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Find k most similar adapters by delta similarity.

        Args:
            query_delta: Query delta embedding (hidden_dim,)
            k: Number of results

        Returns:
            List of (adapter_path, similarity_score) tuples
        """
        query = query_delta.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)

        scores, indices = self.index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # Valid index
                results.append((self.adapter_paths[idx], float(score)))

        return results


class AdapterMerger:
    """
    Merge multiple LoRA adapters using various strategies.
    """

    @staticmethod
    def load_adapter(path: str) -> Dict[str, torch.Tensor]:
        """Load adapter weights from path."""
        adapter_file = Path(path) / "adapter_model.safetensors"
        return load_file(str(adapter_file))

    @staticmethod
    def weighted_average(
        adapters: List[Dict[str, torch.Tensor]],
        weights: List[float],
    ) -> Dict[str, torch.Tensor]:
        """
        Simple weighted average of adapter weights.

        Args:
            adapters: List of adapter weight dicts
            weights: Weights for each adapter (will be normalized)

        Returns:
            Merged adapter weights
        """
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]

        merged = {}
        for key in adapters[0].keys():
            merged[key] = sum(
                w * adapter[key] for adapter, w in zip(adapters, weights)
            )

        return merged

    @staticmethod
    def ties_merge(
        adapters: List[Dict[str, torch.Tensor]],
        weights: Optional[List[float]] = None,
        density: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        TIES (TrIm, Elect Sign, Merge) adapter merging.

        1. Trim: Zero out small magnitude parameters
        2. Elect: Resolve sign conflicts by majority vote
        3. Merge: Average the remaining parameters

        Args:
            adapters: List of adapter weight dicts
            weights: Optional weights for each adapter
            density: Fraction of parameters to keep (0.5 = top 50%)

        Returns:
            Merged adapter weights
        """
        if weights is None:
            weights = [1.0 / len(adapters)] * len(adapters)
        else:
            total = sum(weights)
            weights = [w / total for w in weights]

        merged = {}

        for key in adapters[0].keys():
            # Stack all adapter weights for this parameter
            stacked = torch.stack([a[key] for a in adapters])  # (num_adapters, ...)

            # Step 1: Trim (zero out bottom (1-density) by magnitude)
            threshold = torch.quantile(
                stacked.abs().flatten(),
                1 - density,
            )
            trimmed = torch.where(
                stacked.abs() >= threshold,
                stacked,
                torch.zeros_like(stacked),
            )

            # Step 2: Elect sign (majority vote)
            signs = torch.sign(trimmed)
            sign_counts = signs.sum(dim=0)
            elected_signs = torch.sign(sign_counts)
            elected_signs[elected_signs == 0] = 1  # Default to positive

            # Step 3: Merge (average magnitudes with elected sign)
            magnitudes = trimmed.abs()
            weighted_mags = sum(
                w * mag for w, mag in zip(weights, magnitudes)
            )
            merged[key] = elected_signs * weighted_mags

        return merged

    @staticmethod
    def dare_merge(
        adapters: List[Dict[str, torch.Tensor]],
        weights: Optional[List[float]] = None,
        drop_rate: float = 0.9,
    ) -> Dict[str, torch.Tensor]:
        """
        DARE (Drop And REscale) adapter merging.

        Randomly drop parameters and rescale to preserve expected value.

        Args:
            adapters: List of adapter weight dicts
            weights: Optional weights for each adapter
            drop_rate: Fraction of parameters to drop

        Returns:
            Merged adapter weights
        """
        if weights is None:
            weights = [1.0 / len(adapters)] * len(adapters)

        merged = {}

        for key in adapters[0].keys():
            # Create random mask
            mask = torch.rand_like(adapters[0][key]) > drop_rate

            # Apply mask and rescale
            rescale = 1.0 / (1.0 - drop_rate)
            merged_param = torch.zeros_like(adapters[0][key])

            for adapter, w in zip(adapters, weights):
                dropped = adapter[key] * mask * rescale
                merged_param += w * dropped

            merged[key] = merged_param

        return merged


class RetrievalBasedGenerator:
    """
    Generate LoRA adapters by retrieving and merging similar adapters.

    Pipeline:
    1. Predict delta from condition
    2. Retrieve top-k similar adapters
    3. Merge adapters (weighted by similarity)
    """

    def __init__(
        self,
        delta_predictor,
        adapter_index: AdapterIndex,
        merge_strategy: str = "weighted_average",
        top_k: int = 3,
    ):
        self.delta_predictor = delta_predictor
        self.adapter_index = adapter_index
        self.merge_strategy = merge_strategy
        self.top_k = top_k

        self.merger = AdapterMerger()

    def generate(
        self,
        condition_text: str,
        output_path: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate LoRA adapter for given condition.

        Args:
            condition_text: Text condition
            output_path: Optional path to save merged adapter

        Returns:
            Merged LoRA adapter weights
        """
        # Predict delta
        self.delta_predictor.eval()
        with torch.no_grad():
            delta_pred = self.delta_predictor(texts=[condition_text])
            delta_pred = delta_pred[0].cpu().numpy()

        # Retrieve similar adapters
        results = self.adapter_index.search(delta_pred, k=self.top_k)

        # Load adapters
        adapters = []
        weights = []
        for path, score in results:
            adapters.append(self.merger.load_adapter(path))
            weights.append(score)

        # Merge
        if self.merge_strategy == "weighted_average":
            merged = self.merger.weighted_average(adapters, weights)
        elif self.merge_strategy == "ties":
            merged = self.merger.ties_merge(adapters, weights)
        elif self.merge_strategy == "dare":
            merged = self.merger.dare_merge(adapters, weights)
        else:
            raise ValueError(f"Unknown merge strategy: {self.merge_strategy}")

        # Optionally save
        if output_path:
            save_file(merged, output_path)

        return merged
```

### Step 4: Create Training Scripts

Create `scripts/train_delta_only.py`:

```python
"""Train with delta-only supervision."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from llgbm.training.delta_only import DeltaOnlyLoss, DeltaOnlyTrainer
from llgbm.dataset import Text2Qwen25LoRA_DeltaDataset
from llgbm.delta import DeltaCache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["b1", "b2"], default="b1")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="outputs/delta_only")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.variant == "b1":
        # Variant B1: Full generator with delta-only loss
        train_b1(args, device)
    else:
        # Variant B2: Delta predictor with retrieval
        train_b2(args, device)


def train_b1(args, device):
    """Train full generator with delta-only loss."""
    from workspace.dnd.model.decoderonly import HyperConvDecoderModel_SuperLarge
    from llgbm.functional_lora import DifferentiableDeltaCompute

    # Create model (same as Phase 4)
    model = create_generator_model().to(device)

    # Delta compute
    delta_compute = DifferentiableDeltaCompute(
        base_model_name="Qwen/Qwen2.5-1.5B",
        device=device,
    )

    # Loss with regularization
    loss_fn = DeltaOnlyLoss(
        delta_loss_type="mse",
        lora_norm_weight=0.01,
        lora_variance_weight=0.001,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Data
    dataloader = create_dataloader(args.batch_size)

    # Training
    trainer = DeltaOnlyTrainer(model, delta_compute, loss_fn, optimizer, device)

    for epoch in range(args.epochs):
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            metrics = trainer.train_step(batch)

        print(f"Epoch {epoch+1}: {metrics}")

    # Save
    torch.save(model.state_dict(), f"{args.output_dir}/model_b1.pt")


def train_b2(args, device):
    """Train delta predictor for retrieval."""
    from llgbm.models.delta_predictor import DeltaPredictor, DeltaPredictorTrainer

    # Create delta predictor
    model = DeltaPredictor(
        encoder_name="bert-base-uncased",
        hidden_dim=1536,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Data
    dataloader = create_dataloader(args.batch_size)

    # Trainer
    trainer = DeltaPredictorTrainer(model, optimizer, device)

    for epoch in range(args.epochs):
        epoch_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            metrics = trainer.train_step(batch)
            epoch_loss += metrics["loss"]

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, cos_sim={metrics['cosine_sim']:.4f}")

    # Save
    torch.save(model.state_dict(), f"{args.output_dir}/delta_predictor.pt")


if __name__ == "__main__":
    main()
```

## File Structure After Phase 5

```
llgbm/
├── llgbm/
│   ├── training/
│   │   └── delta_only.py      # Delta-only losses
│   ├── models/
│   │   └── delta_predictor.py # Small delta predictor
│   └── retrieval.py           # Adapter retrieval and merging
├── scripts/
│   └── train_delta_only.py    # Training script
└── outputs/
    └── delta_only/
        ├── model_b1.pt
        └── delta_predictor.pt
```

## Acceptance Criteria

- [ ] Variant B1 trains stably with regularization
- [ ] Variant B2 achieves reasonable delta prediction accuracy
- [ ] Generated/retrieved adapters beat base model on at least one task
- [ ] Retrieval baseline provides competitive "floor"
- [ ] No degenerate solutions (all-zero LoRA, collapsed diversity)

## Comparison Experiments

| Method | Delta MSE | Delta Cosine | Task Accuracy |
|--------|-----------|--------------|---------------|
| Baseline DnD (weights only) | - | baseline | baseline |
| Multi-task (Phase 4) | X.XX | X.XX | +Y% |
| Delta-only B1 | X.XX | X.XX | +/-Y% |
| Delta-only B2 (retrieval) | X.XX | X.XX | +/-Y% |

## Next Phase
Proceed to **Phase 6** for compositionality and behavioral algebra tests.
