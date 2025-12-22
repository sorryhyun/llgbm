# Phase 2 — Dataset Plumbing: Return Delta Labels Alongside Weight Tokens

## Goal
Modify the dataset to return delta embeddings alongside tokenized LoRA weights, enabling behavioral supervision during training.

## Prerequisites
- Phase 1 complete (delta cache populated)
- Working baseline dataset from Phase 0

## Data Flow

```
Before (DnD baseline):
  Dataset → (tokens_teacher, condition)

After (with delta supervision):
  Dataset → (tokens_teacher, condition, delta_teacher)
```

## Implementation Steps

### Step 1: Create Delta-Augmented Dataset Wrapper

Create `llgbm/dataset.py`:

```python
"""Dataset classes with delta embedding support."""
import sys
sys.path.insert(0, "Drag-and-Drop-LLMs")

import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from llgbm.delta import DeltaCache


class DeltaAugmentedDataset(Dataset):
    """
    Wrapper that adds delta embeddings to an existing DnD dataset.

    Wraps any dataset that returns (tokens, condition, checkpoint_path)
    and adds the corresponding delta_teacher from cache.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        delta_cache: DeltaCache,
        require_delta: bool = True,
    ):
        """
        Args:
            base_dataset: The underlying DnD dataset
            delta_cache: Cache containing pre-computed delta embeddings
            require_delta: If True, skip samples without cached deltas
        """
        self.base_dataset = base_dataset
        self.delta_cache = delta_cache
        self.require_delta = require_delta

        # Build index of valid samples (those with cached deltas)
        self.valid_indices = []
        self.delta_map = {}  # index -> delta array

        all_deltas = delta_cache.get_all_deltas()

        for idx in range(len(base_dataset)):
            # Get checkpoint path from base dataset
            sample = base_dataset[idx]
            if len(sample) >= 3:
                checkpoint_path = sample[2]  # (tokens, condition, path)
            else:
                checkpoint_path = getattr(base_dataset, 'get_checkpoint_path', lambda x: None)(idx)

            if checkpoint_path and checkpoint_path in all_deltas:
                self.valid_indices.append(idx)
                self.delta_map[idx] = all_deltas[checkpoint_path]
            elif not require_delta:
                self.valid_indices.append(idx)
                self.delta_map[idx] = None

        print(f"DeltaAugmentedDataset: {len(self.valid_indices)}/{len(base_dataset)} samples have deltas")

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            tokens: Tokenized LoRA weights (num_tokens, H, W)
            condition: Text condition tensor
            delta: Delta embedding (hidden_size,) or zeros if not available
        """
        real_idx = self.valid_indices[idx]
        sample = self.base_dataset[real_idx]

        tokens = sample[0]  # Tokenized LoRA weights
        condition = sample[1]  # Text condition

        # Get delta
        delta = self.delta_map.get(real_idx)
        if delta is not None:
            delta = torch.from_numpy(delta).float()
        else:
            # Return zeros if no delta (when require_delta=False)
            # Infer hidden size from first valid delta
            hidden_size = next(
                (d.shape[0] for d in self.delta_map.values() if d is not None),
                1536  # Default for Qwen2.5-1.5B
            )
            delta = torch.zeros(hidden_size)

        return tokens, condition, delta

    @staticmethod
    def collate_fn(batch: List[Tuple]) -> Dict[str, torch.Tensor]:
        """
        Collate function for DataLoader.

        Returns dict with:
            tokens: (B, num_tokens, H, W)
            condition: (B, seq_len) or dict for attention mask
            delta: (B, hidden_size)
        """
        tokens_list = []
        condition_list = []
        delta_list = []

        for tokens, condition, delta in batch:
            tokens_list.append(tokens)
            condition_list.append(condition)
            delta_list.append(delta)

        return {
            "tokens": torch.stack(tokens_list),
            "condition": torch.stack(condition_list) if isinstance(condition_list[0], torch.Tensor) else condition_list,
            "delta": torch.stack(delta_list),
        }


class Text2Qwen25LoRA_DeltaDataset(Dataset):
    """
    Full dataset implementation combining DnD's LoRA tokenization with delta supervision.

    Alternative to wrapping - directly implements the full pipeline.
    """

    def __init__(
        self,
        checkpoint_folder: str,
        lora_tokenizer,
        text_tokenizer,
        delta_cache: DeltaCache,
        max_text_length: int = 512,
        condition_type: str = "prompt",  # "prompt" or "prompt_answer"
    ):
        """
        Args:
            checkpoint_folder: Path to folder containing LoRA checkpoints
            lora_tokenizer: DnD tokenizer for LoRA weights
            text_tokenizer: HuggingFace tokenizer for text conditions
            delta_cache: Pre-computed delta embeddings
            max_text_length: Max tokens for text condition
            condition_type: Type of conditioning ("prompt" or "prompt_answer")
        """
        self.checkpoint_folder = Path(checkpoint_folder)
        self.lora_tokenizer = lora_tokenizer
        self.text_tokenizer = text_tokenizer
        self.delta_cache = delta_cache
        self.max_text_length = max_text_length
        self.condition_type = condition_type

        # Find all checkpoints
        self.checkpoint_paths = self._find_checkpoints()

        # Load delta mapping
        self.all_deltas = delta_cache.get_all_deltas()

        # Filter to only checkpoints with deltas
        self.valid_checkpoints = [
            p for p in self.checkpoint_paths
            if str(p) in self.all_deltas
        ]

        print(f"Found {len(self.valid_checkpoints)}/{len(self.checkpoint_paths)} checkpoints with deltas")

        # Load prompts/answers
        self.conditions = self._load_conditions()

    def _find_checkpoints(self) -> List[Path]:
        """Find all LoRA checkpoint directories."""
        checkpoints = []
        for path in self.checkpoint_folder.rglob("adapter_config.json"):
            checkpoints.append(path.parent)
        return sorted(checkpoints)

    def _load_conditions(self) -> Dict[str, Dict]:
        """Load text conditions for each checkpoint."""
        import json
        conditions = {}

        for ckpt_path in self.valid_checkpoints:
            # Try to find associated prompts file
            prompts_file = ckpt_path / "prompts.json"
            if prompts_file.exists():
                with open(prompts_file) as f:
                    data = json.load(f)
                conditions[str(ckpt_path)] = {
                    "prompts": data.get("prompts", data),
                    "answers": data.get("answers", []),
                }
            else:
                # Use checkpoint name as fallback condition
                conditions[str(ckpt_path)] = {
                    "prompts": [ckpt_path.name],
                    "answers": [],
                }

        return conditions

    def _load_checkpoint(self, path: Path) -> Dict[str, torch.Tensor]:
        """Load LoRA checkpoint from safetensors."""
        from safetensors.torch import load_file
        weights = load_file(path / "adapter_model.safetensors")
        # Sort keys for consistency
        return {k: weights[k] for k in sorted(weights.keys())}

    def __len__(self) -> int:
        return len(self.valid_checkpoints)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns dict with:
            tokens: Tokenized LoRA weights (num_tokens, H, W)
            scales: Normalization scales for reconstruction
            condition_ids: Token IDs for text condition
            attention_mask: Attention mask for condition
            delta: Delta embedding (hidden_size,)
        """
        ckpt_path = self.valid_checkpoints[idx]
        ckpt_str = str(ckpt_path)

        # Load and tokenize LoRA weights
        weights = self._load_checkpoint(ckpt_path)
        tokens, scales = self.lora_tokenizer.tokenize(weights)

        # Get text condition
        cond_data = self.conditions[ckpt_str]
        if self.condition_type == "prompt_answer" and cond_data["answers"]:
            # Concatenate prompt and answer
            text = cond_data["prompts"][0] + "\n" + cond_data["answers"][0]
        else:
            text = cond_data["prompts"][0]

        # Tokenize text
        text_encoded = self.text_tokenizer(
            text,
            max_length=self.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Get delta embedding
        delta = torch.from_numpy(self.all_deltas[ckpt_str]).float()

        return {
            "tokens": torch.from_numpy(tokens).float(),
            "scales": torch.from_numpy(scales).float(),
            "condition_ids": text_encoded["input_ids"].squeeze(0),
            "attention_mask": text_encoded["attention_mask"].squeeze(0),
            "delta": delta,
            "checkpoint_path": ckpt_str,
        }

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch into tensors."""
        return {
            "tokens": torch.stack([b["tokens"] for b in batch]),
            "scales": torch.stack([b["scales"] for b in batch]),
            "condition_ids": torch.stack([b["condition_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "delta": torch.stack([b["delta"] for b in batch]),
        }


def create_dataloader(
    checkpoint_folder: str,
    lora_tokenizer,
    text_tokenizer,
    delta_cache: DeltaCache,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    condition_type: str = "prompt",
    max_text_length: int = 512,
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader with delta supervision.

    Args:
        checkpoint_folder: Path to LoRA checkpoints
        lora_tokenizer: DnD tokenizer
        text_tokenizer: HuggingFace tokenizer
        delta_cache: Pre-computed deltas
        batch_size: Batch size
        num_workers: DataLoader workers
        shuffle: Whether to shuffle
        condition_type: "prompt" or "prompt_answer"
        max_text_length: Max text tokens

    Returns:
        DataLoader yielding batches with tokens, conditions, and deltas
    """
    dataset = Text2Qwen25LoRA_DeltaDataset(
        checkpoint_folder=checkpoint_folder,
        lora_tokenizer=lora_tokenizer,
        text_tokenizer=text_tokenizer,
        delta_cache=delta_cache,
        max_text_length=max_text_length,
        condition_type=condition_type,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
    )
```

### Step 2: Create Dataset Test Script

Create `scripts/test_delta_dataset.py`:

```python
"""Test the delta-augmented dataset."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "Drag-and-Drop-LLMs")

import torch
from transformers import AutoTokenizer

from workspace.dnd.tokenizer.register import Qwen2515LoRA_Tokenizer2D
from llgbm.delta import DeltaCache
from llgbm.dataset import Text2Qwen25LoRA_DeltaDataset, create_dataloader

def test_dataset():
    """Test dataset creation and iteration."""

    # Initialize components
    lora_tokenizer = Qwen2515LoRA_Tokenizer2D()
    text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    delta_cache = DeltaCache("deltas")

    # Create dataset
    dataset = Text2Qwen25LoRA_DeltaDataset(
        checkpoint_folder="data/teacher_checkpoints",
        lora_tokenizer=lora_tokenizer,
        text_tokenizer=text_tokenizer,
        delta_cache=delta_cache,
        max_text_length=512,
        condition_type="prompt",
    )

    print(f"Dataset size: {len(dataset)}")

    # Test single sample
    sample = dataset[0]
    print("\nSample shapes:")
    print(f"  tokens:         {sample['tokens'].shape}")
    print(f"  scales:         {sample['scales'].shape}")
    print(f"  condition_ids:  {sample['condition_ids'].shape}")
    print(f"  attention_mask: {sample['attention_mask'].shape}")
    print(f"  delta:          {sample['delta'].shape}")

    # Test dataloader
    dataloader = create_dataloader(
        checkpoint_folder="data/teacher_checkpoints",
        lora_tokenizer=lora_tokenizer,
        text_tokenizer=text_tokenizer,
        delta_cache=delta_cache,
        batch_size=4,
        num_workers=0,  # For testing
    )

    batch = next(iter(dataloader))
    print("\nBatch shapes:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}, dtype={value.dtype}")

    # Verify delta properties
    print("\nDelta statistics:")
    print(f"  Mean:  {batch['delta'].mean().item():.4f}")
    print(f"  Std:   {batch['delta'].std().item():.4f}")
    print(f"  Min:   {batch['delta'].min().item():.4f}")
    print(f"  Max:   {batch['delta'].max().item():.4f}")
    print(f"  Norm:  {batch['delta'].norm(dim=1).mean().item():.4f}")

    print("\n[PASS] Dataset test complete")

def test_dtype_consistency():
    """Verify dtypes are consistent for training."""
    lora_tokenizer = Qwen2515LoRA_Tokenizer2D()
    text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    delta_cache = DeltaCache("deltas")

    dataloader = create_dataloader(
        checkpoint_folder="data/teacher_checkpoints",
        lora_tokenizer=lora_tokenizer,
        text_tokenizer=text_tokenizer,
        delta_cache=delta_cache,
        batch_size=2,
        num_workers=0,
    )

    batch = next(iter(dataloader))

    expected_dtypes = {
        "tokens": torch.float32,
        "scales": torch.float32,
        "condition_ids": torch.long,
        "attention_mask": torch.long,
        "delta": torch.float32,
    }

    all_correct = True
    for key, expected in expected_dtypes.items():
        actual = batch[key].dtype
        status = "OK" if actual == expected else "FAIL"
        if actual != expected:
            all_correct = False
        print(f"  {key}: {actual} (expected {expected}) [{status}]")

    if all_correct:
        print("\n[PASS] Dtype consistency check")
    else:
        print("\n[FAIL] Dtype mismatch detected")

if __name__ == "__main__":
    test_dataset()
    print("\n" + "="*50 + "\n")
    test_dtype_consistency()
```

### Step 3: Integration with Existing DnD Dataset

Create `llgbm/dataset_wrapper.py`:

```python
"""Wrapper to add delta labels to existing DnD datasets."""
import sys
sys.path.insert(0, "Drag-and-Drop-LLMs")

import torch
from torch.utils.data import Dataset
from typing import Optional
from pathlib import Path

from workspace.dnd.dataset.register import (
    Text2Qwen25LoRA_FullCondDataset,
    Text2Qwen25LoRA_CondQ_ADataset,
)
from llgbm.delta import DeltaCache


class DnDDatasetWithDelta(Dataset):
    """
    Wraps existing DnD datasets to add delta embeddings.

    Compatible with:
    - Text2Qwen25LoRA_FullCondDataset
    - Text2Qwen25LoRA_CondQ_ADataset
    """

    def __init__(
        self,
        base_dataset: Dataset,
        delta_cache_dir: str = "deltas",
        fallback_delta_norm: float = 0.0,
    ):
        """
        Args:
            base_dataset: Underlying DnD dataset
            delta_cache_dir: Path to delta cache
            fallback_delta_norm: Norm for zero delta fallback
        """
        self.base_dataset = base_dataset
        self.delta_cache = DeltaCache(delta_cache_dir)
        self.all_deltas = self.delta_cache.get_all_deltas()
        self.fallback_delta_norm = fallback_delta_norm

        # Determine hidden size from cache
        if self.all_deltas:
            sample_delta = next(iter(self.all_deltas.values()))
            self.hidden_size = sample_delta.shape[0]
        else:
            self.hidden_size = 1536  # Qwen2.5-1.5B default

        # Build checkpoint path index
        self._build_path_index()

    def _build_path_index(self):
        """Map dataset indices to checkpoint paths."""
        self.idx_to_path = {}

        # Try different ways to get checkpoint paths
        if hasattr(self.base_dataset, 'checkpoint_paths'):
            for idx, path in enumerate(self.base_dataset.checkpoint_paths):
                self.idx_to_path[idx] = str(path)
        elif hasattr(self.base_dataset, 'samples'):
            for idx, sample in enumerate(self.base_dataset.samples):
                if 'path' in sample:
                    self.idx_to_path[idx] = str(sample['path'])

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        """
        Returns the base dataset output plus delta embedding.

        Base output: (tokens, condition, ...)
        New output: (tokens, condition, delta, ...)
        """
        base_output = self.base_dataset[idx]

        # Get delta
        ckpt_path = self.idx_to_path.get(idx)
        if ckpt_path and ckpt_path in self.all_deltas:
            delta = torch.from_numpy(self.all_deltas[ckpt_path]).float()
        else:
            delta = torch.zeros(self.hidden_size)

        # Insert delta after condition (position 2)
        if isinstance(base_output, tuple):
            return base_output[:2] + (delta,) + base_output[2:]
        elif isinstance(base_output, dict):
            base_output['delta'] = delta
            return base_output
        else:
            return base_output, delta

    def collate_fn_train(self, batch):
        """Collate with delta support."""
        # Separate deltas from base batch
        if isinstance(batch[0], tuple):
            tokens = torch.stack([b[0] for b in batch])
            conditions = torch.stack([b[1] for b in batch])
            deltas = torch.stack([b[2] for b in batch])
            return {
                'tokens': tokens,
                'condition': conditions,
                'delta': deltas,
            }
        elif isinstance(batch[0], dict):
            return {
                key: torch.stack([b[key] for b in batch])
                for key in batch[0].keys()
                if isinstance(batch[0][key], torch.Tensor)
            }


def wrap_dnd_dataset(
    dataset_class,
    delta_cache_dir: str = "deltas",
    **dataset_kwargs,
) -> DnDDatasetWithDelta:
    """
    Factory function to create delta-wrapped DnD datasets.

    Args:
        dataset_class: DnD dataset class to instantiate
        delta_cache_dir: Path to delta cache
        **dataset_kwargs: Arguments for the base dataset

    Returns:
        Wrapped dataset with delta support
    """
    base_dataset = dataset_class(**dataset_kwargs)
    return DnDDatasetWithDelta(base_dataset, delta_cache_dir)
```

## File Structure After Phase 2

```
llgbm/
├── llgbm/
│   ├── __init__.py
│   ├── probes.py
│   ├── delta.py
│   ├── dataset.py           # New: Delta-augmented datasets
│   └── dataset_wrapper.py   # New: Wrapper for DnD datasets
├── scripts/
│   ├── compute_teacher_deltas.py
│   ├── visualize_deltas.py
│   └── test_delta_dataset.py  # New: Dataset tests
└── deltas/
    └── ...
```

## Acceptance Criteria

- [ ] `Text2Qwen25LoRA_DeltaDataset` returns `(tokens, condition, delta)` tuples
- [ ] Delta shape is `(hidden_size,)` = `(1536,)` for Qwen2.5-1.5B
- [ ] Delta dtype is `float32`
- [ ] Collate function produces batches with shape `(B, hidden_size)`
- [ ] DataLoader iteration works without errors
- [ ] Samples without cached deltas are handled gracefully

## Usage

```python
from llgbm.dataset import create_dataloader
from llgbm.delta import DeltaCache

delta_cache = DeltaCache("deltas")
dataloader = create_dataloader(
    checkpoint_folder="data/teacher_checkpoints",
    lora_tokenizer=lora_tokenizer,
    text_tokenizer=text_tokenizer,
    delta_cache=delta_cache,
    batch_size=8,
)

for batch in dataloader:
    tokens = batch["tokens"]       # (B, num_tokens, H, W)
    condition = batch["condition_ids"]  # (B, seq_len)
    delta = batch["delta"]         # (B, hidden_size)

    # Training step...
```

## Next Phase
Proceed to **Phase 3** to implement differentiable delta computation for generated LoRAs.
