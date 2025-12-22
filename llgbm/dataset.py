"""Dataset classes with delta embedding support.

This module provides dataset classes that combine DnD's LoRA tokenization
with delta activation supervision for behavioral matching.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import numpy as np
from torch.utils.data import Dataset

from llgbm.delta import DeltaCache


class DeltaAugmentedDataset(Dataset):
    """
    Wrapper that adds delta embeddings to an existing DnD dataset.

    Wraps any dataset that returns (tokens, condition) or dict-based outputs
    and adds the corresponding delta_teacher from cache.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        delta_cache: DeltaCache,
        checkpoint_paths: Optional[List[str]] = None,
        require_delta: bool = True,
        default_hidden_size: int = 1536,
    ):
        """
        Args:
            base_dataset: The underlying DnD dataset
            delta_cache: Cache containing pre-computed delta embeddings
            checkpoint_paths: List of checkpoint paths matching base_dataset indices
            require_delta: If True, skip samples without cached deltas
            default_hidden_size: Hidden size for zero fallback (Qwen2.5-1.5B: 1536)
        """
        self.base_dataset = base_dataset
        self.delta_cache = delta_cache
        self.require_delta = require_delta
        self.default_hidden_size = default_hidden_size

        # Get all cached deltas
        all_deltas = delta_cache.get_all_deltas()

        # Determine hidden size from cache
        if all_deltas:
            sample_delta = next(iter(all_deltas.values()))
            self.hidden_size = sample_delta.shape[0]
        else:
            self.hidden_size = default_hidden_size

        # Build checkpoint path index
        if checkpoint_paths is not None:
            self.checkpoint_paths = checkpoint_paths
        elif hasattr(base_dataset, 'checkpoint_paths'):
            self.checkpoint_paths = [str(p) for p in base_dataset.checkpoint_paths]
        elif hasattr(base_dataset, 'samples'):
            self.checkpoint_paths = [
                str(s.get('path', s.get('checkpoint_path', '')))
                for s in base_dataset.samples
            ]
        else:
            self.checkpoint_paths = [None] * len(base_dataset)

        # Build index of valid samples (those with cached deltas)
        self.valid_indices = []
        self.delta_map = {}

        for idx in range(len(base_dataset)):
            ckpt_path = self.checkpoint_paths[idx] if idx < len(self.checkpoint_paths) else None

            if ckpt_path and ckpt_path in all_deltas:
                self.valid_indices.append(idx)
                self.delta_map[idx] = all_deltas[ckpt_path]
            elif not require_delta:
                self.valid_indices.append(idx)
                self.delta_map[idx] = None

        print(f"DeltaAugmentedDataset: {len(self.valid_indices)}/{len(base_dataset)} samples have deltas")

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns dict with:
            tokens: Tokenized LoRA weights
            condition: Text condition tensor
            delta: Delta embedding (hidden_size,)
        """
        real_idx = self.valid_indices[idx]
        sample = self.base_dataset[real_idx]

        # Handle different sample formats
        if isinstance(sample, dict):
            output = {k: v for k, v in sample.items()}
        elif isinstance(sample, (tuple, list)):
            output = {
                'tokens': sample[0],
                'condition': sample[1],
            }
            # Include any additional items
            for i, item in enumerate(sample[2:], start=2):
                output[f'item_{i}'] = item
        else:
            output = {'data': sample}

        # Get delta
        delta = self.delta_map.get(real_idx)
        if delta is not None:
            output['delta'] = torch.from_numpy(delta).float()
        else:
            output['delta'] = torch.zeros(self.hidden_size)

        return output

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate function for DataLoader.

        Returns dict with all tensor keys stacked.
        """
        result = {}
        for key in batch[0].keys():
            values = [b[key] for b in batch]
            if isinstance(values[0], torch.Tensor):
                result[key] = torch.stack(values)
            else:
                result[key] = values
        return result


class Text2Qwen25LoRA_DeltaDataset(Dataset):
    """
    Full dataset implementation combining DnD's LoRA tokenization with delta supervision.

    This directly implements the complete pipeline for loading LoRA checkpoints,
    tokenizing them, and pairing with delta embeddings.
    """

    def __init__(
        self,
        checkpoint_folder: str,
        lora_tokenizer,
        text_tokenizer,
        delta_cache: DeltaCache,
        max_text_length: int = 512,
        condition_type: str = "prompt",
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

        # Determine hidden size
        if self.all_deltas:
            sample_delta = next(iter(self.all_deltas.values()))
            self.hidden_size = sample_delta.shape[0]
        else:
            self.hidden_size = 1536

        # Filter to only checkpoints with deltas
        self.valid_checkpoints = [
            p for p in self.checkpoint_paths
            if str(p) in self.all_deltas
        ]

        print(f"Found {len(self.valid_checkpoints)}/{len(self.checkpoint_paths)} checkpoints with deltas")

        # Load prompts/conditions
        self.conditions = self._load_conditions()

    def _find_checkpoints(self) -> List[Path]:
        """Find all LoRA checkpoint directories."""
        checkpoints = []
        for path in self.checkpoint_folder.rglob("adapter_config.json"):
            checkpoints.append(path.parent)
        return sorted(checkpoints)

    def _load_conditions(self) -> Dict[str, Dict]:
        """Load text conditions for each checkpoint."""
        conditions = {}

        for ckpt_path in self.valid_checkpoints:
            # Try to find associated prompts file
            prompts_file = ckpt_path / "prompts.json"
            if prompts_file.exists():
                with open(prompts_file) as f:
                    data = json.load(f)
                conditions[str(ckpt_path)] = {
                    "prompts": data.get("prompts", data) if isinstance(data, dict) else data,
                    "answers": data.get("answers", []) if isinstance(data, dict) else [],
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
            checkpoint_path: Path to the checkpoint
        """
        ckpt_path = self.valid_checkpoints[idx]
        ckpt_str = str(ckpt_path)

        # Load and tokenize LoRA weights
        weights = self._load_checkpoint(ckpt_path)
        tokens, scales = self.lora_tokenizer.tokenize(weights)

        # Get text condition
        cond_data = self.conditions[ckpt_str]
        prompts = cond_data["prompts"]
        prompts_list = prompts if isinstance(prompts, list) else [prompts]

        if self.condition_type == "prompt_answer" and cond_data["answers"]:
            text = prompts_list[0] + "\n" + cond_data["answers"][0]
        else:
            text = prompts_list[0] if prompts_list else ckpt_path.name

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
            "tokens": tokens.float(),
            "scales": scales.float() if scales is not None else None,
            "condition_ids": text_encoded["input_ids"].squeeze(0),
            "attention_mask": text_encoded["attention_mask"].squeeze(0),
            "delta": delta,
            "checkpoint_path": ckpt_str,
        }

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch into tensors."""
        result = {}
        for key in batch[0].keys():
            values = [b[key] for b in batch]
            if isinstance(values[0], torch.Tensor):
                result[key] = torch.stack(values)
            else:
                result[key] = values
        return result


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
