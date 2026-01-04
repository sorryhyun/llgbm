"""Pretrained text encoder for prompt embeddings.

This module provides a frozen pretrained text encoder (sentence-transformers)
for generating condition embeddings, matching the DnD paper's approach.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Optional
from pathlib import Path
import numpy as np


class PretrainedTextEncoder(nn.Module):
    """Frozen pretrained text encoder for condition embeddings.

    Uses sentence-transformers models (e.g., all-MiniLM-L6-v2) to generate
    semantic embeddings for text conditions. By default, the encoder is frozen
    to provide stable, pretrained representations.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        freeze: bool = True,
        pooling: str = "mean",  # "mean", "cls", or "last"
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model_name: HuggingFace model name for the text encoder
            freeze: If True, freeze encoder parameters (recommended)
            pooling: Pooling strategy - "mean" (default), "cls", or "last"
            device: Device to place the model on
        """
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pooling = pooling
        self.embed_dim = self.model.config.hidden_size  # 384 for MiniLM-L6
        self.model_name = model_name

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        if device is not None:
            self.model = self.model.to(device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute embeddings from token IDs.

        Args:
            input_ids: (B, seq_len) or (B, N, seq_len) for batched prompts
            attention_mask: same shape as input_ids

        Returns:
            embeddings: (B, embed_dim) or (B, embed_dim) after pooling across N
        """
        orig_shape = input_ids.shape

        # Handle batched prompts (B, N, L) -> flatten to (B*N, L)
        if len(orig_shape) == 3:
            B, N, L = orig_shape
            input_ids = input_ids.view(B * N, L)
            attention_mask = attention_mask.view(B * N, L)
        else:
            B, L = orig_shape
            N = 1

        # Get hidden states
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        hidden = outputs.last_hidden_state  # (B*N, L, hidden_size)

        # Pool according to strategy
        if self.pooling == "cls":
            pooled = hidden[:, 0, :]  # CLS token
        elif self.pooling == "last":
            # Last non-padding token
            seq_lens = attention_mask.sum(dim=1).long() - 1
            batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
            pooled = hidden[batch_idx, seq_lens, :]
        else:  # mean pooling (default)
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_hidden = (hidden * mask_expanded).sum(dim=1)
            pooled = sum_hidden / mask_expanded.sum(dim=1).clamp(min=1e-9)

        # If we had batched prompts, pool across N dimension
        if len(orig_shape) == 3:
            pooled = pooled.view(B, N, -1)  # (B, N, embed_dim)
            pooled = pooled.mean(dim=1)  # (B, embed_dim) - mean over prompts

        return pooled

    def encode_texts(
        self,
        texts: List[str],
        max_length: int = 256,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Convenience method to encode raw text strings.

        Args:
            texts: List of text strings to encode
            max_length: Maximum sequence length
            device: Device for output tensor

        Returns:
            embeddings: (len(texts), embed_dim)
        """
        if device is None:
            device = next(self.model.parameters()).device

        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        return self(input_ids, attention_mask)

    def encode_and_pool_batch(
        self,
        texts_batch: List[List[str]],
        max_length: int = 256,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Encode multiple prompts per sample and pool them.

        Args:
            texts_batch: List of lists, each inner list is prompts for one sample
            max_length: Maximum sequence length
            device: Device for output tensor

        Returns:
            embeddings: (B, embed_dim) - one embedding per sample
        """
        if device is None:
            device = next(self.model.parameters()).device

        embeddings = []
        for texts in texts_batch:
            emb = self.encode_texts(texts, max_length, device)  # (N, embed_dim)
            pooled = emb.mean(dim=0)  # (embed_dim,)
            embeddings.append(pooled)

        return torch.stack(embeddings)  # (B, embed_dim)


class TrainableTextEncoder(nn.Module):
    """Text encoder with trainable projection layer.

    Keeps the pretrained encoder frozen but adds a learnable projection/adapter
    on top, allowing gradients to flow back to the generator while preserving
    pretrained semantic knowledge.

    Architecture:
        frozen_encoder -> projection -> LayerNorm -> output
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        output_dim: Optional[int] = None,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        pooling: str = "mean",
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model_name: HuggingFace model name for the base encoder
            output_dim: Output embedding dimension. If None, uses input dim.
            hidden_dim: Hidden dimension for projection MLP
            num_layers: Number of projection layers (1 = linear, 2+ = MLP)
            dropout: Dropout rate in projection
            pooling: Pooling strategy - "mean", "cls", or "last"
            device: Device to place the model on
        """
        super().__init__()

        # Load frozen base encoder
        self.base_model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pooling = pooling
        self.model_name = model_name

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Get input dimension from base model
        input_dim = self.base_model.config.hidden_size  # 384 for MiniLM-L6
        self.base_embed_dim = input_dim

        # Output dimension
        if output_dim is None:
            output_dim = input_dim
        self.embed_dim = output_dim

        # Build trainable projection
        if num_layers == 1:
            self.projection = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim),
            )
        else:
            layers = []
            dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:  # Not last layer
                    layers.append(nn.GELU())
                    layers.append(nn.Dropout(dropout))
            layers.append(nn.LayerNorm(output_dim))
            self.projection = nn.Sequential(*layers)

        if device is not None:
            self.base_model = self.base_model.to(device)
            self.projection = self.projection.to(device)

    def _pool_hidden(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pool hidden states according to strategy."""
        if self.pooling == "cls":
            return hidden[:, 0, :]
        elif self.pooling == "last":
            seq_lens = attention_mask.sum(dim=1).long() - 1
            batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
            return hidden[batch_idx, seq_lens, :]
        else:  # mean pooling
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_hidden = (hidden * mask_expanded).sum(dim=1)
            return sum_hidden / mask_expanded.sum(dim=1).clamp(min=1e-9)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute embeddings with trainable projection.

        Args:
            input_ids: (B, seq_len) or (B, N, seq_len) for batched prompts
            attention_mask: same shape as input_ids

        Returns:
            embeddings: (B, embed_dim) with gradient connection
        """
        orig_shape = input_ids.shape

        # Handle batched prompts (B, N, L) -> flatten to (B*N, L)
        if len(orig_shape) == 3:
            B, N, L = orig_shape
            input_ids = input_ids.view(B * N, L)
            attention_mask = attention_mask.view(B * N, L)
        else:
            B, L = orig_shape
            N = 1

        # Get frozen base embeddings (no grad through base model)
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            hidden = outputs.last_hidden_state  # (B*N, L, hidden_size)

        # Pool hidden states
        pooled = self._pool_hidden(hidden, attention_mask)  # (B*N, base_embed_dim)

        # Trainable projection (gradients flow here!)
        projected = self.projection(pooled)  # (B*N, embed_dim)

        # If we had batched prompts, pool across N dimension
        if len(orig_shape) == 3:
            projected = projected.view(B, N, -1)  # (B, N, embed_dim)
            projected = projected.mean(dim=1)  # (B, embed_dim)

        return projected

    def encode_texts(
        self,
        texts: List[str],
        max_length: int = 256,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Encode raw text strings."""
        if device is None:
            device = next(self.base_model.parameters()).device

        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        return self(input_ids, attention_mask)


class EmbeddingCache:
    """Cache for precomputed text embeddings.

    Stores per-adapter embeddings to disk to speed up training.
    """

    def __init__(self, cache_dir: str = "embeddings"):
        """
        Args:
            cache_dir: Directory to store cached embeddings
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, adapter_name: str) -> Path:
        """Get path for cached embedding."""
        # Sanitize adapter name for filesystem
        safe_name = adapter_name.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe_name}.npy"

    def has_cached(self, adapter_name: str) -> bool:
        """Check if embedding is cached."""
        return self.get_cache_path(adapter_name).exists()

    def save(self, adapter_name: str, embedding: torch.Tensor):
        """Save embedding to cache."""
        path = self.get_cache_path(adapter_name)
        np.save(path, embedding.cpu().numpy())

    def load(self, adapter_name: str) -> Optional[torch.Tensor]:
        """Load embedding from cache."""
        path = self.get_cache_path(adapter_name)
        if path.exists():
            return torch.from_numpy(np.load(path))
        return None

    def cache_adapter_embeddings(
        self,
        encoder: PretrainedTextEncoder,
        adapter_prompts: dict,
        max_length: int = 256,
        device: Optional[torch.device] = None,
        num_prompts: int = 8,
    ):
        """
        Precompute and cache embeddings for all adapters.

        Args:
            encoder: PretrainedTextEncoder instance
            adapter_prompts: Dict mapping adapter_name -> list of prompts
            max_length: Max sequence length
            device: Device for computation
            num_prompts: Number of prompts to sample per adapter
        """
        import random

        for adapter_name, prompts in adapter_prompts.items():
            if self.has_cached(adapter_name):
                continue

            # Sample prompts
            if len(prompts) >= num_prompts:
                selected = random.sample(prompts, num_prompts)
            else:
                # Repeat if not enough prompts
                selected = (prompts * ((num_prompts // len(prompts)) + 1))[:num_prompts]

            # Encode and pool
            embedding = encoder.encode_texts(selected, max_length, device)
            pooled = embedding.mean(dim=0)  # (embed_dim,)

            self.save(adapter_name, pooled)

        print(f"Cached embeddings for {len(adapter_prompts)} adapters in {self.cache_dir}")


def create_text_encoder(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    freeze: bool = True,
    device: Optional[torch.device] = None,
) -> PretrainedTextEncoder:
    """
    Create a pretrained text encoder.

    Args:
        model_name: HuggingFace model name
        freeze: Whether to freeze parameters
        device: Device to place model on

    Returns:
        PretrainedTextEncoder instance
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = PretrainedTextEncoder(
        model_name=model_name,
        freeze=freeze,
        device=device,
    )

    num_params = sum(p.numel() for p in encoder.parameters())
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"Text encoder: {model_name}")
    print(f"  Total params: {num_params:,}")
    print(f"  Trainable: {trainable:,}")
    print(f"  Embed dim: {encoder.embed_dim}")

    return encoder


def create_trainable_text_encoder(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    output_dim: Optional[int] = None,
    hidden_dim: int = 512,
    num_layers: int = 2,
    dropout: float = 0.1,
    device: Optional[torch.device] = None,
) -> TrainableTextEncoder:
    """
    Create a text encoder with trainable projection layer.

    The base encoder is frozen, but a projection layer on top is trainable,
    allowing gradients to flow back during training.

    Args:
        model_name: HuggingFace model name for base encoder
        output_dim: Output embedding dimension (default: same as input)
        hidden_dim: Hidden dimension for projection MLP
        num_layers: Number of projection layers (1=linear, 2+=MLP)
        dropout: Dropout rate in projection
        device: Device to place model on

    Returns:
        TrainableTextEncoder instance
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = TrainableTextEncoder(
        model_name=model_name,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        device=device,
    )

    # Count parameters
    base_params = sum(p.numel() for p in encoder.base_model.parameters())
    proj_params = sum(p.numel() for p in encoder.projection.parameters())
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)

    print(f"Trainable text encoder: {model_name}")
    print(f"  Base encoder params: {base_params:,} (frozen)")
    print(f"  Projection params: {proj_params:,} (trainable)")
    print(f"  Total trainable: {trainable:,}")
    print(f"  Input dim: {encoder.base_embed_dim} -> Output dim: {encoder.embed_dim}")

    return encoder
