"""LoRA generator models for LLGBM.

This module provides generator architectures that take text conditions
and produce LoRA weights for target models.

Includes:
- LoRAGenerator: Generates LoRA weights from text embeddings
- DeltaPredictor: Predicts behavioral delta from embeddings (auxiliary supervision)
- LoRAGeneratorWithDeltaHead: Combined model for joint training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from llgbm.training import TrainingConfig


class LoRAGenerator(nn.Module):
    """
    LoRA generator that produces adapter weights from text conditions.

    Takes text condition and generates LoRA weights (A and B matrices) for all
    layers. Can use either:
    1. A pretrained text encoder (recommended) for semantic embeddings
    2. A learned embedding + transformer encoder (for backward compatibility)
    """

    def __init__(
        self,
        cfg: TrainingConfig,
        text_encoder: Optional[nn.Module] = None,
    ):
        """
        Args:
            cfg: TrainingConfig with model architecture settings
            text_encoder: Optional pretrained text encoder (e.g., PretrainedTextEncoder).
                         If provided, uses pretrained embeddings. Otherwise, learns from scratch.
        """
        super().__init__()
        self.cfg = cfg
        self.text_encoder = text_encoder
        self.use_pretrained = text_encoder is not None

        # Determine input dimension based on encoder type
        if self.use_pretrained:
            # Get embed_dim from pretrained encoder
            input_dim = getattr(text_encoder, "embed_dim", 384)
        else:
            # Use learned embeddings (backward compatibility)
            self.embed = nn.Embedding(50000, 256)
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=256, nhead=4, batch_first=True, dropout=0.1),
                num_layers=3
            )
            input_dim = 256

        # 7 projections per layer: q, k, v, o, gate, up, down
        self.num_projections = cfg.num_layers * 7

        # Per-projection embedding dimension
        self.proj_embed_dim = 512

        # Generate projection embeddings from condition
        self.proj_embeddings = nn.Linear(input_dim, self.num_projections * self.proj_embed_dim)

        self.lora_rank = cfg.lora_rank

        # MLPs to decode A and B matrices
        self.A_decoder = nn.Sequential(
            nn.Linear(self.proj_embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, cfg.lora_rank * 64),
        )
        self.B_decoder = nn.Sequential(
            nn.Linear(self.proj_embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, cfg.lora_rank * 64),
        )

        # Learnable scale factors per layer/projection type
        self.scales = nn.Parameter(torch.ones(self.num_projections, 2) * 0.01)

        # Cache dimension info
        self._build_dim_info()

    def _build_dim_info(self):
        """Build dimension info for each projection."""
        self.dim_info = []
        for layer in range(self.cfg.num_layers):
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
                if proj in ["k_proj", "v_proj"]:
                    out_d = self.cfg.num_kv_heads * (self.cfg.hidden_size // self.cfg.num_heads)
                elif proj in ["gate_proj", "up_proj"]:
                    out_d = self.cfg.intermediate_size
                elif proj == "down_proj":
                    out_d = self.cfg.hidden_size
                else:
                    out_d = self.cfg.hidden_size

                in_d = self.cfg.intermediate_size if proj == "down_proj" else self.cfg.hidden_size

                mod = "self_attn" if proj in ["q_proj", "k_proj", "v_proj", "o_proj"] else "mlp"
                prefix = f"model.layers.{layer}.{mod}.{proj}"

                self.dim_info.append({
                    "layer": layer,
                    "proj": proj,
                    "in_dim": in_d,
                    "out_dim": out_d,
                    "A_key": f"{prefix}.lora_A.weight",
                    "B_key": f"{prefix}.lora_B.weight",
                })

    def forward(
        self,
        condition_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Generate LoRA weights from text conditions.

        Args:
            condition_ids: Input token IDs (B, seq_len) or (B, N, seq_len) for batched prompts
                          OR pre-computed embeddings (B, embed_dim) if attention_mask is None
            attention_mask: Attention mask, same shape as condition_ids. If None and
                           condition_ids is 2D with matching embed_dim, treats as embeddings.

        Returns:
            List of dicts mapping weight keys to tensors, one per batch item
        """
        B = condition_ids.shape[0]

        # Check if we received pre-computed embeddings
        # (2D tensor with second dim matching embed_dim and no attention mask)
        is_precomputed = (
            len(condition_ids.shape) == 2
            and attention_mask is None
            and self.use_pretrained
            and condition_ids.shape[1] == getattr(self.text_encoder, "embed_dim", -1)
        )

        if is_precomputed:
            # Already have embeddings, use directly
            x = condition_ids
        elif self.use_pretrained:
            # Use pretrained text encoder
            # PretrainedTextEncoder handles (B, L) or (B, N, L) and returns (B, embed_dim)
            x = self.text_encoder(condition_ids, attention_mask)
        else:
            # Use learned embeddings (backward compatibility)
            # Flatten batched prompts if needed
            orig_B = B
            if len(condition_ids.shape) == 3:
                B, N, L = condition_ids.shape
                condition_ids = condition_ids.view(B * N, L)
                if attention_mask is not None:
                    attention_mask = attention_mask.view(B * N, L)

            x = self.embed(condition_ids)
            if attention_mask is not None:
                key_padding_mask = ~attention_mask.bool()
            else:
                key_padding_mask = None

            x = self.encoder(x, src_key_padding_mask=key_padding_mask)

            # Pool to single vector
            if attention_mask is not None:
                x = (x * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True).clamp(min=1)
            else:
                x = x.mean(1)

            # If we had batched prompts, pool across N dimension
            if len(x.shape) == 2 and x.shape[0] != orig_B:
                x = x.view(orig_B, -1, x.shape[-1]).mean(dim=1)  # (B, embed_dim)
            B = orig_B

        # Generate per-projection embeddings: (B, num_proj, proj_embed_dim)
        proj_embeds = self.proj_embeddings(x).view(B, self.num_projections, self.proj_embed_dim)

        # Decode to weights
        batch_weights = []
        for b in range(B):
            weights = {}
            for idx, info in enumerate(self.dim_info):
                embed = proj_embeds[b, idx]

                # Decode A and B base patterns
                A_base = self.A_decoder(embed).view(self.lora_rank, -1)
                B_base = self.B_decoder(embed).view(-1, self.lora_rank)

                # Expand to full dimensions via periodic extension
                in_d, out_d = info["in_dim"], info["out_dim"]

                A_full = A_base[:, :in_d % 64 or 64].repeat(1, (in_d // 64) + 1)[:, :in_d]
                B_full = B_base[:out_d % 64 or 64, :].repeat((out_d // 64) + 1, 1)[:out_d, :]

                # Apply learned scales
                scale_a, scale_b = self.scales[idx]
                A = A_full * scale_a
                B = B_full * scale_b

                weights[info["A_key"]] = A
                weights[info["B_key"]] = B

            batch_weights.append(weights)

        return batch_weights


class DeltaPredictor(nn.Module):
    """
    Predicts behavioral delta directly from embeddings.

    Takes N embeddings (one per prompt) and predicts the delta activation
    that a LoRA adapter would produce. Can predict per-embedding deltas
    and aggregate, or use attention to weight embeddings.

    This provides fast, differentiable supervision for LoRA training.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        hidden_size: int = 896,  # Target model's hidden size (delta dimension)
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        aggregation: str = "attention",  # "attention", "mean", or "weighted"
    ):
        """
        Args:
            embed_dim: Input embedding dimension
            hidden_size: Output delta dimension (matches target model)
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
            aggregation: How to aggregate N embeddings ("attention", "mean", "weighted")
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.aggregation = aggregation

        # Per-embedding delta prediction
        self.embedding_proj = nn.Linear(embed_dim, hidden_size)

        # Self-attention over embeddings
        if aggregation == "attention":
            self.self_attn = nn.MultiheadAttention(
                hidden_size, num_heads, dropout=dropout, batch_first=True
            )
            self.attn_norm = nn.LayerNorm(hidden_size)

        # Learnable aggregation weights
        if aggregation == "weighted":
            self.weight_proj = nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )

        # Delta refinement MLP
        self.delta_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        return_per_embedding: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict delta from embeddings.

        Args:
            embeddings: (B, N, embed_dim) - N embeddings per sample
                       or (B, embed_dim) - single embedding per sample
            return_per_embedding: If True, also return per-embedding deltas

        Returns:
            delta_pred: (B, hidden_size) - predicted delta
            delta_per_emb: (B, N, hidden_size) if return_per_embedding, else None
        """
        # Handle single embedding case
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(1)  # (B, 1, embed_dim)

        B, N, _ = embeddings.shape

        # Project each embedding to hidden_size
        h = self.embedding_proj(embeddings)  # (B, N, hidden_size)

        # Per-embedding delta predictions
        delta_per_emb = self.delta_mlp(h)  # (B, N, hidden_size)

        # Aggregate
        if self.aggregation == "attention" and N > 1:
            # Self-attention to let embeddings interact
            attn_out, _ = self.self_attn(h, h, h)
            h = self.attn_norm(h + attn_out)
            # Mean pool after attention
            delta_pred = h.mean(dim=1)
        elif self.aggregation == "weighted" and N > 1:
            # Learned weighting
            weights = self.weight_proj(h).squeeze(-1)  # (B, N)
            weights = F.softmax(weights, dim=-1)
            delta_pred = (delta_per_emb * weights.unsqueeze(-1)).sum(dim=1)
        else:
            # Simple mean
            delta_pred = delta_per_emb.mean(dim=1)

        # Final refinement
        delta_pred = self.delta_mlp(delta_pred)

        if return_per_embedding:
            return delta_pred, delta_per_emb
        return delta_pred, None


class LoRAGeneratorWithDeltaHead(nn.Module):
    """
    Combined LoRA Generator with Delta Prediction head.

    Architecture:
        embeddings → shared encoder → ┬→ LoRA weights (generator head)
                                      └→ predicted delta (delta head)

    Training modes:
    1. Joint: Train both heads with consistency loss
    2. Delta-guided: Use delta predictor to supervise LoRA generator
    3. Alternating: Train delta predictor, then use it to train generator
    """

    def __init__(
        self,
        cfg: TrainingConfig,
        text_encoder: Optional[nn.Module] = None,
        delta_aggregation: str = "attention",
    ):
        """
        Args:
            cfg: Training configuration
            text_encoder: Optional pretrained text encoder
            delta_aggregation: Aggregation method for delta predictor
        """
        super().__init__()
        self.cfg = cfg
        self.text_encoder = text_encoder
        self.use_pretrained = text_encoder is not None

        # Determine embedding dimension
        if self.use_pretrained:
            embed_dim = getattr(text_encoder, "embed_dim", 384)
        else:
            embed_dim = 256
            self.embed = nn.Embedding(50000, embed_dim)
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True, dropout=0.1),
                num_layers=3
            )

        self.embed_dim = embed_dim

        # Shared projection (processes embeddings before splitting)
        self.shared_proj = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
        )

        # LoRA generation head
        self.num_projections = cfg.num_layers * 7
        self.proj_embed_dim = 512
        self.lora_rank = cfg.lora_rank

        self.lora_head = nn.Linear(512, self.num_projections * self.proj_embed_dim)

        self.A_decoder = nn.Sequential(
            nn.Linear(self.proj_embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, cfg.lora_rank * 64),
        )
        self.B_decoder = nn.Sequential(
            nn.Linear(self.proj_embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, cfg.lora_rank * 64),
        )
        self.scales = nn.Parameter(torch.ones(self.num_projections, 2) * 0.01)

        # Delta prediction head
        self.delta_head = DeltaPredictor(
            embed_dim=512,  # Takes shared projection output
            hidden_size=cfg.hidden_size,
            aggregation=delta_aggregation,
        )

        # Build dimension info for LoRA
        self._build_dim_info()

    def _build_dim_info(self):
        """Build dimension info for each projection."""
        self.dim_info = []
        for layer in range(self.cfg.num_layers):
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
                if proj in ["k_proj", "v_proj"]:
                    out_d = self.cfg.num_kv_heads * (self.cfg.hidden_size // self.cfg.num_heads)
                elif proj in ["gate_proj", "up_proj"]:
                    out_d = self.cfg.intermediate_size
                elif proj == "down_proj":
                    out_d = self.cfg.hidden_size
                else:
                    out_d = self.cfg.hidden_size

                in_d = self.cfg.intermediate_size if proj == "down_proj" else self.cfg.hidden_size

                mod = "self_attn" if proj in ["q_proj", "k_proj", "v_proj", "o_proj"] else "mlp"
                prefix = f"model.layers.{layer}.{mod}.{proj}"

                self.dim_info.append({
                    "layer": layer,
                    "proj": proj,
                    "in_dim": in_d,
                    "out_dim": out_d,
                    "A_key": f"{prefix}.lora_A.weight",
                    "B_key": f"{prefix}.lora_B.weight",
                })

    def _encode_embeddings(
        self,
        condition_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Encode inputs to embeddings (handles batched prompts)."""
        B = condition_ids.shape[0]

        # Check if pre-computed embeddings
        is_precomputed = (
            len(condition_ids.shape) == 2
            and attention_mask is None
            and self.use_pretrained
            and condition_ids.shape[1] == self.embed_dim
        )

        if is_precomputed:
            return condition_ids.unsqueeze(1)  # (B, 1, embed_dim)

        # Handle batched prompts (B, N, L) vs single (B, L)
        if len(condition_ids.shape) == 3:
            B, N, L = condition_ids.shape
            has_batch = True
        else:
            N = 1
            has_batch = False

        if self.use_pretrained:
            # Text encoder handles (B, N, L) internally
            if has_batch:
                # Process each prompt separately to get N embeddings
                embeddings = []
                for i in range(N):
                    emb = self.text_encoder(
                        condition_ids[:, i, :],
                        attention_mask[:, i, :] if attention_mask is not None else None
                    )
                    embeddings.append(emb)
                return torch.stack(embeddings, dim=1)  # (B, N, embed_dim)
            else:
                return self.text_encoder(condition_ids, attention_mask).unsqueeze(1)
        else:
            # Learned embeddings
            if has_batch:
                condition_ids = condition_ids.view(B * N, -1)
                if attention_mask is not None:
                    attention_mask = attention_mask.view(B * N, -1)

            x = self.embed(condition_ids)
            key_padding_mask = ~attention_mask.bool() if attention_mask is not None else None
            x = self.encoder(x, src_key_padding_mask=key_padding_mask)

            if attention_mask is not None:
                x = (x * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True).clamp(min=1)
            else:
                x = x.mean(1)

            if has_batch:
                x = x.view(B, N, -1)
            else:
                x = x.unsqueeze(1)

            return x  # (B, N, embed_dim)

    def forward(
        self,
        condition_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_delta: bool = True,
        return_lora: bool = True,
    ) -> Dict[str, any]:
        """
        Forward pass with both LoRA generation and delta prediction.

        Args:
            condition_ids: (B, L) or (B, N, L) input tokens, or (B, embed_dim) embeddings
            attention_mask: Attention mask matching condition_ids shape
            return_delta: Whether to compute predicted delta
            return_lora: Whether to generate LoRA weights

        Returns:
            Dict with:
                - "lora_weights": List[Dict] of LoRA weights (if return_lora)
                - "delta_pred": (B, hidden_size) predicted delta (if return_delta)
                - "delta_per_emb": (B, N, hidden_size) per-embedding deltas (if return_delta)
                - "embeddings": (B, N, embed_dim) encoded embeddings
        """
        B = condition_ids.shape[0]

        # Encode to embeddings (B, N, embed_dim)
        embeddings = self._encode_embeddings(condition_ids, attention_mask)

        # Shared projection
        shared = self.shared_proj(embeddings)  # (B, N, 512)

        results = {"embeddings": embeddings}

        # Delta prediction (uses all N embeddings)
        if return_delta:
            delta_pred, delta_per_emb = self.delta_head(shared, return_per_embedding=True)
            results["delta_pred"] = delta_pred
            results["delta_per_emb"] = delta_per_emb

        # LoRA generation (pool embeddings first)
        if return_lora:
            shared_pooled = shared.mean(dim=1)  # (B, 512)
            proj_embeds = self.lora_head(shared_pooled).view(B, self.num_projections, self.proj_embed_dim)

            batch_weights = []
            for b in range(B):
                weights = {}
                for idx, info in enumerate(self.dim_info):
                    embed = proj_embeds[b, idx]

                    A_base = self.A_decoder(embed).view(self.lora_rank, -1)
                    B_base = self.B_decoder(embed).view(-1, self.lora_rank)

                    in_d, out_d = info["in_dim"], info["out_dim"]
                    A_full = A_base[:, :in_d % 64 or 64].repeat(1, (in_d // 64) + 1)[:, :in_d]
                    B_full = B_base[:out_d % 64 or 64, :].repeat((out_d // 64) + 1, 1)[:out_d, :]

                    scale_a, scale_b = self.scales[idx]
                    weights[info["A_key"]] = A_full * scale_a
                    weights[info["B_key"]] = B_full * scale_b

                batch_weights.append(weights)

            results["lora_weights"] = batch_weights

        return results

    def generate_lora(
        self,
        condition_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """Convenience method to just get LoRA weights (for inference)."""
        return self.forward(condition_ids, attention_mask, return_delta=False)["lora_weights"]

    def predict_delta(
        self,
        condition_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convenience method to just get predicted delta."""
        return self.forward(condition_ids, attention_mask, return_lora=False)["delta_pred"]


def create_generator(
    cfg: TrainingConfig,
    seed: int = 42,
    device: torch.device = None,
    text_encoder: Optional[nn.Module] = None,
) -> LoRAGenerator:
    """
    Create and initialize a LoRA generator.

    Args:
        cfg: Training configuration
        seed: Random seed for reproducibility
        device: Device to place model on
        text_encoder: Optional pretrained text encoder for condition embeddings

    Returns:
        Initialized LoRAGenerator
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    gen = LoRAGenerator(cfg, text_encoder=text_encoder).to(device)
    num_params = sum(p.numel() for p in gen.parameters() if p.requires_grad)
    print(f"  Generator params: {num_params:,}")
    if text_encoder is not None:
        print(f"  Using pretrained text encoder: {getattr(text_encoder, 'model_name', 'unknown')}")
    return gen


def create_generator_with_delta_head(
    cfg: TrainingConfig,
    seed: int = 42,
    device: torch.device = None,
    text_encoder: Optional[nn.Module] = None,
    delta_aggregation: str = "attention",
) -> LoRAGeneratorWithDeltaHead:
    """
    Create a LoRA generator with delta prediction head.

    This model can:
    1. Generate LoRA weights from text conditions
    2. Predict behavioral delta directly (for fast supervision)
    3. Be trained with consistency loss between predicted and computed delta

    Args:
        cfg: Training configuration
        seed: Random seed
        device: Device to place model on
        text_encoder: Optional pretrained text encoder
        delta_aggregation: How delta head aggregates N embeddings

    Returns:
        LoRAGeneratorWithDeltaHead instance
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    gen = LoRAGeneratorWithDeltaHead(
        cfg,
        text_encoder=text_encoder,
        delta_aggregation=delta_aggregation,
    ).to(device)

    # Count parameters
    total = sum(p.numel() for p in gen.parameters() if p.requires_grad)
    delta_params = sum(p.numel() for p in gen.delta_head.parameters() if p.requires_grad)
    lora_params = total - delta_params

    print("Generator with Delta Head:")
    print(f"  LoRA generation params: {lora_params:,}")
    print(f"  Delta prediction params: {delta_params:,}")
    print(f"  Total trainable: {total:,}")
    print(f"  Delta aggregation: {delta_aggregation}")

    if text_encoder is not None:
        print(f"  Text encoder: {getattr(text_encoder, 'model_name', 'unknown')}")

    return gen
