"""LoRA generator models for LLGBM.

This module provides generator architectures that take text conditions
and produce LoRA weights for target models.
"""

import torch
import torch.nn as nn
from typing import Dict, List

from llgbm.training import TrainingConfig


class LoRAGenerator(nn.Module):
    """
    LoRA generator that produces adapter weights from text conditions.

    Takes text condition and generates LoRA weights (A and B matrices) for all
    layers. Uses a transformer encoder to process condition, then projects to
    LoRA weights via a hypernetwork-style decoder.
    """

    def __init__(self, cfg: TrainingConfig):
        """
        Args:
            cfg: TrainingConfig with model architecture settings
        """
        super().__init__()
        self.cfg = cfg

        # Text encoder
        self.embed = nn.Embedding(50000, 256)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=4, batch_first=True, dropout=0.1),
            num_layers=3
        )

        # 7 projections per layer: q, k, v, o, gate, up, down
        self.num_projections = cfg.num_layers * 7

        # Per-projection embedding dimension
        self.proj_embed_dim = 512

        # Generate projection embeddings from condition
        self.proj_embeddings = nn.Linear(256, self.num_projections * self.proj_embed_dim)

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
            condition_ids: Input token IDs (B, seq_len)
            attention_mask: Attention mask (B, seq_len)

        Returns:
            List of dicts mapping weight keys to tensors, one per batch item
        """
        B = condition_ids.shape[0]

        # Encode condition
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


def create_generator(
    cfg: TrainingConfig,
    seed: int = 42,
    device: torch.device = None,
) -> LoRAGenerator:
    """
    Create and initialize a LoRA generator.

    Args:
        cfg: Training configuration
        seed: Random seed for reproducibility
        device: Device to place model on

    Returns:
        Initialized LoRAGenerator
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    gen = LoRAGenerator(cfg).to(device)
    num_params = sum(p.numel() for p in gen.parameters() if p.requires_grad)
    print(f"  Generator params: {num_params:,}")
    return gen
