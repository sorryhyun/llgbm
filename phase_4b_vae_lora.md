# Phase 4b — Discrete Codebook for LoRA: Quantizing the Adapter Manifold

## Core Idea

Instead of generating continuous LoRA weights from text, **learn a finite
dictionary of LoRA building blocks** and reduce adapter generation to
**picking and composing discrete codes**.

An adapter is a sequence of 24 layer-level decisions. Each decision picks from a
learned codebook of "what this layer should do." Generation becomes
classification, not regression. The codebook itself *is* the manifold.

```
Current (Phase 5):   text → MLP → 168 continuous (A,B) matrices
Proposed (Phase 4b): text → encoder → 24 code picks → codebook lookup → LoRA weights
```

This gives us:
- **Efficiency** — inference is codebook lookup, not dense MLP decoding
- **Interpretability** — each code is a reusable, inspectable "layer behavior"
- **Sharp generation** — discrete codes don't blur across modes
- **Manifold discovery** — the codebook reveals how many distinct adapter
  patterns actually exist

Prior art: **FVAE-LoRA** (factorized VAE for LoRA), **ICM-LoRA** (IJCAI 2025,
Conditional VAE for LoRA generation). Both validate the latent-variable approach;
we push further by going fully discrete.

---

## Why Discrete Over Continuous?

The conversation that motivated this proposal identified several problems with
continuous VAE over adapter weights:

1. **Non-identifiability**: `DW = B @ A` has infinite `(A,B)` factorizations.
   A Gaussian VAE sees the same function as many points → mushy latent.
2. **Multimodal data**: DnD adapters span distinct task families (commonsense,
   math, coding). Gaussian prior averages modes → "blurry adapters."
3. **Row-wise linear correlation is weak**: Flattened weight matrices don't have
   the structure that makes standard VAEs work well.

All three problems point toward **discrete codes**:

| Problem | Continuous VAE | Discrete VQ |
|---------|---------------|-------------|
| Non-identifiability | Learns mushy latent | Codes snap to prototypes; gauge freedom absorbed |
| Multimodal | Gaussian averages modes | Each code IS a mode |
| Weak linear structure | MLP must learn nonlinear manifold | Codebook stores actual exemplars |

The gauge-fixing problem (Trick #1 from the original conversation) still matters
— we should quantize **DW-space** (`B @ A * scaling`), not raw `(A, B)`. But
discrete codes are more forgiving because the codebook entries themselves become
the canonical representatives.

---

## Data

From [`Jerrylz/DnD-checkpoints-and-logs`](https://huggingface.co/datasets/Jerrylz/DnD-checkpoints-and-logs):

| Family | Checkpoints | Tasks |
|--------|-------------|-------|
| **Qwen2.5-0.5B LoRA** | 28 | ARC-c/e, BoolQ, OBQA, PIQA, WinoGrande, HellaSwag + ablations |
| **Qwen2.5-1.5B LoRA** | 3 | Coding, Math |
| **Qwen2.5-7B LoRA** | 2 | Coding, Math |
| **Qwen3-VL LoRA** | 1 | Multimodal |

**Start with:** The 28 `qwen0.5lora__*` checkpoints. Same architecture (24
layers, rank 16, 7 targets), diverse tasks. Hold out 3 for validation.

Each checkpoint yields **24 layers x 7 targets = 168 (A, B) pairs**. That's
28 x 168 = **4,704 layer-target weight pairs** to learn from — enough for a
codebook of ~64-256 entries.

---

## Architecture

### Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   STAGE 1: Learn the Codebook               │
│                   (auto-encoding, no text)                   │
│                                                             │
│  Teacher adapter (24 layers)                                │
│       │                                                     │
│  Per-layer: DW_i = B_i @ A_i * scaling    (gauge-fix)      │
│       │                                                     │
│  [Layer Encoder] → e_i ∈ R^d                                │
│       │                                                     │
│  [VQ] → code_i ∈ {1..K},  z_i = Codebook[code_i]          │
│       │                                                     │
│  [Layer Decoder] → DW_i_reconstructed                       │
│       │                                                     │
│  Loss = L_recon + L_commit + L_codebook                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   STAGE 2: Learn the Prior                   │
│                   (conditional generation)                   │
│                                                             │
│  Text condition                                             │
│       │                                                     │
│  [TextEncoder] → (B, 384)                                   │
│       │                                                     │
│  [Code Predictor] → logits (B, 24, K)                       │
│       │                                                     │
│  argmax or sample → code sequence (B, 24)                   │
│       │                                                     │
│  [Frozen codebook lookup + decoder] → LoRA weights          │
└─────────────────────────────────────────────────────────────┘
```

Two-stage training separates "what are the building blocks?" from "when to use
which block?" This is the standard VQ-VAE recipe (van den Oord et al., 2017)
adapted for weight space.

### Stage 1: VQ Auto-Encoder for Layer Patterns

#### Representation: What Gets Quantized

Each layer `i` in an adapter has 7 target projections (q/k/v/o/gate/up/down).
We gauge-fix each to DW-space, then compress:

```python
# For layer i, target t:
DW_it = B_it @ A_it * scaling   # (out_dim, in_dim) — unique functional update

# Since rank(DW) <= 16, compress via truncated SVD:
U, S, Vh = svd(DW_it)
token_it = concat(S[:r], flatten(U[:,:r]), flatten(Vh[:r,:]))  # compact token
```

But DW dimensions vary across targets:

| Target | Shape | Rank-16 SVD token size |
|--------|-------|----------------------|
| q_proj | (896, 896) | 16 + 896\*16 + 896\*16 = 28,688 |
| k_proj | (128, 896) | 16 + 128\*16 + 896\*16 = 16,400 |
| gate_proj | (4864, 896) | 16 + 4864\*16 + 896\*16 = 92,176 |

These are too large to quantize directly. Instead, use **a learned projection
to a fixed-size embedding** before quantization:

```python
class LayerEncoder(nn.Module):
    """Encode one layer's 7 DW matrices into a single embedding."""

    def __init__(self, lora_rank=16, base_dim=64, embed_dim=256):
        super().__init__()
        # Per-target projectors: project each DW's SVD factors to fixed size
        # Input: rank singular values + rank*base_dim for U/V factors
        self.target_projectors = nn.ModuleDict()
        for target in TARGETS:
            in_size = lora_rank + 2 * lora_rank * base_dim  # S + U_base + V_base
            self.target_projectors[target] = nn.Sequential(
                nn.Linear(in_size, 256),
                nn.GELU(),
                nn.Linear(256, embed_dim // 7),  # Each target contributes a slice
            )

        # Cross-target fusion
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, layer_dw_tokens: Dict[str, Tensor]) -> Tensor:
        """
        Args:
            layer_dw_tokens: {target_name: compact_svd_token} for one layer
        Returns:
            (embed_dim,) embedding for this layer
        """
        parts = []
        for target in TARGETS:
            parts.append(self.target_projectors[target](layer_dw_tokens[target]))
        fused = torch.cat(parts, dim=-1)
        return self.fusion(fused)
```

The key insight: we project each target's rank-16 SVD factors through a small MLP
to get a fixed-size slice, concatenate all 7 targets, then fuse. This compresses
an entire layer's LoRA update into a single `embed_dim`-sized vector (e.g., 256d).

The "base_dim=64" comes from the existing generator's periodic extension trick:
the current `A_decoder` outputs `rank * 64` floats. We reuse this: truncate/pad
the SVD factors to `rank x 64` before encoding. This aligns with the decoder's
existing reconstruction mechanism.

#### Vector Quantization

```python
class VectorQuantizer(nn.Module):
    """
    VQ layer with EMA codebook updates and dead code reset.

    Uses exponential moving average (EMA) for codebook updates instead of
    gradient descent — more stable and doesn't require separate codebook optimizer.
    """

    def __init__(
        self,
        num_codes: int = 256,
        embed_dim: int = 256,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        dead_code_threshold: int = 2,  # Reset codes used < N times per epoch
    ):
        super().__init__()
        self.num_codes = num_codes
        self.embed_dim = embed_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.dead_code_threshold = dead_code_threshold

        # Codebook
        self.codebook = nn.Embedding(num_codes, embed_dim)
        nn.init.uniform_(self.codebook.weight, -1/num_codes, 1/num_codes)

        # EMA tracking
        self.register_buffer('ema_count', torch.zeros(num_codes))
        self.register_buffer('ema_weight', self.codebook.weight.clone())
        self.register_buffer('usage_count', torch.zeros(num_codes, dtype=torch.long))

    def forward(self, z_e: Tensor) -> Tuple[Tensor, Tensor, LongTensor]:
        """
        Args:
            z_e: encoder output (..., embed_dim)
        Returns:
            z_q: quantized (..., embed_dim), with straight-through gradient
            vq_loss: commitment + codebook loss
            indices: code indices (...,)
        """
        flat = z_e.reshape(-1, self.embed_dim)

        # Find nearest codes
        distances = torch.cdist(flat, self.codebook.weight)
        indices = distances.argmin(dim=-1)
        z_q = self.codebook(indices).view_as(z_e)

        # EMA update (training only)
        if self.training:
            self._ema_update(flat, indices)

        # Losses
        commitment = F.mse_loss(z_e, z_q.detach())
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment

        # Straight-through estimator
        z_q = z_e + (z_q - z_e).detach()

        return z_q, vq_loss, indices.view(z_e.shape[:-1])

    def _ema_update(self, flat, indices):
        """EMA codebook update."""
        with torch.no_grad():
            onehot = F.one_hot(indices, self.num_codes).float()
            self.ema_count = self.ema_decay * self.ema_count + (1-self.ema_decay) * onehot.sum(0)
            self.ema_weight = self.ema_decay * self.ema_weight + (1-self.ema_decay) * (onehot.T @ flat)

            n = self.ema_count.unsqueeze(1)
            self.codebook.weight.data = self.ema_weight / n.clamp(min=1e-5)

            # Track usage
            self.usage_count += onehot.sum(0).long()

    def reset_dead_codes(self, encoder_outputs: Tensor):
        """Replace dead codes with random encoder outputs."""
        dead = self.usage_count < self.dead_code_threshold
        if dead.any():
            n_dead = dead.sum().item()
            # Sample random encoder outputs as replacement
            perm = torch.randperm(encoder_outputs.size(0))[:n_dead]
            self.codebook.weight.data[dead] = encoder_outputs[perm].detach()
            self.usage_count[dead] = self.dead_code_threshold
```

#### Layer Decoder

Mirrors the existing generator's `A_decoder`/`B_decoder` + periodic extension:

```python
class LayerDecoder(nn.Module):
    """Decode quantized embedding back to 7 (A, B) pairs for one layer."""

    def __init__(self, embed_dim=256, lora_rank=16, base_dim=64):
        super().__init__()
        self.target_decoders = nn.ModuleDict()
        for target in TARGETS:
            self.target_decoders[target] = nn.ModuleDict({
                'A': nn.Sequential(
                    nn.Linear(embed_dim, 256), nn.GELU(),
                    nn.Linear(256, lora_rank * base_dim),
                ),
                'B': nn.Sequential(
                    nn.Linear(embed_dim, 256), nn.GELU(),
                    nn.Linear(256, lora_rank * base_dim),
                ),
            })
        self.scales = nn.Parameter(torch.ones(7, 2) * 0.01)

    def forward(self, z_q: Tensor, dim_info: List[Dict]) -> Dict[str, Tensor]:
        """Decode one layer's quantized code into LoRA weight dict."""
        weights = {}
        for t_idx, (target, decoder) in enumerate(self.target_decoders.items()):
            info = dim_info[t_idx]
            A_base = decoder['A'](z_q).view(self.lora_rank, -1)
            B_base = decoder['B'](z_q).view(-1, self.lora_rank)

            # Periodic extension (same as existing generator)
            in_d, out_d = info["in_dim"], info["out_dim"]
            A = A_base[:, :in_d % 64 or 64].repeat(1, (in_d//64)+1)[:, :in_d]
            B = B_base[:out_d % 64 or 64, :].repeat((out_d//64)+1, 1)[:out_d, :]

            weights[info["A_key"]] = A * self.scales[t_idx, 0]
            weights[info["B_key"]] = B * self.scales[t_idx, 1]
        return weights
```

This reuses the exact periodic extension mechanism from `LoRAGenerator`, so
the decoder output format is directly compatible with `FunctionalLoRA`.

#### Residual VQ (RVQ) for Progressive Detail

One codebook level may not capture fine-grained adapter differences. Residual VQ
adds levels that encode the reconstruction error from previous levels:

```
Level 1: z_q1 = VQ_1(z_e)           → captures coarse adapter pattern
Level 2: z_q2 = VQ_2(z_e - z_q1)    → captures residual detail
Level 3: z_q3 = VQ_3(z_e - z_q1 - z_q2)  → fine corrections

Final:   z_q = z_q1 + z_q2 + z_q3
```

This is the architecture behind SoundStream/Encodec for audio. For LoRA:

- **Level 1** (K=64 codes): "Is this an attention-heavy or MLP-heavy layer update?"
- **Level 2** (K=64 codes): "What's the specific direction of the update?"
- **Level 3** (K=64 codes): "Fine magnitude/sign corrections"

At inference, you can trade quality for speed by using fewer levels. Level 1
alone might capture 80% of the variance.

```python
class ResidualVQ(nn.Module):
    """Multi-level residual vector quantization."""

    def __init__(self, num_levels=3, num_codes=64, embed_dim=256):
        super().__init__()
        self.levels = nn.ModuleList([
            VectorQuantizer(num_codes, embed_dim)
            for _ in range(num_levels)
        ])

    def forward(self, z_e):
        z_q_total = torch.zeros_like(z_e)
        total_vq_loss = 0.0
        all_indices = []
        residual = z_e

        for vq in self.levels:
            z_q, vq_loss, indices = vq(residual)
            z_q_total = z_q_total + z_q
            total_vq_loss = total_vq_loss + vq_loss
            all_indices.append(indices)
            residual = z_e - z_q_total.detach()  # Next level encodes the residual

        return z_q_total, total_vq_loss, all_indices

    def decode_at_level(self, all_indices, max_level):
        """Partial decode using only first N levels (quality/speed tradeoff)."""
        z_q = torch.zeros(...)
        for level_idx in range(max_level):
            z_q = z_q + self.levels[level_idx].codebook(all_indices[level_idx])
        return z_q
```

#### Full Auto-Encoder

```python
class LoRACodebook(nn.Module):
    """
    VQ auto-encoder for LoRA adapters.

    Encodes each layer's 7-target LoRA update into a discrete code,
    decodes back to weight matrices. The codebook learns prototypical
    "layer behaviors" across all training adapters.
    """

    def __init__(
        self,
        num_codes: int = 256,
        embed_dim: int = 256,
        lora_rank: int = 16,
        base_dim: int = 64,
        num_layers: int = 24,
        num_rq_levels: int = 3,      # Residual VQ levels
        use_residual_vq: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.encoder = LayerEncoder(lora_rank, base_dim, embed_dim)
        self.decoder = LayerDecoder(embed_dim, lora_rank, base_dim)

        if use_residual_vq:
            self.quantizer = ResidualVQ(num_rq_levels, num_codes, embed_dim)
        else:
            self.quantizer = VectorQuantizer(num_codes, embed_dim)

        # Optional: cross-layer transformer to capture dependencies BEFORE quantizing
        self.cross_layer_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=4, batch_first=True, dropout=0.1
            ),
            num_layers=2,
        )

    def encode(self, adapter_dw_tokens: List[Dict[str, Tensor]]) -> Tensor:
        """
        Encode a full adapter (24 layers) into pre-quantization embeddings.

        Args:
            adapter_dw_tokens: List of 24 dicts, each mapping target names
                              to compact SVD tokens for that layer.
        Returns:
            (24, embed_dim) pre-quantization embeddings
        """
        layer_embeds = []
        for layer_tokens in adapter_dw_tokens:
            e = self.encoder(layer_tokens)
            layer_embeds.append(e)
        layer_embeds = torch.stack(layer_embeds)  # (24, embed_dim)

        # Cross-layer attention: let layers see each other before quantizing
        layer_embeds = self.cross_layer_attn(layer_embeds.unsqueeze(0)).squeeze(0)

        return layer_embeds

    def quantize(self, layer_embeds: Tensor):
        """Quantize layer embeddings to discrete codes."""
        return self.quantizer(layer_embeds)

    def decode(self, z_q: Tensor, dim_info_per_layer) -> Dict[str, Tensor]:
        """Decode quantized layer codes back to full adapter weights."""
        all_weights = {}
        for layer_idx in range(self.num_layers):
            layer_weights = self.decoder(z_q[layer_idx], dim_info_per_layer[layer_idx])
            all_weights.update(layer_weights)
        return all_weights

    def forward(self, adapter_dw_tokens, dim_info_per_layer):
        layer_embeds = self.encode(adapter_dw_tokens)
        z_q, vq_loss, indices = self.quantize(layer_embeds)
        weights_recon = self.decode(z_q, dim_info_per_layer)
        return weights_recon, vq_loss, indices, layer_embeds
```

### Stage 2: Code Predictor (Conditional Prior)

Once the codebook is trained, generation becomes **predicting a sequence of
24 code indices** from a text condition:

```python
class CodePredictor(nn.Module):
    """
    Predict code sequence from text condition.

    This is a classifier, not a regressor. For each of the 24 layers,
    predict which of K codebook entries to use.
    """

    def __init__(
        self,
        text_embed_dim: int = 384,
        num_layers: int = 24,
        num_codes: int = 256,
        hidden_dim: int = 512,
        num_rq_levels: int = 3,
    ):
        super().__init__()

        # Shared text processing
        self.proj = nn.Sequential(
            nn.Linear(text_embed_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # Layer position embeddings
        self.layer_pos = nn.Embedding(num_layers, hidden_dim)

        # Cross-attention: layer queries attend to text condition
        self.cross_attn = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim, nhead=4, batch_first=True, dropout=0.1,
            ),
            num_layers=2,
        )

        # Per-level code classifiers (for RVQ)
        self.code_heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_codes)
            for _ in range(num_rq_levels)
        ])

    def forward(self, text_embedding: Tensor) -> List[Tensor]:
        """
        Args:
            text_embedding: (B, text_embed_dim) from frozen text encoder
        Returns:
            List of (B, 24, K) logits, one per RVQ level
        """
        B = text_embedding.shape[0]

        # Process text condition
        memory = self.proj(text_embedding).unsqueeze(1)  # (B, 1, hidden)

        # Layer queries
        layer_ids = torch.arange(self.num_layers, device=text_embedding.device)
        queries = self.layer_pos(layer_ids).unsqueeze(0).expand(B, -1, -1)

        # Cross-attention: layers attend to text
        h = self.cross_attn(queries, memory)  # (B, 24, hidden)

        # Classify each layer at each RVQ level
        all_logits = [head(h) for head in self.code_heads]
        return all_logits  # List of (B, 24, K)

    def predict_codes(self, text_embedding: Tensor, temperature=0.0) -> List[LongTensor]:
        """Generate code sequences (inference)."""
        all_logits = self.forward(text_embedding)
        codes = []
        for logits in all_logits:
            if temperature == 0:
                codes.append(logits.argmax(dim=-1))
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                codes.append(torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(logits.shape[:2]))
        return codes
```

Key insight: **this is 24 classification problems, not weight regression**. A
24-layer adapter with K=256 codes per level and 3 RVQ levels requires predicting
24 x 3 = 72 categorical distributions. Compare to the current generator which
outputs 168 x 2 continuous matrices.

### Integration with Existing Pipeline

The `CodePredictor` replaces `LoRAGenerator`'s continuous `lora_head`. Everything
downstream stays the same:

```
[CodePredictor] → code indices → [Frozen LoRACodebook.decode()] → weight dict
     ↓
Same as current:
     ↓
[FunctionalLoRA.apply_lora_weights()] → delta_computed
     ↓
[DeltaGuidedLoss or behavioral eval]
```

The delta supervision from Phase 5 works unchanged — we just swap how
weights are produced.

---

## Loss Functions

### Stage 1: Codebook Learning

```
L_stage1 = L_recon + L_vq

L_recon = sum over layers, targets:
    MSE(DW_reconstructed, DW_teacher)

L_vq = sum over RVQ levels:
    L_codebook + beta_commit * L_commitment
```

Where `L_recon` compares in DW-space (gauge-fixed). This avoids the `(A, B)`
non-identifiability entirely.

Optionally add **behavioral reconstruction loss** — compute the delta from the
reconstructed adapter and compare to the teacher's delta:

```
L_recon_behavioral = cosine_distance(
    delta(decode(quantize(encode(adapter)))),
    delta(adapter)
)
```

This is expensive (requires base model forward pass) but ensures the codebook
preserves *function*, not just weight magnitude.

```python
class CodebookLoss(nn.Module):
    """Loss for Stage 1 codebook training."""

    def __init__(
        self,
        lambda_recon: float = 1.0,
        lambda_vq: float = 1.0,
        lambda_behavioral: float = 0.0,    # Enable for behavioral recon
        recon_space: str = "delta_w",       # "delta_w" or "ab"
    ):
        ...

    def forward(self, weights_recon, weights_teacher, vq_loss,
                delta_recon=None, delta_teacher=None):
        losses = {}

        # Reconstruction in DW-space
        losses["recon"] = self._recon_loss(weights_recon, weights_teacher)
        losses["vq"] = vq_loss

        total = self.lambda_recon * losses["recon"] + self.lambda_vq * losses["vq"]

        if delta_recon is not None and self.lambda_behavioral > 0:
            losses["behavioral"] = 1 - F.cosine_similarity(
                delta_recon, delta_teacher, dim=-1
            ).mean()
            total += self.lambda_behavioral * losses["behavioral"]

        losses["loss"] = total
        return losses
```

### Stage 2: Code Prediction

```
L_stage2 = sum over RVQ levels:
    CrossEntropy(predicted_logits, teacher_codes)
  + lambda_delta * L_delta(delta_from_predicted, delta_teacher)
```

The cross-entropy loss is the primary signal. Delta supervision (optional) adds
a behavioral gradient that helps when multiple code sequences produce similar
adapters.

```python
class CodePredictionLoss(nn.Module):
    """Loss for Stage 2 code prediction."""

    def __init__(
        self,
        lambda_ce: float = 1.0,
        lambda_delta: float = 0.5,
        label_smoothing: float = 0.1,
    ):
        ...

    def forward(self, predicted_logits, teacher_codes, delta_pred=None,
                delta_teacher=None):
        losses = {}
        total_ce = 0
        for level, (logits, codes) in enumerate(zip(predicted_logits, teacher_codes)):
            # logits: (B, 24, K), codes: (B, 24)
            ce = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                codes.view(-1),
                label_smoothing=self.label_smoothing,
            )
            total_ce += ce
            losses[f"ce_level_{level}"] = ce

        losses["ce_total"] = total_ce

        total = self.lambda_ce * total_ce
        if delta_pred is not None and self.lambda_delta > 0:
            losses["delta"] = 1 - F.cosine_similarity(
                delta_pred, delta_teacher, dim=-1
            ).mean()
            total += self.lambda_delta * losses["delta"]

        losses["loss"] = total
        return losses
```

---

## What the Codebook Reveals (Manifold Discovery)

After Stage 1 training, the codebook is a **complete catalog of how layers
behave**. Analysis we can do immediately:

### 1. Code Usage Histogram

For each of the 28 adapters, record its 24-layer code sequence. Plot:

- **How many unique codes are actually used?** If 256 codes exist but only
  40 are used, the adapter manifold is ~40-dimensional, not 256.
- **Usage frequency** — power-law? uniform? A few dominant codes?

### 2. Code-to-Task Mapping

Color each code by which tasks use it:

```
Code 17: [ARC-e: 80%, BoolQ: 60%, PIQA: 30%]  → "general reasoning pattern"
Code 42: [ARC-e: 0%, BoolQ: 0%, HellaSwag: 90%] → "sentence completion specialist"
Code 3:  [all tasks: 95%]                        → "universal early-layer identity"
```

If codes cluster by task, the codebook has learned a task taxonomy.

### 3. Layer Position × Code Heatmap

```
         Code 0  Code 1  Code 2  ...  Code K
Layer 0  [████]  [    ]  [██  ]  ...
Layer 1  [████]  [    ]  [██  ]  ...
Layer 12 [    ]  [████]  [    ]  ...   ← mid-layer divergence point
Layer 23 [    ]  [██  ]  [████]  ...
```

Expect: early layers converge to few codes (universal patterns), mid/late
layers diverge (task-specific behaviors).

### 4. Code Arithmetic

If the codebook is well-structured:

```
codes(ARC_adapter) ⊕ codes(math_adapter) → mixed-task adapter?
```

Replace specific layer codes from one adapter with codes from another:

```
Take ARC adapter's code sequence
Swap layers 12-18 codes with BoolQ adapter's
→ Does the resulting adapter do both tasks?
```

This is the discrete version of Phase 6's compositionality experiments,
but much more precise — we're swapping specific layer behaviors, not
interpolating in continuous weight space.

---

## Sizing the Codebook

How many codes do we need? Consider:

- **28 adapters x 24 layers = 672 layer-level data points**
- With 7 targets per layer, there's internal structure, but the VQ
  sees each layer as one vector
- Rule of thumb: `K ≈ sqrt(N_datapoints)` for basic VQ → K ≈ 26
- With RVQ (3 levels x 64 codes): effective codebook = 64^3 = 262K
  combinations, but only 192 stored vectors

| Config | Stored vectors | Effective combinations | Bits per layer |
|--------|---------------|----------------------|---------------|
| Flat K=64 | 64 | 64 | 6 |
| Flat K=256 | 256 | 256 | 8 |
| RVQ 3x64 | 192 | 262,144 | 18 |
| RVQ 3x256 | 768 | 16.7M | 24 |

**Recommended starting point:** RVQ with 3 levels x 64 codes.
This gives enormous expressive capacity (262K combinations) while keeping
the codebook small enough to avoid overfitting with 672 training examples.

---

## Implementation Plan

### Files to Create

| File | Contents |
|------|----------|
| `llgbm/canonical.py` | `compute_delta_w()`, `to_svd_compact()`, `from_svd_compact()` — gauge-fixing |
| `llgbm/codebook.py` | `VectorQuantizer`, `ResidualVQ`, `LoRACodebook` (full auto-encoder) |
| `llgbm/code_predictor.py` | `CodePredictor` (Stage 2: text → code sequence) |

### Files to Modify

| File | Changes |
|------|---------|
| `llgbm/losses.py` | Add `CodebookLoss`, `CodePredictionLoss` |
| `llgbm/dataset.py` | Add `AdapterDWDataset` (loads adapters, computes DW-space tokens) |
| `llgbm/training.py` | Add `VQTrainingConfig` dataclass |
| `llgbm/__init__.py` | Export new classes |

### New Notebook

| File | Purpose |
|------|---------|
| `phase_4b_vq_lora.ipynb` | Stage 1 codebook training + analysis + Stage 2 prediction |

### Step-by-Step

#### Step 0: Gauge-Fixing Utilities (`llgbm/canonical.py`)

```python
def compute_delta_w(A: Tensor, B: Tensor, scaling: float) -> Tensor:
    """DW = B @ A * scaling. The unique functional representation."""
    return B @ A * scaling

def to_svd_compact(delta_w: Tensor, rank: int, base_dim: int = 64) -> Tensor:
    """
    Compress DW to compact SVD-based token.

    Returns: (rank + 2*rank*base_dim,) vector containing:
        - rank singular values
        - rank*base_dim flattened U factors (truncated/padded to base_dim rows)
        - rank*base_dim flattened V factors (truncated/padded to base_dim cols)
    """
    U, S, Vh = torch.linalg.svd(delta_w, full_matrices=False)
    U, S, Vh = U[:, :rank], S[:rank], Vh[:rank, :]

    # Sign convention: largest abs element of each U column is positive
    signs = torch.sign(U[U.abs().argmax(dim=0), range(rank)])
    U = U * signs.unsqueeze(0)
    Vh = Vh * signs.unsqueeze(1)

    # Truncate/pad to base_dim for fixed-size representation
    U_base = _truncate_pad(U, base_dim, dim=0)  # (base_dim, rank)
    V_base = _truncate_pad(Vh.T, base_dim, dim=0)  # (base_dim, rank)

    return torch.cat([S, U_base.flatten(), V_base.flatten()])

def from_svd_compact(token: Tensor, rank: int, base_dim: int, out_dim: int, in_dim: int):
    """Recover (A, B) from compact SVD token + periodic extension."""
    S = token[:rank]
    U_base = token[rank:rank + rank*base_dim].view(base_dim, rank)
    V_base = token[rank + rank*base_dim:].view(base_dim, rank)

    sqrt_s = S.sqrt()
    A_base = (V_base * sqrt_s.unsqueeze(0)).T   # (rank, base_dim)
    B_base = (U_base * sqrt_s.unsqueeze(0))      # (base_dim, rank)

    # Periodic extension to full dims (matches existing generator)
    A = A_base[:, :in_dim % base_dim or base_dim].repeat(1, (in_dim//base_dim)+1)[:, :in_dim]
    B = B_base[:out_dim % base_dim or base_dim, :].repeat((out_dim//base_dim)+1, 1)[:out_dim, :]

    return A, B

def canonicalize_adapter(weights: Dict[str, Tensor], scaling: float) -> List[Dict[str, Tensor]]:
    """Convert full adapter dict into list of per-layer DW token dicts."""
    # Group by layer, compute DW for each target, return SVD compact tokens
    ...
```

#### Step 1: VQ Codebook Module (`llgbm/codebook.py`)

Build `VectorQuantizer`, `ResidualVQ`, `LayerEncoder`, `LayerDecoder`,
`LoRACodebook` as described in the Architecture section above.

#### Step 2: Dataset for DW Tokens (`llgbm/dataset.py`)

```python
class AdapterDWDataset(Dataset):
    """
    Dataset of canonicalized adapter weight tokens.

    Loads teacher adapters, computes DW = B @ A * scaling for each layer/target,
    produces compact SVD tokens for the codebook auto-encoder.
    """

    def __init__(self, adapter_paths, scaling=2.0, lora_rank=16, base_dim=64):
        self.samples = []
        for path in adapter_paths:
            weights = load_adapter(path)
            dw_tokens = canonicalize_adapter(weights, scaling)
            self.samples.append({
                "dw_tokens": dw_tokens,      # List[Dict[str, Tensor]] per layer
                "adapter_path": path,
                "task": parse_task(path),
            })

    def __getitem__(self, idx):
        return self.samples[idx]
```

#### Step 3: Stage 1 Training — Learn the Codebook

```python
def train_codebook(
    codebook: LoRACodebook,
    dataset: AdapterDWDataset,
    config: VQTrainingConfig,
):
    """
    Train the VQ auto-encoder on teacher adapters.

    Iterates over adapters, encodes each to layer embeddings, quantizes,
    decodes, and optimizes reconstruction + VQ losses.
    """
    optimizer = torch.optim.AdamW(codebook.parameters(), lr=config.lr)

    for epoch in range(config.num_epochs):
        for sample in dataset:
            dw_tokens = sample["dw_tokens"]

            # Forward
            weights_recon, vq_loss, indices, layer_embeds = codebook(
                dw_tokens, dim_info_per_layer
            )

            # Reconstruction loss in DW-space
            recon_loss = compute_dw_reconstruction_loss(weights_recon, dw_tokens)
            loss = recon_loss + vq_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Reset dead codes each epoch
        codebook.quantizer.reset_dead_codes(all_encoder_outputs)
```

#### Step 4: Codebook Analysis (the payoff)

After Stage 1, run the analysis described in "What the Codebook Reveals":

1. Extract code sequences for all 28 adapters
2. Plot usage histogram, code-task heatmap, layer-position heatmap
3. Measure reconstruction quality (DW MSE, delta cosine, downstream accuracy)
4. Try code swapping experiments

**This analysis alone is a contribution** — even before Stage 2, we'll know:
- How many distinct layer-level adapter patterns exist
- Whether patterns cluster by task
- Whether early/late layer patterns are separable
- The effective dimensionality of the adapter manifold

#### Step 5: Stage 2 Training — Code Prediction from Text

```python
def train_code_predictor(
    predictor: CodePredictor,
    codebook: LoRACodebook,  # Frozen
    text_encoder: nn.Module,  # Frozen
    dataset: AdapterDWDataset,
    config: VQTrainingConfig,
):
    """
    Train text-to-code predictor with frozen codebook.

    For each training adapter:
    1. Encode + quantize to get teacher code sequence
    2. Encode text condition
    3. Predict code sequence from text
    4. Cross-entropy loss against teacher codes
    """
    codebook.eval()
    for p in codebook.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(predictor.parameters(), lr=config.lr)

    for epoch in range(config.num_epochs):
        for sample in dataset:
            # Get teacher codes (from frozen codebook)
            with torch.no_grad():
                layer_embeds = codebook.encode(sample["dw_tokens"])
                _, _, teacher_codes = codebook.quantize(layer_embeds)

            # Get text embedding
            text_emb = text_encoder(sample["condition_ids"], sample["attention_mask"])

            # Predict codes
            predicted_logits = predictor(text_emb)

            # Cross-entropy loss per level
            loss = code_prediction_loss(predicted_logits, teacher_codes)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

#### Step 6 (Optional): End-to-End Fine-Tuning

After Stage 2 converges, optionally fine-tune the whole pipeline end-to-end
with behavioral delta supervision:

```
text → predictor → codes → codebook.decode() → weights → FunctionalLoRA → delta
                                                                           ↓
                                                         Loss(delta, delta_teacher)
```

This requires making the code selection differentiable. Options:
- **Gumbel-Softmax**: Differentiable approximation to argmax
- **Straight-through**: Use argmax forward, softmax backward
- **Soft codes**: Use softmax-weighted codebook lookup during fine-tuning

---

## Config

```python
@dataclass
class VQTrainingConfig:
    """Configuration for VQ-LoRA training."""

    # Base (inherited context)
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    use_small_model: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    hidden_size: int = 896
    num_layers: int = 24

    # Codebook
    num_codes: int = 64           # Codes per VQ level
    embed_dim: int = 256          # Layer embedding dimension
    num_rq_levels: int = 3        # Residual VQ levels
    commitment_cost: float = 0.25
    ema_decay: float = 0.99
    dead_code_threshold: int = 2
    base_dim: int = 64            # Base pattern dim (matches existing generator)

    # Cross-layer attention
    use_cross_layer_attn: bool = True
    cross_layer_heads: int = 4
    cross_layer_depth: int = 2

    # Stage 1: Codebook training
    stage1_lr: float = 3e-4
    stage1_epochs: int = 200      # Many epochs needed for 28 samples
    stage1_lambda_recon: float = 1.0
    stage1_lambda_vq: float = 1.0
    stage1_lambda_behavioral: float = 0.0  # Enable after basic recon works

    # Stage 2: Code prediction
    stage2_lr: float = 1e-4
    stage2_epochs: int = 100
    stage2_label_smoothing: float = 0.1
    stage2_lambda_delta: float = 0.5

    # Data
    canonical_method: str = "svd_compact"
    scaling: float = 2.0          # lora_alpha / lora_rank

    # Analysis
    run_codebook_analysis: bool = True
```

---

## Experiment Plan

### Exp 4b.1: Flat VQ Codebook (Baseline)

- Single VQ level, K=256 codes, embed_dim=256
- No cross-layer attention (each layer encoded independently)
- Train on 25 adapters, validate on 3

**Questions answered:**
- Can we reconstruct adapters from discrete codes at all?
- How many codes does each adapter use?
- What's the reconstruction quality ceiling?

### Exp 4b.2: Residual VQ

- 3 RVQ levels, K=64 each
- Measure reconstruction at each level (1, 2, 3)
- How much does each level contribute?

**Questions answered:**
- Is adapter structure hierarchical (coarse + fine)?
- How many levels are needed for good reconstruction?
- Can we trade quality for efficiency by using fewer levels?

### Exp 4b.3: Cross-Layer Dependencies

- Add 2-layer transformer between encoder and quantizer
- Compare: independent per-layer VQ vs cross-layer-aware VQ

**Questions answered:**
- Do layers share information useful for reconstruction?
- Does cross-layer attention reduce codebook size needed?
- Does it help with code reuse across layers?

### Exp 4b.4: Code Prediction from Text (Stage 2)

- Train CodePredictor on frozen codebook
- Evaluate: top-1 accuracy per layer, per level
- Full pipeline: text → codes → weights → downstream eval

**Questions answered:**
- Is the code prediction problem tractable?
- How does code prediction accuracy translate to downstream accuracy?
- Is this better than the current continuous generator?

### Exp 4b.5: Code Swapping and Composition

- Take adapter A's codes, swap layers 12-18 with adapter B's codes
- Evaluate the hybrid adapter on both A's and B's tasks
- Does code-level composition work better than weight interpolation?

---

## Parameter Budget

| Component | Parameters | When |
|-----------|-----------|------|
| LayerEncoder (7 target projectors + fusion) | ~350K | Stage 1 |
| VQ codebook (3 levels x 64 x 256d) | 49K | Stage 1 |
| Cross-layer transformer (2 layers) | ~530K | Stage 1 |
| LayerDecoder (7 target decoders + scales) | ~350K | Stage 1 |
| **Stage 1 total** | **~1.3M** | |
| | | |
| Text encoder (frozen MiniLM) | 22M (frozen) | Stage 2 |
| CodePredictor (proj + cross-attn + heads) | ~800K | Stage 2 |
| **Stage 2 total** | **~800K trainable** | |

Much smaller than the current generator (~3.6M). The codebook itself is only
49K parameters — the information is in the codes, not in a large network.

---

## Risk Mitigation

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| **Codebook collapse** (few codes used) | High with 28 samples | EMA updates; dead code reset; start with small K; entropy bonus on code usage |
| **Poor reconstruction** (discretization too lossy) | Medium | RVQ adds levels until quality sufficient; increase embed_dim; behavioral recon loss |
| **Overfitting codebook to 28 adapters** | High | Heavy augmentation (noise, interpolation, partial); small K; strong EMA decay |
| **Stage 2 prediction too hard** | Medium | Label smoothing; soft-code predictions; Gumbel-Softmax fine-tuning |
| **Cross-layer attention hurts** (adds params for small data) | Medium | Ablate (Exp 4b.3); use lightweight version (1 layer, 2 heads) |

### Data Augmentation (Critical for 28 Samples)

1. **Weight noise**: Add Gaussian noise (sigma=0.01) to DW before encoding.
   Same adapter, slightly different tokens → encoder learns to be robust.
2. **Layer dropout**: Randomly zero out 1-3 layers' DW → encoder learns
   that some layers can be "identity" (code for zero-effect).
3. **Synthetic interpolation**: Interpolate DW between pairs of adapters
   in weight space → fill gaps between known points.
4. **Sub-adapter extraction**: Each checkpoint contains intermediate training
   states. Extract adapters at multiple training steps for more diversity.
5. **Per-layer shuffling**: Some layers may be interchangeable across adapters.
   Create synthetic adapters by mixing layers from different teachers.

With 5x augmentation: 28 × 5 = 140 effective samples, 140 × 24 = 3,360
layer-level training examples.

---

## Comparison Table (Expected)

| Method | Recon Cos | Delta Cos | ARC-e | BoolQ | HellaSwag | Inference Cost |
|--------|-----------|-----------|-------|-------|-----------|---------------|
| Phase 4 (continuous gen) | N/A | 0.7-0.8 | X% | X% | X% | ~50ms (MLP) |
| Phase 5 (delta-only) | N/A | 0.8-0.9 | X% | X% | X% | ~50ms (MLP) |
| **4b.1: Flat VQ** | 0.7+ | 0.6-0.7 | X% | X% | X% | ~1ms (lookup) |
| **4b.2: RVQ 3-level** | 0.85+ | 0.75-0.85 | X% | X% | X% | ~3ms (3 lookups) |
| **4b.3: + cross-layer** | 0.85+ | 0.8+ | X% | X% | X% | ~3ms |
| **4b.4: text → codes** | 0.8+ | 0.75+ | X% | X% | X% | ~5ms (classify + lookup) |

Inference cost estimates: codebook lookup is O(K*d) per layer, vs O(d^2)
for MLP decoding. With K=64, d=256, this is 64x faster.

---

## Deliverables

1. **`llgbm/canonical.py`** — Gauge-fixing: DW computation, SVD compact tokens
2. **`llgbm/codebook.py`** — `VectorQuantizer`, `ResidualVQ`, `LoRACodebook`
3. **`llgbm/code_predictor.py`** — `CodePredictor` (text → code sequence)
4. **`llgbm/losses.py`** additions — `CodebookLoss`, `CodePredictionLoss`
5. **`llgbm/dataset.py`** additions — `AdapterDWDataset`
6. **`phase_4b_vq_lora.ipynb`** — Full pipeline notebook
7. **Codebook analysis** — Usage histograms, task heatmaps, layer heatmaps
8. **Code swapping experiments** — Discrete compositionality results

## Acceptance Criteria

- [ ] DW-space canonical representation validated (roundtrip error < 1e-3)
- [ ] Stage 1 codebook trains stably; codebook utilization > 50%
- [ ] RVQ reconstruction cosine > 0.8 on held-out adapters
- [ ] Codebook analysis reveals interpretable task/layer structure
- [ ] Stage 2 code prediction accuracy > 70% per layer (top-1)
- [ ] End-to-end generated adapters improve over base model on 2+ tasks
- [ ] Code swapping produces viable hybrid adapters

## Relationship to Other Phases

```
Phase 4  (multi-task, continuous generator)
    │
    ├──→ Phase 4b (discrete codebook)     ← you are here
    │        │
    │        ├──→ 4b.1 Flat VQ baseline
    │        ├──→ 4b.2 Residual VQ
    │        ├──→ 4b.3 Cross-layer attention
    │        ├──→ 4b.4 Text → code prediction
    │        └──→ 4b.5 Code swapping / composition
    │
Phase 5  (delta-only supervision)
    │
Phase 6  (compositionality) ← code swapping is discrete compositionality
```

Phase 4b subsumes much of Phase 6: code-level composition is a cleaner
version of weight-space interpolation. If codes are interpretable, we
get compositionality "for free."
