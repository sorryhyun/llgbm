# Phase 4b — VAE for LoRA: Learning a Latent Adapter Manifold

## Motivation

Phases 4-5 train a generator that maps text conditions directly to LoRA weights
(or behavioral deltas). This works, but the mapping is deterministic and the
latent representation (512-dim projection embeddings) has no explicit structure.

A VAE bottleneck between the text encoder and the weight decoder would give us:

- **Structured latent space** — smooth interpolation, disentanglement
- **Uncertainty quantification** — variance estimates on generated adapters
- **Regularized generalization** — KL pressure prevents memorization of training adapters
- **Conditional generation** — sample diverse adapters for the same task description

Two recent papers validate this direction:

- **FVAE-LoRA** factorizes VAE latents into task-salient vs residual components
- **ICM-LoRA** (IJCAI 2025) uses a Conditional VAE to generate LoRA parameters

This proposal integrates a VAE into the existing LLGBM generator pipeline.

---

## Data: What We Have

From [`Jerrylz/DnD-checkpoints-and-logs`](https://huggingface.co/datasets/Jerrylz/DnD-checkpoints-and-logs):

| Family | Files | Size | Tasks |
|--------|-------|------|-------|
| **Qwen2.5-0.5B LoRA** | 28 | ~125 GB | ARC-c/e, BoolQ, OBQA, PIQA, WinoGrande, HellaSwag + ablations |
| **Qwen2.5-1.5B LoRA** | 3 | ~21 GB | Coding, Math |
| **Qwen2.5-7B LoRA** | 2 | ~16 GB | Coding, Math |
| **Qwen3-VL LoRA** | 1 | ~10 GB | Multimodal |

The 0.5B family is the natural starting point: 28 checkpoints across 6+ tasks,
plus extractor variants (GloVe, T5, Qwen-7B), conditioning ablations
(128/256/512/1024-dim), and train/test split experiments. This is enough
diversity to learn a meaningful manifold while keeping shapes consistent.

Each `.pth` checkpoint is a full DnD generator state. The LoRA weights within
each can be extracted via the existing `Qwen2515LoRA_Tokenizer2D` tokenizer
or loaded directly as `{layer}.{proj}.lora_{A,B}.weight` dicts.

**Recommendation:** Start with `qwen0.5lora__*` (28 checkpoints). Hold out
2-3 for validation (e.g., one ARC variant, one reconstruction variant, one
conditioning ablation).

---

## The Core Problem: Non-Identifiability of (A, B)

The same functional update `DW = B @ A` can be represented by infinitely many
`(A, B)` pairs — any invertible matrix `R` gives `B' = B @ R`, `A' = R^{-1} @ A`
with identical `DW`. If we VAE raw `(A, B)` tensors, the encoder sees the same
function as many distinct points in weight-space, producing a mushy latent that
doesn't correspond to stable meaning.

### Trick 1: Canonical Representation (Gauge Fixing)

Instead of encoding raw `(A, B)`, encode one of:

| Representation | Formula | Pros | Cons |
|----------------|---------|------|------|
| **DW-space** | `DW_i = B_i @ A_i * scaling` | Unique, compact | Loses factorization; `rank(DW)` reveals rank but dims are large |
| **SVD-canonical** | `DW = U @ S @ V^T`, store `(U[:,:r], S[:r], V[:,:r])` | Unique up to sign; compact | Sign ambiguity on singular vectors |
| **QR-canonical** | `Q, R = qr(A^T)`, store `(R^T, B @ Q)` | Deterministic (positive diag R) | Less standard |
| **A-normalized** | `A' = A / ||A||_row`, `B' = B * ||A||_row` | Simple; preserves factorization | Not fully canonical |

**Recommended for Phase 4b:** Use **DW-space** (`B @ A * scaling`) per layer per
target. This is the most principled choice — it's the actual function the adapter
computes, and it's unique. The decoder then needs to output a low-rank
factorization of DW, which can be done by predicting `U, S, V` or by directly
predicting `A, B` with a rank constraint.

For Qwen2.5-0.5B with rank=16:

```
DW per target: (out_dim, in_dim)
  q_proj: (896, 896)     = 802K params
  k_proj: (128, 896)     = 115K params  (GQA heads)
  v_proj: (128, 896)     = 115K params
  o_proj: (896, 896)     = 802K params
  gate:   (4864, 896)    = 4.4M params
  up:     (4864, 896)    = 4.4M params
  down:   (896, 4864)    = 4.4M params
```

These are too large to flatten naively. We must compress per-layer DW into a
token representation.

### Trick 2: Low-Rank Token Representation

Since `rank(DW) = rank(B @ A) <= 16`, each DW lives on a rank-16 manifold.
Represent each target's update as a **compact token**:

```
token_i = [vec(S_i), vec(U_i[:, :r]), vec(V_i[:, :r])]
```

Or more practically, since the current generator already produces `(A, B)` via
64-dim base patterns with periodic extension:

```
token_i = [A_base_i (rank x 64), B_base_i (64 x rank)]
         = 2 * rank * 64 = 2048 floats per target
```

This matches the existing `A_decoder`/`B_decoder` output shape and keeps the
periodic extension mechanism intact.

---

## Architecture

### Current Pipeline (Phase 5)

```
Text (B, N, L)
    |
[TextEncoder] --> (B, N, 384)
    |
[SharedProj 384->512] --> (B, N, 512)
    |
    +---> [DeltaHead] --> d_pred (B, 896)
    |
    +---> pool(N) --> (B, 512) --> [LoRA Head] --> weights
```

### Proposed Pipeline (Phase 4b)

```
Text (B, N, L)
    |
[TextEncoder] --> (B, N, 384)
    |
[SharedProj 384->512] --> (B, N, 512)
    |
pool(N) --> (B, 512)       [optional: also DeltaHead]
    |
[VAE Encoder] --> mu (B, z_dim), logvar (B, z_dim)
    |
[Reparameterize] --> z (B, z_dim)
    |
    +---> [LayerCodeGen] --> z_layer (B, num_layers, z_layer_dim)
    |
    +---> [LoRA Decoder] --> weights per layer
    |
    +---> [DeltaDecoder] --> d_pred (B, hidden_size)  [fast path]
```

### Key Components

#### 2a. VAE Encoder

```python
class VAEEncoder(nn.Module):
    """Maps condition embedding to latent distribution."""

    def __init__(self, input_dim=512, hidden_dim=512, latent_dim=64):
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = F.gelu(self.fc1(x))
        h = F.gelu(self.fc2(h))
        return self.fc_mu(h), self.fc_logvar(h)
```

**Latent dimension:** 64 (compresses 512-dim condition to 64-dim; the existing
generator already does 512 -> `num_projections * proj_embed_dim`, so 64 is a
reasonable bottleneck).

#### 2b. Hierarchical Latent Structure

Adapters are not i.i.d. matrices — they have cross-layer structure ("this task
tweaks attention in mid layers"). A flat latent ignores this.

```python
class HierarchicalDecoder(nn.Module):
    """Decode z_global into per-layer codes + LoRA weights."""

    def __init__(self, latent_dim=64, num_layers=24, layer_code_dim=32):
        # Global-to-layer mapping
        self.layer_code_gen = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, num_layers * layer_code_dim),
        )
        # Per-target decoders (shared across layers)
        self.target_decoders = nn.ModuleDict({
            target: nn.Sequential(
                nn.Linear(latent_dim + layer_code_dim, 256),
                nn.GELU(),
                nn.Linear(256, rank * 64 * 2),  # A_base + B_base
            )
            for target in ['q', 'k', 'v', 'o', 'gate', 'up', 'down']
        })

    def forward(self, z_global):
        B = z_global.shape[0]
        # Generate per-layer codes
        layer_codes = self.layer_code_gen(z_global)
        layer_codes = layer_codes.view(B, self.num_layers, -1)

        weights = {}
        for layer_idx in range(self.num_layers):
            z_layer = layer_codes[:, layer_idx]            # (B, layer_code_dim)
            z_combined = torch.cat([z_global, z_layer], -1)  # (B, latent + layer_code)

            for target_name, decoder in self.target_decoders.items():
                raw = decoder(z_combined)  # (B, rank*64*2)
                A_base, B_base = raw.chunk(2, dim=-1)
                # ... periodic extension + scaling (same as current generator)
```

This gives the model `z_global` for "what kind of adapter" and `z_layer[i]` for
"how this layer participates." The per-target decoders are shared across layers
(parameter efficient) but conditioned on layer-specific codes.

#### 2c. Prior Choice: VQ-VAE vs Gaussian vs Mixture

The DnD adapters span distinct task families (commonsense reasoning, math,
coding, multimodal). A plain Gaussian prior averages modes ("blurry adapters").

| Prior | When to Use | Complexity |
|-------|-------------|------------|
| **Gaussian** | Baseline; if tasks are similar | Low |
| **GMM / VampPrior** | Continuous but multimodal | Medium |
| **VQ-VAE** | Distinct clusters; discrete codes | Medium |
| **Flow prior** | Expressive continuous | High |

**Recommended for Phase 4b:** Start with **Gaussian** (simplest, validates the
architecture), then upgrade to **VQ-VAE** if latent space shows clear clusters.

For VQ-VAE, the codebook would be:

```python
class VQLayer(nn.Module):
    """Vector Quantization layer."""

    def __init__(self, latent_dim=64, num_codes=128, commitment_cost=0.25):
        self.codebook = nn.Embedding(num_codes, latent_dim)
        self.commitment_cost = commitment_cost

    def forward(self, z_e):
        # Find nearest codebook entry
        distances = torch.cdist(z_e, self.codebook.weight)
        indices = distances.argmin(dim=-1)
        z_q = self.codebook(indices)

        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()

        # Losses
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        return z_q_st, vq_loss, indices
```

**Number of codes:** 128 (we have ~28 adapters but want sub-adapter-level
granularity — different layers/behaviors within adapters should activate
different codes).

---

## Loss Function

### Gaussian VAE

```
L_total = L_recon + beta * L_KL [+ L_delta + L_consistency]

where:
  L_recon  = MSE(DW_pred, DW_teacher)  or  MSE(A_pred, A_teacher) + MSE(B_pred, B_teacher)
  L_KL     = KL(q(z|x), N(0,I))
  L_delta  = MSE(d_pred, d_teacher)    [behavioral supervision, optional]
  L_consistency = MSE(d_computed, d_pred)  [computed delta matches predicted]
```

### VQ-VAE

```
L_total = L_recon + L_codebook + beta_commit * L_commitment [+ L_delta]
```

### Beta Annealing Schedule

KL collapse is the main risk with VAE for structured outputs. Use cyclical
annealing or monotonic warmup:

```
beta(t) = min(beta_max, beta_min + (beta_max - beta_min) * t / warmup_steps)
```

With `beta_min=0.0`, `beta_max=0.1`, `warmup_steps=200`. Keep beta small —
reconstruction quality matters more than prior matching for LoRA generation.

### Integration with Existing Losses

Phase 4b's loss is a superset of the existing `DeltaGuidedLoss`:

```python
class VAELoRALoss(nn.Module):
    """VAE loss for LoRA generation with optional behavioral supervision."""

    def __init__(
        self,
        lambda_recon: float = 1.0,       # Weight reconstruction
        lambda_kl: float = 0.1,          # KL divergence (or VQ losses)
        lambda_delta: float = 1.0,        # Behavioral delta matching
        lambda_consistency: float = 0.5,  # d_computed vs d_predicted
        beta_schedule: str = "warmup",    # "constant", "warmup", "cyclical"
        beta_warmup_steps: int = 200,
        beta_max: float = 0.1,
        normalize_delta: bool = True,
        recon_target: str = "delta_w",    # "delta_w", "ab_canonical", "both"
    ):
        ...
```

---

## Implementation Plan

### Files to Create

| File | Purpose |
|------|---------|
| `llgbm/vae.py` | `VAEEncoder`, `VAEDecoder`, `VQLayer`, `HierarchicalDecoder`, reparameterize |
| `llgbm/canonical.py` | `compute_delta_w()`, `svd_canonicalize()`, `from_delta_w()` — gauge-fixing utilities |

### Files to Modify

| File | Changes |
|------|---------|
| `llgbm/generator.py` | Add `LoRAGeneratorVAE` class (extends `LoRAGeneratorWithDeltaHead`) |
| `llgbm/losses.py` | Add `VAELoRALoss` class |
| `llgbm/training.py` | Add `VAETrainingConfig` dataclass; modify `train_step()` to handle VAE outputs |
| `llgbm/dataset.py` | Add `DnDCheckpointDataset` for loading .pth files from the HF dataset |
| `llgbm/__init__.py` | Export new classes |

### New Notebook

| File | Purpose |
|------|---------|
| `phase_4b_vae_lora.ipynb` | End-to-end: load data, train VAE, visualize latent space, evaluate |

### Step-by-Step

#### Step 1: Canonical Representation (`llgbm/canonical.py`)

Build utilities to convert between `(A, B)` and canonical forms:

```python
def compute_delta_w(A: Tensor, B: Tensor, scaling: float) -> Tensor:
    """Compute DW = B @ A * scaling. The unique functional update."""
    return B @ A * scaling

def to_svd_canonical(delta_w: Tensor, rank: int) -> Tuple[Tensor, Tensor, Tensor]:
    """SVD with sign convention: largest element of each U column is positive."""
    U, S, Vh = torch.linalg.svd(delta_w, full_matrices=False)
    U, S, Vh = U[:, :rank], S[:rank], Vh[:rank, :]
    # Sign convention
    signs = torch.sign(U[U.abs().argmax(dim=0), range(rank)])
    U = U * signs.unsqueeze(0)
    Vh = Vh * signs.unsqueeze(1)
    return U, S, Vh

def from_svd_to_ab(U: Tensor, S: Tensor, Vh: Tensor) -> Tuple[Tensor, Tensor]:
    """Recover (A, B) from SVD: A = diag(sqrt(S)) @ Vh, B = U @ diag(sqrt(S))."""
    sqrt_s = S.sqrt()
    A = sqrt_s.unsqueeze(1) * Vh    # (rank, in_dim)
    B = U * sqrt_s.unsqueeze(0)     # (out_dim, rank)
    return A, B
```

Also provide a function to canonicalize an entire adapter dict:

```python
def canonicalize_adapter(
    weights: Dict[str, Tensor],
    scaling: float,
    method: str = "delta_w",
) -> Dict[str, Tensor]:
    """Convert (A_key, B_key) pairs to canonical representation."""
    ...
```

#### Step 2: DnD Checkpoint Dataset (`llgbm/dataset.py` addition)

Load `.pth` checkpoints from the HF dataset and extract LoRA weights:

```python
class DnDCheckpointDataset(Dataset):
    """
    Load LoRA weights from DnD .pth checkpoints.

    Each .pth is a full generator state. We extract the teacher LoRA weights
    it was trained to produce, canonicalize them, and pair with metadata
    (task, conditioning strategy, etc.) parsed from the filename.
    """

    def __init__(
        self,
        checkpoint_paths: List[str],
        canonical_method: str = "delta_w",
        scaling: float = 2.0,  # lora_alpha / lora_rank
    ):
        ...

    def _parse_filename(self, path: str) -> Dict[str, str]:
        """Extract task/model/variant from filename like 'qwen0.5lora__ARC-e.pth'."""
        ...
```

This bridges the HF dataset to the training pipeline.

#### Step 3: VAE Module (`llgbm/vae.py`)

```python
class LoRAVAE(nn.Module):
    """
    VAE for LoRA adapter generation.

    Encoder: condition embedding -> (mu, logvar)
    Decoder: z -> per-layer LoRA weights (via hierarchical decoding)

    Supports both Gaussian VAE and VQ-VAE modes.
    """

    def __init__(
        self,
        condition_dim: int = 512,
        latent_dim: int = 64,
        num_layers: int = 24,
        layer_code_dim: int = 32,
        lora_rank: int = 16,
        base_pattern_dim: int = 64,
        num_targets: int = 7,
        mode: str = "gaussian",       # "gaussian" or "vq"
        num_vq_codes: int = 128,
        dim_info: List[Dict] = None,  # From generator._build_dim_info()
    ):
        super().__init__()
        self.mode = mode
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = VAEEncoder(condition_dim, 512, latent_dim)

        # VQ layer (if using VQ-VAE)
        if mode == "vq":
            self.vq = VQLayer(latent_dim, num_vq_codes)

        # Hierarchical decoder
        self.decoder = HierarchicalDecoder(
            latent_dim=latent_dim,
            num_layers=num_layers,
            layer_code_dim=layer_code_dim,
            lora_rank=lora_rank,
            base_pattern_dim=base_pattern_dim,
            num_targets=num_targets,
            dim_info=dim_info,
        )

        # Optional: delta prediction from z (fast path)
        self.delta_from_z = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.GELU(),
            nn.Linear(512, 896),  # hidden_size
        )

    def encode(self, condition_emb):
        """Encode condition to latent distribution."""
        mu, logvar = self.encoder(condition_emb)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Sample z via reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # Use mean at inference

    def decode(self, z):
        """Decode z to LoRA weights."""
        return self.decoder(z)

    def forward(self, condition_emb, return_delta=True):
        mu, logvar = self.encode(condition_emb)

        if self.mode == "vq":
            z_e = mu  # Use mu as continuous embedding
            z, vq_loss, indices = self.vq(z_e)
            kl_loss = vq_loss
        else:
            z = self.reparameterize(mu, logvar)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()

        lora_weights = self.decode(z)

        results = {
            "lora_weights": lora_weights,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "kl_loss": kl_loss,
        }

        if return_delta:
            results["delta_pred"] = self.delta_from_z(z)

        if self.mode == "vq":
            results["vq_indices"] = indices

        return results
```

#### Step 4: Generator Integration (`llgbm/generator.py` addition)

```python
class LoRAGeneratorVAE(nn.Module):
    """
    LoRA Generator with VAE bottleneck.

    Extends LoRAGeneratorWithDeltaHead by inserting a VAE between
    the shared projection and the LoRA/delta heads.
    """

    def __init__(
        self,
        cfg: TrainingConfig,
        text_encoder: Optional[nn.Module] = None,
        vae_mode: str = "gaussian",
        latent_dim: int = 64,
        num_vq_codes: int = 128,
    ):
        ...
        # Reuse text encoder + shared_proj from parent
        # Insert VAE between shared_proj output and heads
        self.vae = LoRAVAE(
            condition_dim=512,
            latent_dim=latent_dim,
            num_layers=cfg.num_layers,
            mode=vae_mode,
            num_vq_codes=num_vq_codes,
            dim_info=self.dim_info,
        )

    def forward(self, condition_ids, attention_mask=None,
                return_delta=True, return_lora=True):
        # Encode text -> shared projection (same as parent)
        embeddings = self._encode_embeddings(condition_ids, attention_mask)
        shared = self.shared_proj(embeddings)
        shared_pooled = shared.mean(dim=1)  # (B, 512)

        # VAE forward
        vae_out = self.vae(shared_pooled, return_delta=return_delta)

        results = {
            "mu": vae_out["mu"],
            "logvar": vae_out["logvar"],
            "z": vae_out["z"],
            "kl_loss": vae_out["kl_loss"],
            "embeddings": embeddings,
        }

        if return_lora:
            results["lora_weights"] = vae_out["lora_weights"]
        if return_delta:
            results["delta_pred"] = vae_out["delta_pred"]

        return results

    def generate_from_z(self, z):
        """Generate LoRA weights from a latent code (for interpolation/sampling)."""
        return self.vae.decode(z)

    def interpolate(self, condition_a, condition_b, alpha=0.5):
        """Interpolate between two conditions in latent space."""
        mu_a, _ = self.vae.encode(self._embed(condition_a))
        mu_b, _ = self.vae.encode(self._embed(condition_b))
        z_interp = (1 - alpha) * mu_a + alpha * mu_b
        return self.vae.decode(z_interp)
```

#### Step 5: Loss Function (`llgbm/losses.py` addition)

```python
class VAELoRALoss(nn.Module):
    """
    Loss = lambda_recon * L_recon(weights_pred, weights_teacher)
         + beta(t) * L_KL(q(z|x), p(z))
         + lambda_delta * L_delta(d_pred, d_teacher)
         + lambda_consistency * L_consistency(d_computed, d_pred)
    """

    def __init__(self, ...):
        ...

    def _get_beta(self, step: int) -> float:
        """Beta annealing schedule."""
        if self.beta_schedule == "constant":
            return self.beta_max
        elif self.beta_schedule == "warmup":
            return min(self.beta_max, self.beta_max * step / self.beta_warmup_steps)
        elif self.beta_schedule == "cyclical":
            cycle = step % (2 * self.beta_warmup_steps)
            if cycle < self.beta_warmup_steps:
                return self.beta_max * cycle / self.beta_warmup_steps
            return self.beta_max

    def forward(self, vae_output, teacher_weights, delta_teacher,
                delta_computed=None, step=0):
        losses = {}

        # Reconstruction loss (on canonical DW or raw A,B)
        losses["loss_recon"] = self._recon_loss(
            vae_output["lora_weights"], teacher_weights
        )

        # KL / VQ loss
        beta = self._get_beta(step)
        losses["loss_kl"] = vae_output["kl_loss"]
        losses["loss_kl_scaled"] = beta * vae_output["kl_loss"]

        # Delta loss
        if "delta_pred" in vae_output and delta_teacher is not None:
            losses["loss_delta"] = self._delta_loss(
                vae_output["delta_pred"], delta_teacher
            )

        # Consistency loss
        if delta_computed is not None and "delta_pred" in vae_output:
            losses["loss_consistency"] = self._delta_loss(
                delta_computed, vae_output["delta_pred"].detach()
            )

        # Total
        losses["loss"] = (
            self.lambda_recon * losses["loss_recon"]
            + losses["loss_kl_scaled"]
            + self.lambda_delta * losses.get("loss_delta", 0)
            + self.lambda_consistency * losses.get("loss_consistency", 0)
        )

        losses["beta"] = beta
        return losses
```

#### Step 6: Training Config Extension

```python
@dataclass
class VAETrainingConfig(TrainingConfig):
    """Extended config for VAE training."""

    # VAE settings
    vae_mode: str = "gaussian"        # "gaussian" or "vq"
    latent_dim: int = 64
    num_vq_codes: int = 128
    layer_code_dim: int = 32

    # Loss weights
    lambda_recon: float = 1.0
    lambda_kl: float = 0.1
    lambda_delta_vae: float = 1.0
    lambda_consistency_vae: float = 0.5

    # Beta schedule
    beta_schedule: str = "warmup"     # "constant", "warmup", "cyclical"
    beta_max: float = 0.1
    beta_warmup_steps: int = 200

    # Canonical representation
    canonical_method: str = "delta_w"  # "delta_w", "svd", "raw"
    recon_target: str = "delta_w"     # What reconstruction loss compares
```

---

## Experiment Plan

### Experiment 4b.1: Gaussian VAE Baseline

**Config:**
- `vae_mode="gaussian"`, `latent_dim=64`
- `canonical_method="delta_w"` (gauge-fixed representation)
- `beta_schedule="warmup"`, `beta_max=0.1`, `beta_warmup_steps=200`
- `lambda_recon=1.0`, `lambda_delta_vae=1.0`

**Train on:** 25 qwen0.5lora checkpoints, hold out 3 for validation.

**Evaluate:**
- Reconstruction quality: MSE(DW_pred, DW_teacher) and cosine similarity
- Delta matching: cosine(d_computed, d_teacher)
- Downstream accuracy: ARC-e, BoolQ, HellaSwag via `compute_accuracy_with_lora()`
- Latent space quality: t-SNE of mu colored by task, interpolation smoothness

**Success criteria:**
- Reconstruction cosine > 0.8 on held-out adapters
- Downstream accuracy within 3% of teacher adapters
- Latent space shows task clustering

### Experiment 4b.2: VQ-VAE with Discrete Codes

**Config:**
- `vae_mode="vq"`, `latent_dim=64`, `num_vq_codes=128`
- Same beta/lambda as 4b.1

**Evaluate:**
- Same metrics as 4b.1
- Additionally: codebook utilization (what fraction of codes are used),
  code-to-task mapping (does each code correspond to a task?)

**Success criteria:**
- Codebook utilization > 50% (no codebook collapse)
- Sharper reconstructions than Gaussian VAE
- Codes cluster by task family

### Experiment 4b.3: Hierarchical Latent

**Config:**
- `layer_code_dim=32` (per-layer refinement)
- Compare: flat `z` (64-dim) vs hierarchical `z_global` (64) + `z_layer` (24x32)

**Evaluate:**
- Does hierarchical structure improve reconstruction of layer-varying adapters?
- Ablation: freeze `z_global` and vary `z_layer` — do individual layers change
  independently?

### Experiment 4b.4: Conditional Generation (the product)

The real use case: generate adapters from task descriptions.

```
"Adapt model for ARC-style multiple choice reasoning" --> z --> LoRA weights
```

**Evaluate:**
- Generate adapters for held-out task descriptions
- Compare to nearest-neighbor retrieval baseline (Phase 5 B2)
- Measure diversity: sample N adapters for same condition, measure variance

---

## Training Protocol

### Phase 4b.0: Validate Canonical Representation

Before training any VAE, verify that:

1. `DW = B @ A * scaling` reconstructs the adapter behavior
   (compute delta from DW vs from original A,B — should be identical)
2. SVD canonicalization is invertible: `from_svd_to_ab(to_svd_canonical(DW))` recovers
   original DW within float precision
3. Canonical form reduces variance across equivalent adapters (if we have any
   augmented pairs)

### Phase 4b.1: Train Reconstruction-Only VAE

1. Load 25 qwen0.5lora checkpoints
2. Extract and canonicalize LoRA weights
3. Train VAE with `L_recon + beta * L_KL` only (no delta supervision)
4. Validate: reconstruction quality on held-out 3 checkpoints
5. Visualize latent space

### Phase 4b.2: Add Behavioral Supervision

1. Compute delta cache for all 28 checkpoints (reuse Phase 1 infrastructure)
2. Add `L_delta` to the VAE loss
3. Compare: recon-only vs recon+delta

### Phase 4b.3: Full Pipeline

1. End-to-end: text -> VAE -> LoRA -> delta -> loss
2. Train with full `VAELoRALoss`
3. Evaluate on held-out tasks
4. Compare to Phase 4 (multi-task, no VAE) and Phase 5 (delta-only, no VAE)

---

## Parameter Budget

For Qwen2.5-0.5B (24 layers, 7 targets/layer = 168 projections):

| Component | Parameters |
|-----------|-----------|
| Text encoder (frozen MiniLM) | 22M (frozen) |
| Shared projection (384->512) | ~200K |
| VAE encoder (512->512->64 mu+logvar) | ~400K |
| Hierarchical decoder (64->24*32, per-target MLPs) | ~1.5M |
| Per-target weight decoders (7 shared, 321->2048 each) | ~1M |
| Delta-from-z head (64->512->896) | ~500K |
| Scales (168 x 2) | 336 |
| **Total trainable** | **~3.6M** |

Comparable to the current `LoRAGeneratorWithDeltaHead` (~3-4M params).
The VAE adds ~400K for the encoder and reuses the decoder budget.

---

## Risk Mitigation

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| **KL collapse** (posterior = prior, ignoring z) | High | Beta annealing; start with beta=0; free bits; use delta supervision to force z to carry information |
| **Blurry reconstruction** (Gaussian prior averages modes) | Medium | Upgrade to VQ-VAE (Exp 4b.2); use cosine loss instead of MSE |
| **Codebook collapse** (VQ-VAE uses few codes) | Medium | EMA codebook update; codebook reset for dead codes; entropy regularization |
| **Gauge ambiguity** (despite canonicalization) | Low | DW-space is unique; validate in Phase 4b.0 |
| **Overfitting** (28 checkpoints is small) | High | Strong regularization via KL; data augmentation (noise on weights); hold out 3 for validation |
| **Memory pressure** (VAE + base model + delta computation) | Medium | Reuse existing memory management patterns; gradient checkpointing; sequential delta computation |

### Data Augmentation for Small Dataset

28 checkpoints is small. Augment by:

1. **Noise injection:** Add Gaussian noise to teacher weights, creating synthetic
   neighbors in weight space
2. **Interpolation:** Create synthetic adapters by interpolating between pairs
   of teacher adapters (in DW-space)
3. **Partial adapters:** Use subsets of layers (zero out some layers' LoRA) as
   additional training examples
4. **Cross-checkpoint:** If training logs have intermediate checkpoints,
   use those as additional data points along the optimization trajectory

---

## Comparison Table (Expected)

| Method | Recon Cos | Delta Cos | ARC-e | BoolQ | HellaSwag | Latent Structure |
|--------|-----------|-----------|-------|-------|-----------|-----------------|
| Phase 4 (multi-task, no VAE) | N/A | 0.7-0.8 | X% | X% | X% | None |
| Phase 5 (delta-only) | N/A | 0.8-0.9 | X% | X% | X% | None |
| **4b.1: Gaussian VAE** | 0.8+ | 0.7-0.8 | X% | X% | X% | Smooth, some blur |
| **4b.2: VQ-VAE** | 0.85+ | 0.75-0.85 | X% | X% | X% | Discrete, sharp |
| **4b.3: Hierarchical** | 0.85+ | 0.8+ | X% | X% | X% | Layer-aware |
| **4b.4: Conditional gen** | 0.7+ | 0.7+ | X% | X% | X% | Task-conditioned |

---

## Deliverables

1. **`llgbm/canonical.py`** — Gauge-fixing utilities (DW-space, SVD canonical)
2. **`llgbm/vae.py`** — VAE encoder, decoder, VQ layer, hierarchical decoder
3. **`llgbm/generator.py`** — `LoRAGeneratorVAE` class
4. **`llgbm/losses.py`** — `VAELoRALoss` with beta annealing
5. **`llgbm/dataset.py`** — `DnDCheckpointDataset` for HF .pth files
6. **`phase_4b_vae_lora.ipynb`** — End-to-end notebook for all experiments
7. **Latent space visualizations** — t-SNE, interpolation plots, codebook usage
8. **Comparison table** — Filled in with actual numbers from experiments

## Acceptance Criteria

- [ ] Canonical representation is validated (Phase 4b.0)
- [ ] VAE trains stably without KL collapse (active KL > 0.1 nats)
- [ ] Reconstruction cosine similarity > 0.8 on held-out checkpoints
- [ ] Generated adapters improve over base model on at least 2 tasks
- [ ] Latent space shows interpretable structure (task clustering in t-SNE)
- [ ] Interpolation between two task adapters produces smoothly varying behavior
- [ ] VQ-VAE codebook utilization > 50%

## Relationship to Other Phases

```
Phase 4  (multi-task)
    |
    +---> Phase 4b (VAE bottleneck) <-- you are here
    |         |
    |         +---> 4b.1 Gaussian VAE
    |         +---> 4b.2 VQ-VAE
    |         +---> 4b.3 Hierarchical
    |         +---> 4b.4 Conditional generation
    |
Phase 5  (delta-only)
    |
Phase 6  (compositionality) <-- VAE latent space enables better composition
```

Phase 4b is orthogonal to Phase 5 (delta-only) — they can proceed in parallel.
Phase 6 (compositionality) benefits from VAE because interpolation/composition
in latent space is better-behaved than in raw weight space.

---

## Next Steps After 4b

If the VAE approach works:

1. **Scale to 1.5B family** — Use the 3 qwen1.5lora checkpoints for transfer
   learning (fine-tune the latent decoder, keep encoder frozen)
2. **Cross-task generation** — Generate adapters for tasks not seen in training
   (zero-shot in adapter space)
3. **Adapter search** — Use the latent space for efficient search over adapter
   configurations (Bayesian optimization in z-space)
4. **Publication angle** — "Structured Latent Spaces for LoRA Generation"
   combining gauge fixing + hierarchical VAE + behavioral supervision
