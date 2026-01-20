# LLGBM: Learning LoRA Generator by Behavioral Matching

A novel approach to training prompt-conditioned LoRA generators using **behavioral supervision**. Instead of only matching LoRA weights via MSE, the generator is trained to match **delta activations**—the difference in hidden states between base and adapted models.

## Core Innovation

**Standard DnD Loss:** `L = L_weight(predicted_weights, teacher_weights)`

**LLGBM Loss:** `L = λ_w * L_weight + λ_d * L_delta`

Where delta is computed as:
```
Δ(M, B) = v(M) - v(B)
v(M) = E[h^(L)_M(p, last_token)]  for p in P (probe texts)
```

The generator learns to produce LoRA weights that induce the *same behavioral shift* as the teacher adapters, not just numerically similar weights.

---

## Key Results

| Achievement | Status |
|-------------|--------|
| DnD framework reproduced on Qwen2.5 | ✅ Phase 0 |
| Delta supervision is differentiable | ✅ Phase 3 |
| Multi-task training (weights + deltas) | ✅ Phase 4 |
| Ablation studies (3 trials × 2 configs) | ✅ Phase 4.5 |
| Delta-only training validated | ✅ Phase 5 |

### Validated Findings

- **Tokenizer roundtrip error < 1e-3**: DnD's tokenize→detokenize maintains LoRA weight fidelity
- **Delta norms ~0.1**: Reasonable scale for supervision signals; deltas cluster by domain in t-SNE
- **100% gradient flow**: All LoRA tensors receive gradients through FunctionalLoRA hooks
- **Loss convergence**: Multi-task training successfully balances weight and delta objectives

---

## Technical Approach

### Pipeline Architecture

```
Text Prompts
    ↓
[PretrainedTextEncoder] → condition embeddings (384-dim, MiniLM-L6-v2)
    ↓
[LoRAGenerator] → LoRA weights (A, B matrices per layer)
    ↓
[FunctionalLoRA] → Apply LoRA via hooks (memory-efficient, gradient-enabled)
    ↓
[DeltaComputation] → Δ_pred via probe texts
    ↓
[MultiTaskLoss] → λ_w * L_weight + λ_d * L_delta
```

### FunctionalLoRA (Key Innovation)

Instead of modifying model weights in-place, `FunctionalLoRA` applies LoRA weights through forward hooks:
- Memory-efficient: No weight cloning
- Gradient-enabled: Full backprop to generator
- Production-ready: Tested on Qwen2.5-0.5B and 1.5B

---

## Completed Phases

| Phase | Notebook | What Was Done | Key Metrics |
|-------|----------|---------------|-------------|
| 0 | `phase_0_baseline.ipynb` | DnD reproduction, tokenizer roundtrip | Error < 1e-3 |
| 1 | `phase_1_delta.ipynb` | Delta computation & caching, t-SNE visualization | Norms ~0.1, clusters by domain |
| 2 | `phase_2_dataset.ipynb` | Dataset returns `(tokens, condition, delta)` tuples | 100% batch iteration success |
| 3 | `phase_3_differentiable.ipynb` | FunctionalLoRA with gradient flow | All LoRA tensors get gradients |
| 4 | `phase_4_multitask.ipynb` | Multi-task training (weights + deltas) | Loss converges |
| 4.5 | `phase_4_5_ablations.ipynb` | Delta-only vs delta-guided comparison | 3 trials × 2 configs |
| 5 | `phase_5_delta_only.ipynb` | Behavioral supervision only (no weight MSE) | Validates delta-only approach |

---

## Package Structure

```
llgbm/
├── probes.py        # Probe templates: create_generic_probes(), create_domain_probes()
├── delta.py         # Delta computation: compute_base_activation(), DeltaCache
├── dataset.py       # Datasets: DeltaAugmentedDataset, RealAdapterDataset
├── functional.py    # FunctionalLoRA: differentiable LoRA application via hooks
├── training.py      # TrainingConfig, MultiTaskLoss, DeltaOnlyLoss, train()
├── generator.py     # LoRAGenerator, create_generator()
├── text_encoder.py  # PretrainedTextEncoder, EmbeddingCache
├── evaluation.py    # Task evaluation: compute_accuracy_with_lora()
└── ablations.py     # AblationConfig, run_ablations()
```

---

## Quick Start

### Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -r dnd_repo/requirements.txt
uv pip install peft scikit-learn matplotlib seaborn
```

### Train Teacher LoRA Adapters

```bash
python train_lora_adapters.py --tasks arc_e boolq gsm8k --lora_rank 8
```

### Run Notebooks

The notebooks are designed to run sequentially:

1. **Phase 0**: Verify DnD framework works
2. **Phase 1**: Compute and cache delta embeddings
3. **Phase 2**: Test dataset with delta labels
4. **Phase 3**: Verify gradient flow through FunctionalLoRA
5. **Phase 4**: Train with multi-task loss
6. **Phase 4.5**: Run ablation studies
7. **Phase 5**: Train with delta-only supervision

---

## Model Configuration

| Config | Qwen2.5-0.5B (testing) | Qwen2.5-1.5B (production) |
|--------|------------------------|---------------------------|
| Layers | 24 | 28 |
| Hidden | 896 | 1536 |
| Intermediate | 4864 | 8960 |
| LoRA rank | 8 | 16 |
| LoRA alpha | 16 | 32 |

LoRA targets: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`

---

## Training Objectives

### Multi-task Loss (Phase 4)
```python
L = lambda_weight * MSE(tokens_pred, tokens_teacher)
  + lambda_delta * MSE(delta_pred, delta_teacher)
```

### Delta-only Loss (Phase 5)
```python
L = MSE(delta_pred, delta_teacher) + regularization
```

---

## Data Layout

```
checkpoints/
├── {task}/              # arc_e, boolq, gsm8k, etc.
│   └── {task}_{idx}/    # e.g., arc_e_000
│       ├── adapter_model.safetensors
│       └── prompts.json
├── deltas/              # Cached delta embeddings
│   ├── manifest.json
│   ├── base_activation.npy
│   └── {adapter_id}.npy
└── manifest.json        # Central adapter index

data/
├── ARC-e_train.json
├── BoolQ_train.json
└── GSM8K_train.json
```

---

## Future Work

| Phase | Goal |
|-------|------|
| 6 | **Compositionality**: Can compositional adapters be built via compositional deltas? |
| 7 | **Scaling**: Package for reproducibility, larger models, more tasks |

---

## References

- [Drag-and-Drop LLMs (DnD)](https://github.com/ryanzhangfan/DnD) - Base framework
- Qwen2.5 series - Target model architecture
