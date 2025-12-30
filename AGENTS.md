# AGENTS.md


## Project Overview

LLGBM (Learning LoRA Generator by Behavioral Matching) extends the Drag-and-Drop LLMs (DnD) framework with behavioral supervision. Instead of only matching LoRA weights via MSE, the generator is trained to match **delta activations**—the difference in hidden states between base and adapted models.

Core equation:
```
Loss = λ_w * L_weight(predicted_weights, teacher_weights) + λ_d * L_delta(Δ_pred, Δ_teacher)
```

Where `Δ(M, B) = v(M) - v(B)` is computed as the average last-layer, last-token hidden state over probe texts.

## Development Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -r dnd_repo/requirements.txt
uv pip install peft scikit-learn matplotlib seaborn
```

Python 3.12+. Main dependencies: transformers 4.49.0, torch 2.5.1, peft, safetensors, accelerate.

## Package Structure

**`llgbm/`** - Core module with these submodules:

| Module | Key Exports |
|--------|-------------|
| `probes.py` | `create_generic_probes()`, `create_domain_probes(domain)` |
| `delta.py` | `compute_base_activation()`, `compute_adapter_delta()`, `DeltaCache` |
| `dataset.py` | `DeltaAugmentedDataset`, `Text2Qwen25LoRA_DeltaDataset`, `create_dataloader()` |
| `functional.py` | `FunctionalLoRA`, `compute_delta_differentiable()` |
| `training.py` | `TrainingConfig`, `MultiTaskLoss`, `DeltaOnlyLoss`, `train()`, `evaluate()` |

**`dnd_repo/`** - Submodule with base DnD framework:
- `workspace/dnd/model/decoderonly.py`: HyperConv generator models
- `workspace/dnd/tokenizer/register.py`: `Qwen2515LoRA_Tokenizer2D`

## Notebooks

| Notebook | Phase | Focus |
|----------|-------|-------|
| `phase_0_baseline.ipynb` | 0 | DnD reproduction |
| `phase_1_delta.ipynb` | 1 | Delta computation + caching |
| `phase_2_dataset.ipynb` | 2 | Dataset with delta labels |
| `phase_3_differentiable.ipynb` | 3 | Differentiable delta via `torch.func.functional_call` |
| `phase_4_multitask.ipynb` | 4 | Multi-task training (weights + deltas) |
| `phase_4_5_ablations.ipynb` | 4.5 | Ablation studies (3 trials × 3 configs) |
| `phase_5_delta_only.ipynb` | 5 | Delta-only training (behavioral supervision only) |

## Key Abstractions

**FunctionalLoRA**: Applies LoRA weights to base model without in-place modification, enabling gradient flow from delta loss back to generator. Uses `torch.func.functional_call`.

**DeltaCache**: Persists embeddings to `deltas/` directory. Base activation computed once; per-adapter deltas stored as `.npy` files with manifest tracking.

**TrainingConfig**: Dataclass with all hyperparameters. Automatically configures model architecture (Qwen2.5-0.5B vs 1.5B) based on `use_small_model` flag.

## Model Configuration

**Qwen2.5-0.5B** (default for testing):
- 24 layers, hidden=896, intermediate=4864
- LoRA rank=8, alpha=16

**Qwen2.5-1.5B** (production):
- 28 layers, hidden=1536, intermediate=8960
- LoRA rank=16, alpha=32
- 7 LoRA targets per layer: q/k/v/o_proj, gate/up/down_proj

## Implementation Notes

**Tokenizer roundtrip**: Must maintain error < 1e-3. Skip NaN values (padding) when computing average error.

**Delta computation**: Uses last-layer, last-token hidden states averaged over probe texts. Cache base activation once and reuse.

**Memory management**: Force `gc.collect()` and `torch.cuda.empty_cache()` after each adapter delta computation.

**Gradient flow**: Phase 3+ uses `FunctionalLoRA.apply_lora_weights()` which returns effective params with gradient connection through the `delta = B @ A * scaling` computation.

## DnD Training Scripts

```bash
./dnd_repo/scripts/launch_multi.sh workspace/main/tasks/math/train_qwen1.5lora_Math.py 4
```
