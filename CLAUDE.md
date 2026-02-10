# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

## Common Commands

**Train teacher LoRA adapters:**
```bash
python train_lora_adapters.py \
    --data_dir data \
    --output_dir checkpoints \
    --tasks arc_e boolq gsm8k \
    --lora_rank 8
```

**Run DnD training (multi-GPU):**
```bash
./dnd_repo/scripts/launch_multi.sh workspace/main/tasks/math/train_qwen1.5lora_Math.py 4
```

**Task-specific DnD workflows:**
```bash
./dnd_repo/scripts/common_sense_reasoning/ARC-e/training_and_generation.sh
./dnd_repo/scripts/math1.5B/training_and_generation.sh
```

## Package Structure

**`llgbm/`** - Core module with these submodules:

| Module | Key Exports |
|--------|-------------|
| `probes.py` | `create_generic_probes()`, `create_domain_probes(domain)` |
| `delta.py` | `compute_base_activation()`, `compute_adapter_delta()`, `DeltaCache` |
| `dataset.py` | `DeltaAugmentedDataset`, `Text2Qwen25LoRA_DeltaDataset`, `RealAdapterDataset`, `create_dataloader()` |
| `functional.py` | `FunctionalLoRA`, `compute_delta_differentiable()` |
| `losses.py` | `WeightLoss`, `DeltaWLoss`, `MultiTaskLoss`, `DeltaOnlyLoss`, `DeltaGuidedLoss` |
| `training.py` | `TrainingConfig`, `train()`, `evaluate()` |
| `text_encoder.py` | `PretrainedTextEncoder`, `EmbeddingCache` |
| `generator.py` | `LoRAGenerator`, `create_generator()` |
| `evaluation.py` | `compute_accuracy_with_lora()`, `evaluate_all_tasks()` |
| `ablations.py` | `AblationConfig`, `run_ablations()` |

**`dnd_repo/`** - Submodule with base DnD framework:
- `workspace/dnd/model/decoderonly.py`: HyperConv generator models
- `workspace/dnd/tokenizer/register.py`: `Qwen2515LoRA_Tokenizer2D`

## Architecture: Generator Pipeline

```
Text Prompts
    ↓
[text_encoder] → condition embeddings (384-dim)
    ↓
[generator] → LoRA weights (A, B matrices per layer)
    ↓
[functional] → Apply LoRA differentiably (hooks mode)
    ↓
[delta] → Compute Δ_pred via probe texts
    ↓
[training] → Loss = λ_w * L_weight + λ_d * L_delta
```

## Notebooks

| Notebook | Focus |
|----------|-------|
| `toy_baseline.ipynb` | Weight-only baseline (raw A, B MSE) |
| `toy_delta_w.ipynb` | Delta-W supervision (MSE on B@A*scaling, gauge-invariant) |
| `phase_4_multitask.ipynb` | Multi-task training (weights + deltas) |
| `phase_4_5_ablations.ipynb` | Ablation studies (3 trials × N configs) |
| `phase_5_delta_only.ipynb` | Delta-only training (behavioral supervision only) |
| `train_lora_adapters.ipynb` | Train teacher LoRA adapters |

## Key Abstractions

**FunctionalLoRA**: Applies LoRA weights to base model without in-place modification, enabling gradient flow from delta loss back to generator. Uses hooks mode (default, memory-efficient) or `torch.func.functional_call` (legacy).

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

## Data Layout

```
checkpoints/
├── {task}/              # arc_e, boolq, gsm8k
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
├── GSM8K_train.json
└── HellaSwag_train.json
```
