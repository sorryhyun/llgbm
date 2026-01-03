# LLGBM: Learning LoRA Generator by Behavioral Matching

Train a prompt-conditioned LoRA generator supervised with **behavioral signals**: match **delta activations** (base vs adapted) rather than (or in addition to) matching LoRA weights by MSE.

## Core Idea

Instead of learning a LoRA generator by minimizing MSE between teacher LoRA adapters and generated ones, minimize the gap in delta activations—the difference between base model's last-layer representation and fine-tuned model's.

```
Inference: (x, y) -> [LoRA Generator] -> (LoRA adapter)

Training:  (x, y) -> [LoRA Generator] -> (LoRA adapter) -> [Inference] ->
                     delta(base, generated) <-> delta(base, teacher)
```

For a probe set P, define a delta embedding:

```
v(M) = E[h^(L)_M(p, last_token)] for p in P
Δ(M, B) = v(M) - v(B)
```

Train generator G(condition) → θ_LoRA so that: `Δ(B + θ_LoRA, B) ≈ Δ(M_teacher, B)`

---

## Research Questions

| RQ | Question |
|----|----------|
| **RQ1** | Does delta supervision improve zero-shot adaptation quality vs DnD weight-only? |
| **RQ2** | Is weight supervision necessary, or can delta-only training match/beat DnD? |
| **RQ3** | What probe design (count, genericness, layer/token choice) is needed for identifiability? |
| **RQ4** | Does delta supervision improve cross-domain robustness to prompt phrasing changes? |
| **RQ5** | Can compositional adapters be built via compositional deltas (additive property)? |
| **RQ6** | What merge rule works best for retrieval/mixture (avg vs TIES)? |
| **RQ7** | Does Delta Meaning enable cross-backbone transfer? |

---

## Implementation Phases

| Phase | Description | Status |
|-------|-------------|--------|
| **0** | Baseline DnD reproduction on Qwen2.5 | ✅ `phase_0_baseline.ipynb` |
| **1** | Offline delta computation & caching | ✅ `phase_1_delta.ipynb` |
| **2** | Dataset plumbing (return delta labels) | ✅ `phase_2_dataset.ipynb` |
| **3** | Differentiable delta via `functional_call` | ✅ `phase_3_differentiable.ipynb` |
| **4** | Multi-task training (weights + deltas) | ✅ `phase_4_multitask.ipynb` |
| **4.5** | Ablation studies | ✅ `phase_4_5_ablations.ipynb` |
| **5** | Delta-only training | ✅ `phase_5_delta_only.ipynb` |
| **6** | Compositionality & behavioral algebra | 📋 `phase_6.md` |
| **7** | Packaging, reproducibility, scaling | 📋 `phase_7.md` |

---

## Quick Start

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install -r dnd_repo/requirements.txt
uv pip install peft scikit-learn matplotlib seaborn

# Train teacher LoRA adapters
python train_lora_adapters.py --tasks arc_e boolq gsm8k --lora_rank 8

# Run training (see notebooks for interactive workflow)
```

---

## Package Structure

```
llgbm/
├── probes.py        # Probe templates: create_generic_probes(), create_domain_probes()
├── delta.py         # Delta computation: compute_base_activation(), DeltaCache
├── dataset.py       # Datasets: DeltaAugmentedDataset, RealAdapterDataset
├── functional.py    # FunctionalLoRA: differentiable LoRA application
├── training.py      # TrainingConfig, MultiTaskLoss, train(), evaluate()
├── generator.py     # LoRAGenerator, create_generator()
├── text_encoder.py  # PretrainedTextEncoder, EmbeddingCache
├── evaluation.py    # Task evaluation utilities
└── ablations.py     # AblationConfig, run_ablations()
```

---

## Training Objectives

**Multi-task loss (Phase 4):**
```
L = λ_w * L_weight + λ_d * L_delta
```

**Delta-only loss (Phase 5):**
```
L = L_delta + regularization
```

---

## Known Issues & Solutions

### Weight supervision not working
If `lambda_weight` has no effect, ensure dataset returns `tokens` (tokenized LoRA weights), not just `lora_weights` dict. See `MultiTaskLoss` in `training.py`.

### All ablation results identical
Check:
1. **Hook matching**: Use `FunctionalLoRA.debug_key_matching()` to verify LoRA keys match base model
2. **LoRA effect**: Compare logits with/without generated LoRA on one prompt
3. **Condition sensitivity**: Check if generator output varies across different prompts

### Conditioning is weak
The DnD paper uses **prompt batches** embedded via a **pretrained encoder**, not single prompts with random embeddings. Use `PretrainedTextEncoder` with `sentence-transformers/all-MiniLM-L6-v2` or similar.

---

## References

- [Drag-and-Drop LLMs](https://github.com/...) - Base framework
- [Delta Activations](https://arxiv.org/...) - Behavioral matching theory
