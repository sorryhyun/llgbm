# Learning LoRA Generator by Behavioral Matching (Qwen2.5-1.5B) — Implementation Plan

## Goal
Train a DnD-style prompt-conditioned LoRA generator, but supervise it with **behavioral signals**: match **delta activations** (base vs adapted) rather than (or in addition to) matching LoRA weights by MSE.

Concretely, for a probe set \(P\), define a delta embedding:
\[
v(M) = \mathbb{E}_{p\in P}\left[h^{(L)}_{M}(p,\text{last token})\right], \quad
\Delta(M, B) = v(M) - v(B)
\]
where \(B\) is the base model and \(M\) is an adapted model (teacher finetune, teacher LoRA, or generated LoRA).

Train a generator \(G(\text{condition}) \rightarrow \hat{\theta}_{\text{LoRA}}\) so that:
\[
\Delta(B+\hat{\theta}_{\text{LoRA}}, B) \approx \Delta(M_{\text{teacher}}, B)
\]

We start with **Qwen2.5-1.5B** and use **dataset-dependent conditioning**:
- *Prompts-only* datasets: use prompts as condition.
- *\(x,y\) behavior* datasets: use prompt+answer (or prompt+desired-output) as condition.

## What We Reuse
- From `Drag-and-Drop-LLMs`:
  - Hypernetwork generator: `workspace/dnd/model/decoderonly.py` (`HyperConvDecoderModel_*`)
  - Tokenization/detokenization of LoRA checkpoints: `workspace/dnd/tokenizer/tokenizer.py`
  - Dataset plumbing and condition variants:
    - prompts-only: `Text2Qwen25LoRA_FullCondDataset` (`workspace/dnd/dataset/register.py`)
    - prompt+answer: `Text2Qwen25LoRA_CondQ_ADataset` (`workspace/dnd/dataset/register.py`)
- From `delta_activations`:
  - Probe templates and embedding definition:
    - probes: `delta_activations/delta_activations.py#create_generic_probes`
    - last-layer/last-token hidden state: `delta_activations/delta_activations.py#get_average_activation`

## MVP Scope (First Milestone)
**Target:** Add delta-activation supervision to an existing DnD Qwen2.5-1.5B training setup, initially as a **multi-task loss**:
\[
\mathcal{L} = \mathcal{L}_{\text{weights}} + \lambda \mathcal{L}_{\Delta}
\]
where:
- \(\mathcal{L}_{\text{weights}}\): DnD’s existing MSE over tokenized LoRA weights.
- \(\mathcal{L}_{\Delta}\): MSE (or cosine) between teacher and generated delta embeddings.

We defer “delta-only” training and retrieval/merge until the multi-loss pipeline is stable and measurable.

---

## Phase 0 — Baseline Reproduction (DnD on Qwen2.5-1.5B)
### Tasks
1. Identify the existing DnD scripts for Qwen2.5-1.5B training and generation (e.g., `workspace/main/tasks/*/train_qwen1.5lora_*.py`).
2. Run a small sanity training (few hundred steps) to ensure:
   - dataset loading works (`./data/...`, `./prepare/data/...`)
   - LoRA checkpoint tokenization shapes match model config
   - generator loss decreases and checkpoint saving works

### Deliverables
- A known-working baseline run config for Qwen2.5-1.5B (domain chosen; can be a subset).
- A reference generated LoRA adapter written to `workspace/datasets/.../` by the existing pipeline.

### Acceptance Criteria
- Training loop runs end-to-end, produces `.safetensors` outputs, and does not OOM on your hardware.

---

## Phase 1 — Define Delta Targets (Teacher Signals) Offline

### 1.1 Delta embedding spec (initial)
- Probe set \(P\): start with the 5 “generic probes” in `delta_activations/delta_activations.py#create_generic_probes`.
- Representation: `hidden_states[-1][:, -1, :]` (last layer, last token), float32.
- Aggregation: mean over probes.
- Delta: adapted mean − base mean.
- Optional later extensions:
  - multi-layer concatenation or averaging (improves identifiability)
  - multi-token pooling
  - projection to fixed dimension (if comparing across backbones later)

### 1.2 Decide what the “teacher” is (two supported modes)
**Mode A (recommended MVP): teacher = per-dataset LoRA checkpoint**
- Teacher embedding: \(\Delta(B+\theta^*_{\text{LoRA}}, B)\)
- Pros: already aligned with DnD’s data; much cheaper than full fine-tunes.

**Mode B (future): teacher = fully finetuned model**
- Teacher embedding: \(\Delta(M^*_{\text{FT}}, B)\)
- Pros: closest to your original README framing.
- Cons: heavier to store/compute; requires full finetune checkpoints.

### 1.3 Implement offline delta computation + caching
Write a script that:
1. Loads base Qwen2.5-1.5B once.
2. Pre-tokenizes the probe set once.
3. Iterates over teacher checkpoints (LoRA adapters):
   - Loads the adapter onto base (PEFT).
   - Computes \(v(B+\theta^*)\).
   - Computes and saves \(\Delta^* = v(B+\theta^*) - v(B)\).
4. Writes a manifest mapping `checkpoint_path -> delta_path` (or packs into one `.npz` keyed by checkpoint id).

### Deliverables
- `deltas/` cache containing teacher deltas for the training pool.
- A manifest/index file to look up deltas by checkpoint path.
- Logged stats: delta norms, distribution, any outliers/broken adapters.

### Acceptance Criteria
- Can compute deltas for N adapters without leaking GPU memory and with deterministic output (same seed).

---

## Phase 2 — Dataset Plumbing: Return Delta Labels Alongside Weight Tokens

### Tasks
1. Add a dataset wrapper (or new dataset class) that returns:
   - `tokens_teacher` (existing)
   - `condition` (existing; prompts-only or prompt+answer depending on dataset)
   - `delta_teacher` (new; loaded from manifest/cache)
2. Ensure the dataloader collate keeps shapes consistent:
   - `delta_teacher`: `[batch, hidden_size]` (float32/bfloat16 depending on loss stability)

### Deliverables
- New dataset class (or wrapper) used in one training script.

### Acceptance Criteria
- A single training batch yields `(tokens, cond, delta)` and matches expected shapes/dtypes.

---

## Phase 3 — Differentiable Delta Computation for Generated LoRA (Core Engineering)

This is the hard part: compute \(\Delta(B+\hat{\theta}, B)\) **with gradients flowing back into the generator**.

### 3.1 Requirements
- Base model weights are frozen (no grads).
- Generated LoRA weights are differentiable outputs of the generator.
- Forward pass must incorporate the generated LoRA updates without `.copy_()`, `.load_state_dict()`, or save/reload, which would break gradients.

### 3.2 Preferred approach: `torch.func.functional_call` with a PEFT-instrumented base model
1. Instantiate a PEFT LoRA-wrapped base model (same LoRA config as the teacher adapters) once.
2. On each training step:
   - Generator outputs `tokens_pred`.
   - Detokenize to a LoRA weight dict `lora_pred` (A/B matrices) in-memory.
   - Build a `params` mapping that replaces only the LoRA parameters in the PEFT model.
   - Run `functional_call(peft_model, params, (probe_inputs,))` to compute `v_pred`.
3. Compute `delta_pred = v_pred - v_base_cached`.
4. Loss: `L_delta(delta_pred, delta_teacher)`; backprop reaches generator through `lora_pred`.

### 3.3 Backup approach: custom functional LoRA injection
If PEFT parameter naming or `functional_call` proves brittle:
- Build a wrapper that intercepts each target `nn.Linear` and computes:
  \[
  y = xW^T + s \cdot (x A^T) B^T
  \]
  with `A,B` provided from the generator outputs.
- Keep dropout=0 and a fixed scaling `s` to match training adapters.

### 3.4 Cost controls (must-have)
- Compute delta loss on a small microbatch each step (e.g., 1–4 items from the batch).
- Use the 5-probe set initially; shorten max length (e.g., 128–256).
- Cache `v_base` once.
- Mixed precision for the probe forward; keep the final delta in float32 for stability if needed.

### Deliverables
- A module/function: `compute_delta_embedding(base_model, generated_lora, probes) -> delta_pred` (differentiable).

### Acceptance Criteria
- A unit “gradient sanity” check: changing generator outputs changes `delta_pred`, and `delta_loss.backward()` produces non-zero grads in generator parameters.

---

## Phase 4 — Multi-Task Training: Weight MSE + Delta Loss

### Tasks
1. Integrate delta loss into one Qwen2.5-1.5B DnD training script.
2. Add a configurable `lambda_delta` schedule:
   - start at 0 (stabilize weight regression)
   - ramp to target value over N steps
3. Choose delta loss:
   - Start with MSE on deltas.
   - Try cosine distance if scale varies too much.
4. Logging:
   - `L_weight`, `L_delta`, delta cosine similarity, delta norms.

### Key ablations (minimum set)
- `lambda_delta`: {0, small, medium, large}
- probe count: {5, 10/20} (if you add more templates)
- representation choice:
  - last-layer last-token (default)
  - last-layer mean-pool tokens (optional)
  - multi-layer concat/avg (optional)

### Deliverables
- Trained checkpoints for baseline DnD and delta-augmented DnD.
- A simple evaluation script that compares them on a held-out dataset/domain.

### Acceptance Criteria
- Delta-augmented training runs stably.
- Shows at least one measurable improvement signal on held-out tasks (task metric or better delta alignment correlating with better task performance).

---

## Phase 5 — Delta-Only Training (After Multi-Loss Works)

### Variant B1: Generator still outputs LoRA weights
Train with only:
\[
\min_G \; \lVert \Delta(B+\hat{\theta},B) - \Delta^* \rVert
\]
Add regularizers to avoid degenerate solutions:
- LoRA norm penalty / weight decay on \(\hat{\theta}\)
- optional KL tether on probe logits (keep behavior consistent beyond hidden state)

### Variant B2: Predict delta embedding only, retrieve/merge adapters
1. Train a small `g(condition) -> delta_hat`.
2. At inference:
   - retrieve top-k teacher adapters by cosine similarity in delta space
   - merge adapters (weighted avg, then TIES merge as a stronger baseline)

### Acceptance Criteria
- Delta-only-to-LoRA can produce non-trivial adapters (beats base model on at least one domain).
- Retrieval baseline is competitive and provides a “floor” for delta-only learning.

---

## Phase 6 — Compositionality + “Behavioral Algebra” Tests
Motivation: Delta Activations suggests deltas are approximately additive for mixed data.

### Tasks
1. Build mixed-task conditions (dataset unions or prompt mixtures).
2. Compare:
   - teacher LoRA trained on mix (if available)
   - generated LoRA from mixed condition
   - delta-sum heuristic: `delta_hat_mix = delta_hat_1 + delta_hat_2` (B2 path)
   - merged adapters: avg vs TIES

### Acceptance Criteria
- `delta_mix` closer to `delta_1 + delta_2` than to either component alone (at least qualitatively).
- Mixed adapters do not catastrophically interfere compared to naive merges.

---

## Phase 7 — Packaging, Reproducibility, and Scaling
### Engineering tasks
- Centralize configuration (model id, probe set, lambda schedule, dataset conditioning form).
- Cache management: version deltas by (base model hash, probe set hash, adapter id).
- Make runs reproducible: fixed random seeds, deterministic probe ordering.

### Scaling steps
- Increase probe diversity (but keep cost controlled).
- Expand to multiple domains (math/coding/commonsense) within Qwen2.5-1.5B.
- Only after stable: consider cross-backbone “Delta Meaning”-style targets.

---

## Risks & Mitigations
- **Gradient breaks when applying generated adapters**
  - Mitigation: enforce functional application (`functional_call` or custom functional LoRA).
- **Under-constraint / “internal match but output mismatch”**
  - Mitigation: keep multi-loss; optionally add logit KL on probes.
- **Probe sensitivity / identifiability**
  - Mitigation: probe ablations; add a few domain-neutral but diverse probes; consider multi-layer deltas.
- **Compute blow-up**
  - Mitigation: microbatch delta loss; cache base activations; short probes; mixed precision; fewer target modules early.
- **Adapter key/name mismatches between DnD tokenizer and PEFT model**
  - Mitigation: implement a single canonical mapping layer from detokenized dict keys → model param names; validate on one adapter end-to-end.

---

## Concrete Next Actions (for Qwen2.5-1.5B start)
1. Pick the first target domain/dataset (e.g., math or coding) and decide conditioning form:
   - prompts-only: `Text2Qwen25LoRA_FullCondDataset`
   - prompt+answer: `Text2Qwen25LoRA_CondQ_ADataset`
2. Run a short baseline DnD training to confirm the environment.
3. Implement offline delta caching for the same teacher adapters.
4. Add delta labels to the dataset and integrate multi-loss training.
