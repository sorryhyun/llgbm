# Suggestions: Aligning This Repo With DnD “Drag-and-Drop” Training

This document explains the likely reasons you’re seeing “all results are the same” in `phase_4_5_ablations copy.ipynb`, and proposes a concrete path to better match the DnD paper’s *prompt–checkpoint pairing* + *prompt embedding* training loop.

## 1) What the DnD paper is doing (in practice)

From the excerpt you pasted, the core training dataset for the parameter generator is:

- Split dataset prompts `P` into **non-overlapping prompt batches**: `p_1, …, p_I`
- Train the base LLM (or LoRA adapters) to obtain a set of **checkpoints** `m_1, …, m_J`
- Train the parameter generator using paired samples `{p_i, m_j}` (the text says “corresponding checkpoint”; typically this means the checkpoint was trained on `p_i`, i.e., `j=i` or a known mapping).

Then:

- Compute **prompt batch embedding** `c_i = Encoder(p_i, θ)` using a *pretrained text encoder* (often frozen).
- Feed `c_i` to the generator, supervise the generator with the checkpoint parameters (and optionally other signals).

The important details are:

1) The input is a **batch of prompts**, not a single prompt string.
2) The input representation is a **pretrained encoder embedding**, not a randomly initialized embedding trained from scratch (unless explicitly ablated).
3) The supervision is **checkpoint parameters** (e.g., LoRA adapter weights), so a “weight-only” baseline is meaningful.

## 2) What the current code does (and why it diverges)

### 2.1 The “weight” objective is effectively disabled

In `llgbm/training.py`, weight supervision is only used if the batch contains `tokens`:

- `tokens_teacher = batch.get("tokens")` (`llgbm/training.py:355`)
- otherwise `loss_weight` becomes 0 and `lambda_weight` does nothing (`llgbm/training.py:179`–`193`)

But the ablation dataset (`RealAdapterDataset`) returns `lora_weights`, not `tokens`:

- `llgbm/dataset.py:426`–`433`

So **MultiTaskLoss never sees teacher tokens**, and the “weight-only” configuration (`lambda_delta=0`) collapses to **zero loss / no learning**.

This alone makes “multitask / delta-only / weight-only” much closer than intended.

### 2.2 Conditioning is “single prompt, first item” (low diversity)

`RealAdapterDataset.__getitem__` picks **one** text condition:

- First probe from `probes_file`, else first prompt from `prompts.json`, else adapter name (`llgbm/dataset.py:401`–`414`)

This is not “prompt batches `p_i`”. It is “one representative string per adapter”, and it’s always the *first* entry. That reduces the variation the generator sees and can easily encourage collapse to nearly constant behavior.

### 2.3 Prompt embedding is not a pretrained encoder embedding

The generator consumes token IDs and uses its own randomly initialized embedding:

- `nn.Embedding(50000, 256)` + small transformer encoder (`llgbm/generator.py:24`–`41`)

This is *not* what Eq. (3) describes (`Encoder(p_i, θ)`), and it can be much weaker than using a pretrained encoder or sentence embedding model.

### 2.4 Evaluation “uses the same condition per task” (hides differences)

In the notebook’s extra evaluation cell, for each task it always uses:

- `sample = dataset[task_indices[0]]` then uses that `condition_ids` / `attention_mask`

So within a task you are effectively evaluating: “how good is the LoRA generated from *one fixed condition*”.

If the generator is insensitive to conditioning (or LoRA application is ineffective), all configs will look identical.

## 3) How to match the paper more closely (recommended design)

### 3.1 Make the training sample be “prompt batch ↔ checkpoint”

For each teacher adapter/checkpoint `m_j` (e.g., `arc_e_003`), you should store the corresponding prompt batch `p_i` that produced it.

Concretely, each training example should include:

- `prompts`: list[str] (size `K`, the prompt batch)
- `adapter_weights`: dict[str, Tensor] for teacher LoRA weights (or a tokenized representation)
- optionally `delta_teacher`: np.ndarray or Tensor for delta embedding (if you want behavioral supervision too)

**Important**: if you already have `prompts.json` per adapter, that’s your natural `p_i` (but it must be a *batch*, not just the first element).

Suggested minimal change:

- Instead of `text = prompts[0]`, keep `prompts = prompts[:K]` (or random sample `K` each epoch)
- Embed and pool those prompts to get `c_i`

### 3.2 Implement prompt embedding `c_i = Encoder(p_i, θ)` (frozen encoder)

Pick one default encoder and keep it frozen initially:

- Option A (simple, HF only): `bert-base-uncased` (take `[CLS]` or mean pooling of last hidden state)
- Option B (usually stronger): Sentence-Transformers (e.g., `all-MiniLM-L6-v2`)

Then pool across the prompt batch:

- `e_k = Encoder(prompt_k)` for each prompt in the batch
- `c_i = mean_k(e_k)` (or attention pooling)

**Practical tip**: precompute and cache `c_i` per adapter on disk (numpy) to make training fast and deterministic.

### 3.3 Fix weight supervision (so `lambda_weight` means something)

You have two good options:

**Option 1: supervise directly in weight space**

- Generator outputs LoRA A/B matrices per module (already the case).
- Dataset loads teacher LoRA A/B matrices from `adapter_model.safetensors`.
- Compute MSE between predicted and teacher weights over matched keys.

This is closest to “checkpoint supervision” conceptually and avoids needing a “tokenizer”.

**Option 2: use DnD’s LoRA tokenizer**

The DnD repo in this workspace includes a tokenizer implementation:

- `dnd_repo/workspace/dnd/tokenizer/tokenizer.py`

You can tokenize the teacher adapter into “LoRA tokens” and have the generator output tokens.

This matches the paper more literally (if they supervise tokenized parameters), but it’s more moving parts. I’d start with Option 1.

### 3.4 Keep delta supervision, but ensure it’s consistent

If you keep `delta_teacher`:

- Make sure `delta_teacher` was computed using the *same probe set definition* as `delta_pred` in training.
- If the paper’s “prompt batch” is the semantic signal, consider computing deltas using probes derived from `p_i` (or a fixed probe set that is stable across samples, but then store that choice clearly).

Right now probes come from manifest/prompt files in a way that’s not guaranteed to match how deltas were created.

### 3.5 Training loop sampling should be “random pair”

In practice your DataLoader already randomizes samples (`shuffle=True`), so once each sample is a paired `(p_i, m_j)` you’re close to Eq. (2).

What matters is that your dataset item is truly the paired data, not “task name → first prompt”.

## 4) Concrete implementation plan (minimal first pass)

### Step A — Upgrade the dataset to return prompt batches + teacher LoRA weights

Modify/replace `RealAdapterDataset` so `__getitem__` returns:

- `prompts: List[str]` (the prompt batch)
- `teacher_lora: Dict[str, Tensor]` (from `adapter_model.safetensors`)
- `delta_teacher: Tensor` (if available)

Avoid returning only `condition_ids` at this stage. Token IDs are a downstream detail; the paper’s “embedding” is the real input.

### Step B — Add a text encoder module and cache embeddings

Add a small component that:

- takes `prompts: List[str]`
- returns `c_i: Tensor[d_embed]`

Then optionally:

- cache per-adapter embeddings under something like `embeddings/<adapter_name>.npy`

### Step C — Update the generator to accept `c_i` (embedding input)

Simplest generator:

- `MLP(c_i) -> projection embeddings -> decode LoRA A/B`

This keeps your existing “decode A/B per projection” structure but removes the from-scratch token embedding.

### Step D — Implement real weight loss

In `train_step`, compute `loss_weight` by comparing predicted LoRA weights to `teacher_lora`:

- restrict to keys that exist in both dicts
- ensure shapes match; if not, log and skip those keys

Then `lambda_weight` will actually affect optimization.

### Step E — Update the notebook evaluation

To avoid misleading “everything is same”:

- evaluate multiple random conditions per task (not just `task_indices[0]`)
- log whether LoRA hooks match keys (`FunctionalLoRA.debug_key_matching`)
- verify “LoRA actually changes logits” by comparing base vs LoRA on a single prompt

## 5) Debugging checklist (quick sanity tests)

If accuracies still look identical after fixing dataset + weight loss, check:

1) **Hook matching**: are LoRA keys matching the base model modules?
   - `FunctionalLoRA.debug_key_matching` (`llgbm/functional.py:161`)

2) **LoRA effect size**: do generated weights meaningfully change outputs?
   - Compare logits/base loss with and without generated LoRA on one prompt.

3) **Condition sensitivity**: does the generator output change across different prompt batches?
   - Compare L2 distance between generated LoRA weights for different `p_i`.

4) **Delta supervision consistency**: are `delta_teacher` and `delta_pred` computed with consistent probes?

## 6) Why prompt–checkpoint pairing matters for “drag-and-drop”

If the generator sees prompt batches that *really correspond* to the checkpoint it’s trying to imitate, it can learn:

- “these prompts → those LoRA parameters”

But if:

- the prompts are not representative (only first item),
- embeddings are weak (random embedding from scratch),
- supervision is missing (no weight loss),

then the easiest solution is often to produce a near-constant adapter that gives baseline performance, which looks like “all configs are the same”.

---

If you want, I can implement the minimal first pass (dataset returns prompt batches + teacher LoRA weights, add weight loss, switch generator input to pretrained encoder embeddings) and update the ablation notebook accordingly.
