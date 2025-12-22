# Learning LoRA Generator by Behavioral Matching

Idea is simple.
Instead of learning lora generator by minimizing MSE between lora adapter and generated one, let's minimize gap of delta activations, which is delta between base model's last layer representation and fine-tune models'.

Inference: (x, y) -> [lora generator] -> (lora adapter), where y is desired output.
Train: (x, y) -> [lora generator] -> (lora adapter) -> [inference] ->
                       (delta of representation {base model, lora adapted model}) <-> (delta of representation {base model, finetuned model})

Here’s a concrete experimental plan + a clean set of research questions for the two options you described (multi-task loss vs “delta-only”), grounded in what DnD and Delta Activations emphasize.

## Research questions (RQs)

**RQ1 — Does delta supervision improve zero-shot adaptation quality?**
Compared to DnD’s prompt→LoRA weight generation , does adding a delta-activation objective improve downstream task metrics on *unseen datasets* (and reduce bad “wrong adapters”)?

**RQ2 — Is weight supervision actually necessary?**
If you train *only* from delta-activation matching, can you reach (or beat) DnD weight-regression performance? If not, what failure mode shows up: instability, underconstraint, or mode collapse?

**RQ3 — What probe design is required for delta-based training to be identifiable?**
Delta Activations shows probe prompt choices matter (multiple prompts help; 5→20 gives little gain; domain-specific probes can suppress specialization) . How sensitive is your method to:

* #probe prompts
* prompt “genericness”
* which layers/tokens are used (they note last-token last-layer is a common choice) 

**RQ4 — Does delta supervision improve cross-domain robustness?**
Delta Activations separates domains much better than weight embeddings or sentence embeddings . Does that translate into better robustness to prompt phrasing changes / distribution shifts?

**RQ5 — Can you get compositional adapters via compositional deltas?**
Delta Activations exhibits an additive property: deltas for mixed data look closer to the sum of component deltas than to either component alone . Can you exploit this to build “oracle-ish” adapters for mixed tasks?

**RQ6 — What is the best merge rule when you do retrieval/mixture?**
If you retrieve k nearest training adapters (by delta space) and merge them, do you need interference-aware merging like TIES (Trim→Elect sign→Disjoint merge) ? How quickly do sign conflicts bite as k grows? 

**RQ7 — Cross-backbone transfer:**
Delta activations aren’t naturally comparable across different architectures, but Delta Meaning is compact and architecture-agnostic . Does swapping the delta target (Activations→Meaning) let one predictor generalize across backbones (matching DnD’s “smooth transfer” claim)? 

---

## Experimental plan

### Phase 0: Reproduce a strong baseline

1. **Pick a DnD-like setup**: base LLM(s), LoRA config, and the “handful of unlabeled prompts” conditioning scheme (text encoder → condition embedding → generator outputs LoRA for all layers) .
2. **Train per-task LoRAs (w^*)** on your training tasks/domains (these are your teacher adapters).

Deliverable: baseline DnD numbers + oracle LoRA numbers.

---

### Phase 1: Build delta targets once (teacher signals)

For each training task adapter (w^*):

1. Choose a **probe set** (D_{\text{probe}}) (try 5 and 20 prompts, and a “generic instruction template” vs domain-specific probes, since probe content affects distinguishability) .
2. Compute delta embeddings:

* **Delta Activations** (same-backbone): aggregate projected activation deltas; the paper uses 4096-dim for Delta Activations .
* Optionally **Delta Meaning** (cross-backbone): compact + architecture-agnostic .

Deliverable: dataset of pairs ((c, w^*, v^*)) where (c)=task sentence embedding, (v^*)=delta embedding.

---

### Phase 2: Option A (safer): Multi-task loss (weights + deltas)

Train the same parameter generator (f(c)\to \hat{w}) as DnD, but add delta loss:

* **Weight loss:** (\mathcal{L}_w = |\hat{w} - w^*|^2) (or cosine/MSE on flattened low-rank factors).
* **Delta loss:** run the base+generated-adapter model on (D_{\text{probe}}) and match delta embeddings:
  (\mathcal{L}_\Delta = |v(\hat{w}) - v(w^*)|^2).
* Total: (\mathcal{L} = \mathcal{L}*w + \lambda \mathcal{L}*\Delta) (+ optional KL on logits).

**Ablations (must-do):**

* sweep (\lambda) (0, small, medium, large)
* delta target type: Activations vs Meaning
* probe size/content (5 vs 20; generic vs domain prompts)

Success signal: improved unseen-task performance vs baseline DnD, without hurting in-domain.

---

### Phase 3: Option B (riskier): Delta-only learning

You have two clean variants—test both because they fail differently:

**B1: Delta-only generation (still outputs LoRA)**
Train (f(c)\to \hat{w}) with *no* direct weight supervision, only:
[
\min_f \ |v(\hat{w}) - v^*|^2
]
Add constraints to avoid degenerate solutions:

* LoRA norm penalty / rank regularization
* small KL/logit tether (to reduce “internal match, output mismatch”)

**B2: Predict delta embedding only, then retrieve/merge adapters**
Train small (g(c)\to \hat{v}). At inference:

* retrieve top-k training adapters by similarity in delta space (Delta Activations tends to cluster well) 
* merge them into an adapter using:

  * weighted average baseline
  * **TIES merging** (Trim→Elect sign→Disjoint merge) 
    and test sensitivity to k (because sign conflicts rise with more merges) .

Success signal: competitive performance with much smaller learned module (especially if retrieval+merge works).

---

### Phase 4: Compositionality and “oracle adapter” experiments

Use the additive property test as a *mechanistic* check:

1. Build mixed tasks (D_1\cup D_2).
2. Compare:

* finetune-on-mix LoRA (teacher)
* merge predicted adapters (avg vs TIES)
* “delta-sum” construction: (\hat{v}_{mix} = \hat{v}_1 + \hat{v}_2) (then either solve for adapter B1, or retrieve/merge B2)

Delta Activations suggests the mix embedding is closer to the sum than to either alone , so if your system is aligned, this should be one of your strongest qualitative wins.

---

## What I’d expect (so you can sanity-check quickly)

* Multi-task loss should be the most reliable: it keeps DnD’s strong supervision but pushes toward “functionally correct” adapters.
* Delta-only-to-LoRA (B1) is the most likely to underconstrain unless probe design is excellent.
* Delta→retrieve→TIES (B2) is surprisingly strong when delta embeddings are clean, and it naturally leverages interference-aware merging .

