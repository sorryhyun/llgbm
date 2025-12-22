# Phase 6 — Compositionality + "Behavioral Algebra" Tests

## Goal
Test whether delta embeddings exhibit compositional properties - specifically, whether task mixtures can be approximated by adding/combining delta embeddings.

## Prerequisites
- Phase 5 complete (delta-only training working)
- Multiple domain-specific teacher LoRAs available
- Evaluation tasks defined

## Hypothesis

The Delta Activations paper suggests that deltas are approximately additive:
```
Δ(task1 + task2) ≈ Δ(task1) + Δ(task2)
```

This would enable:
1. **Zero-shot task composition**: Combine known task deltas for new mixed tasks
2. **Interpretable task algebra**: Add/subtract behavioral directions
3. **Efficient multi-task adaptation**: No need to train on all combinations

## Implementation Steps

### Step 1: Create Compositionality Test Framework

Create `llgbm/compositionality.py`:

```python
"""Compositionality and behavioral algebra experiments."""
import sys
sys.path.insert(0, "Drag-and-Drop-LLMs")

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from safetensors.torch import load_file, save_file
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

from llgbm.delta import DeltaCache, compute_adapter_delta
from llgbm.probes import create_generic_probes
from llgbm.retrieval import AdapterMerger


class CompositionExperiment:
    """
    Framework for testing compositional properties of delta embeddings.
    """

    def __init__(
        self,
        delta_cache: DeltaCache,
        base_model_name: str = "Qwen/Qwen2.5-1.5B",
        device: torch.device = None,
    ):
        self.delta_cache = delta_cache
        self.base_model_name = base_model_name
        self.device = device or torch.device("cuda")
        self.probes = create_generic_probes()

        # Load all deltas
        self.deltas = delta_cache.get_all_deltas()

    def get_domain_deltas(self) -> Dict[str, np.ndarray]:
        """
        Group deltas by domain and compute average per domain.

        Returns:
            Dict mapping domain name to average delta
        """
        domain_deltas = {}

        for path, delta in self.deltas.items():
            # Extract domain from path
            domain = self._extract_domain(path)
            if domain not in domain_deltas:
                domain_deltas[domain] = []
            domain_deltas[domain].append(delta)

        # Average within each domain
        return {
            domain: np.mean(deltas, axis=0)
            for domain, deltas in domain_deltas.items()
        }

    def _extract_domain(self, path: str) -> str:
        """Extract domain name from adapter path."""
        path_parts = Path(path).parts
        for part in path_parts:
            if part in ["math", "code", "commonsense", "legal", "medical", "science"]:
                return part
        return "unknown"

    def test_additivity(
        self,
        domain1: str,
        domain2: str,
        mixed_adapter_path: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Test if delta(domain1) + delta(domain2) ≈ delta(mixed).

        Args:
            domain1: First domain name
            domain2: Second domain name
            mixed_adapter_path: Optional path to adapter trained on mixed data

        Returns:
            Metrics comparing summed vs actual mixed delta
        """
        domain_deltas = self.get_domain_deltas()

        delta1 = domain_deltas[domain1]
        delta2 = domain_deltas[domain2]

        # Predicted mixed delta (sum)
        delta_sum = delta1 + delta2

        # If we have actual mixed adapter, compute its delta
        if mixed_adapter_path:
            from llgbm.delta import compute_base_activation, compute_adapter_delta
            base_act, tokenizer = compute_base_activation(
                self.base_model_name, self.probes, self.device
            )
            delta_actual = compute_adapter_delta(
                self.base_model_name,
                mixed_adapter_path,
                self.probes,
                base_act,
                tokenizer,
                self.device,
            )
        else:
            # Use average of adapters trained on both domains as proxy
            mixed_deltas = [
                d for p, d in self.deltas.items()
                if domain1 in p or domain2 in p
            ]
            delta_actual = np.mean(mixed_deltas, axis=0)

        # Compute metrics
        cos_sim = cosine_similarity(
            delta_sum.reshape(1, -1),
            delta_actual.reshape(1, -1),
        )[0, 0]

        mse = np.mean((delta_sum - delta_actual) ** 2)

        # Compare to individual deltas
        cos_with_d1 = cosine_similarity(
            delta_actual.reshape(1, -1),
            delta1.reshape(1, -1),
        )[0, 0]
        cos_with_d2 = cosine_similarity(
            delta_actual.reshape(1, -1),
            delta2.reshape(1, -1),
        )[0, 0]

        return {
            "sum_vs_actual_cosine": cos_sim,
            "sum_vs_actual_mse": mse,
            "actual_vs_d1_cosine": cos_with_d1,
            "actual_vs_d2_cosine": cos_with_d2,
            "sum_norm": np.linalg.norm(delta_sum),
            "actual_norm": np.linalg.norm(delta_actual),
        }

    def test_subtraction(
        self,
        source_domain: str,
        subtract_domain: str,
    ) -> Dict[str, float]:
        """
        Test if delta(A) - delta(B) produces meaningful results.

        E.g., delta(math+code) - delta(code) ≈ delta(math)?

        Args:
            source_domain: Domain to start from
            subtract_domain: Domain to subtract

        Returns:
            Metrics for the subtraction experiment
        """
        domain_deltas = self.get_domain_deltas()

        delta_source = domain_deltas[source_domain]
        delta_subtract = domain_deltas[subtract_domain]

        # Compute difference
        delta_diff = delta_source - delta_subtract

        # Compare to all other domains
        similarities = {}
        for domain, delta in domain_deltas.items():
            if domain not in [source_domain, subtract_domain]:
                cos_sim = cosine_similarity(
                    delta_diff.reshape(1, -1),
                    delta.reshape(1, -1),
                )[0, 0]
                similarities[domain] = cos_sim

        return {
            "diff_norm": np.linalg.norm(delta_diff),
            "similarities_to_other_domains": similarities,
        }

    def test_interpolation(
        self,
        domain1: str,
        domain2: str,
        num_points: int = 5,
    ) -> List[Dict[str, float]]:
        """
        Test interpolation between two domain deltas.

        delta(alpha) = alpha * delta1 + (1-alpha) * delta2

        Args:
            domain1: First domain
            domain2: Second domain
            num_points: Number of interpolation points

        Returns:
            List of metrics for each interpolation point
        """
        domain_deltas = self.get_domain_deltas()

        delta1 = domain_deltas[domain1]
        delta2 = domain_deltas[domain2]

        alphas = np.linspace(0, 1, num_points)
        results = []

        for alpha in alphas:
            delta_interp = alpha * delta1 + (1 - alpha) * delta2

            # Measure similarity to both source domains
            cos_d1 = cosine_similarity(
                delta_interp.reshape(1, -1),
                delta1.reshape(1, -1),
            )[0, 0]
            cos_d2 = cosine_similarity(
                delta_interp.reshape(1, -1),
                delta2.reshape(1, -1),
            )[0, 0]

            results.append({
                "alpha": alpha,
                "cosine_to_domain1": cos_d1,
                "cosine_to_domain2": cos_d2,
                "norm": np.linalg.norm(delta_interp),
            })

        return results


class AdapterComposer:
    """
    Compose adapters based on delta algebra.
    """

    def __init__(
        self,
        delta_cache: DeltaCache,
        base_model_name: str = "Qwen/Qwen2.5-1.5B",
    ):
        self.delta_cache = delta_cache
        self.base_model_name = base_model_name
        self.merger = AdapterMerger()

    def compose_by_delta_sum(
        self,
        adapter_paths: List[str],
        weights: Optional[List[float]] = None,
        output_path: Optional[str] = None,
    ) -> Tuple[Dict[str, torch.Tensor], np.ndarray]:
        """
        Compose adapters by summing their weights (delta-inspired).

        Args:
            adapter_paths: Paths to adapters to combine
            weights: Optional weights for each adapter
            output_path: Optional path to save result

        Returns:
            Composed adapter weights and predicted delta
        """
        if weights is None:
            weights = [1.0] * len(adapter_paths)

        # Load adapters
        adapters = [self.merger.load_adapter(p) for p in adapter_paths]

        # Weighted sum of adapter weights
        composed = {}
        for key in adapters[0].keys():
            composed[key] = sum(
                w * adapter[key] for adapter, w in zip(adapters, weights)
            )

        # Predict composed delta
        deltas = self.delta_cache.get_all_deltas()
        delta_pred = sum(
            w * deltas[p] for p, w in zip(adapter_paths, weights)
        )

        if output_path:
            save_file(composed, output_path)

        return composed, delta_pred

    def compose_by_delta_target(
        self,
        target_delta: np.ndarray,
        top_k: int = 3,
        merge_strategy: str = "ties",
    ) -> Dict[str, torch.Tensor]:
        """
        Compose adapters to approximate a target delta.

        Finds adapters whose deltas sum to target, then merges them.

        Args:
            target_delta: Target delta embedding to approximate
            top_k: Number of adapters to use
            merge_strategy: How to merge adapters

        Returns:
            Composed adapter weights
        """
        from llgbm.retrieval import AdapterIndex

        # Build index
        index = AdapterIndex(self.delta_cache)

        # Find most similar adapters
        results = index.search(target_delta, k=top_k)

        # Load and merge
        adapters = [self.merger.load_adapter(p) for p, _ in results]
        weights = [score for _, score in results]

        if merge_strategy == "weighted_average":
            return self.merger.weighted_average(adapters, weights)
        elif merge_strategy == "ties":
            return self.merger.ties_merge(adapters, weights)
        else:
            raise ValueError(f"Unknown strategy: {merge_strategy}")
```

### Step 2: Create Evaluation for Composed Adapters

Create `llgbm/evaluation/composition_eval.py`:

```python
"""Evaluate composed adapters on downstream tasks."""
import torch
from typing import Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import numpy as np


class CompositionEvaluator:
    """
    Evaluate whether composed adapters work on mixed tasks.
    """

    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-1.5B",
        device: torch.device = None,
    ):
        self.base_model_name = base_model_name
        self.device = device or torch.device("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    def evaluate_adapter(
        self,
        adapter_weights: Dict[str, torch.Tensor],
        eval_data: List[Dict],
        max_length: int = 512,
    ) -> Dict[str, float]:
        """
        Evaluate an adapter on given evaluation data.

        Args:
            adapter_weights: LoRA adapter weights
            eval_data: List of {"input": str, "target": str} dicts
            max_length: Maximum generation length

        Returns:
            Evaluation metrics
        """
        # Load model with adapter
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map=self.device,
        )

        # Apply adapter weights directly
        # This requires matching the weight format to PEFT's expectations
        # For simplicity, save to temp file and load via PEFT
        import tempfile
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as tmpdir:
            save_file(adapter_weights, f"{tmpdir}/adapter_model.safetensors")

            # Create adapter config
            import json
            config = {
                "base_model_name_or_path": self.base_model_name,
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"],
                "lora_dropout": 0.0,
                "bias": "none",
                "task_type": "CAUSAL_LM",
            }
            with open(f"{tmpdir}/adapter_config.json", "w") as f:
                json.dump(config, f)

            model = PeftModel.from_pretrained(model, tmpdir)

        model.eval()

        # Evaluate
        correct = 0
        total = 0
        perplexities = []

        for item in tqdm(eval_data, desc="Evaluating"):
            input_text = item["input"]
            target = item["target"]

            # Generate
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            generated = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            # Simple exact match (customize per task)
            if target.lower().strip() in generated.lower():
                correct += 1
            total += 1

            # Compute perplexity on target
            target_inputs = self.tokenizer(
                input_text + target,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                loss = model(**target_inputs, labels=target_inputs["input_ids"]).loss
                perplexities.append(torch.exp(loss).item())

        return {
            "accuracy": correct / total if total > 0 else 0,
            "perplexity": np.mean(perplexities),
            "num_samples": total,
        }

    def compare_composition_methods(
        self,
        adapters: Dict[str, Dict[str, torch.Tensor]],
        eval_data: Dict[str, List[Dict]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare different composed adapters.

        Args:
            adapters: Dict mapping method name to adapter weights
            eval_data: Dict mapping domain to evaluation data

        Returns:
            Nested dict of results
        """
        results = {}

        for method_name, adapter in adapters.items():
            results[method_name] = {}

            for domain, data in eval_data.items():
                metrics = self.evaluate_adapter(adapter, data)
                results[method_name][domain] = metrics

        return results
```

### Step 3: Create Visualization Tools

Create `scripts/visualize_composition.py`:

```python
"""Visualize composition experiment results."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from llgbm.delta import DeltaCache
from llgbm.compositionality import CompositionExperiment


def visualize_additivity(experiment: CompositionExperiment, output_dir: str):
    """
    Visualize additivity test results across domain pairs.
    """
    domain_deltas = experiment.get_domain_deltas()
    domains = list(domain_deltas.keys())

    # Test all pairs
    results = np.zeros((len(domains), len(domains)))
    labels = []

    for i, d1 in enumerate(domains):
        for j, d2 in enumerate(domains):
            if i < j:
                metrics = experiment.test_additivity(d1, d2)
                results[i, j] = metrics["sum_vs_actual_cosine"]
                results[j, i] = metrics["sum_vs_actual_cosine"]

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        results,
        xticklabels=domains,
        yticklabels=domains,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
    )
    plt.title("Delta Additivity: cos(Δ₁+Δ₂, Δ_mixed)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/additivity_heatmap.png", dpi=150)
    plt.close()


def visualize_interpolation(experiment: CompositionExperiment, output_dir: str):
    """
    Visualize interpolation between domains.
    """
    domain_deltas = experiment.get_domain_deltas()
    domains = list(domain_deltas.keys())

    if len(domains) < 2:
        return

    # Pick two domains
    d1, d2 = domains[0], domains[1]
    results = experiment.test_interpolation(d1, d2, num_points=11)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    alphas = [r["alpha"] for r in results]
    cos_d1 = [r["cosine_to_domain1"] for r in results]
    cos_d2 = [r["cosine_to_domain2"] for r in results]
    norms = [r["norm"] for r in results]

    # Similarity plot
    axes[0].plot(alphas, cos_d1, "b-o", label=f"Similarity to {d1}")
    axes[0].plot(alphas, cos_d2, "r-o", label=f"Similarity to {d2}")
    axes[0].set_xlabel("Alpha (interpolation weight)")
    axes[0].set_ylabel("Cosine Similarity")
    axes[0].set_title(f"Delta Interpolation: {d1} ↔ {d2}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Norm plot
    axes[1].plot(alphas, norms, "g-o")
    axes[1].set_xlabel("Alpha")
    axes[1].set_ylabel("Delta Norm")
    axes[1].set_title("Interpolated Delta Norm")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/interpolation_{d1}_{d2}.png", dpi=150)
    plt.close()


def visualize_delta_space(experiment: CompositionExperiment, output_dir: str):
    """
    Visualize delta embedding space with arithmetic examples.
    """
    from sklearn.manifold import TSNE

    domain_deltas = experiment.get_domain_deltas()
    domains = list(domain_deltas.keys())

    if len(domains) < 3:
        return

    # Compute some composed deltas
    embeddings = []
    labels = []
    colors = []

    color_map = plt.cm.tab10

    for i, (domain, delta) in enumerate(domain_deltas.items()):
        embeddings.append(delta)
        labels.append(domain)
        colors.append(color_map(i))

    # Add some composed deltas
    for i in range(min(3, len(domains))):
        for j in range(i + 1, min(4, len(domains))):
            d1, d2 = domains[i], domains[j]
            composed = domain_deltas[d1] + domain_deltas[d2]
            embeddings.append(composed)
            labels.append(f"{d1}+{d2}")
            colors.append("gray")

    # t-SNE
    embeddings = np.stack(embeddings)
    tsne = TSNE(n_components=2, perplexity=min(5, len(embeddings) - 1), random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(12, 10))

    for i, (x, y) in enumerate(embeddings_2d):
        plt.scatter(x, y, c=[colors[i]], s=200 if "+" not in labels[i] else 100,
                   marker="o" if "+" not in labels[i] else "^")
        plt.annotate(labels[i], (x, y), fontsize=10)

    plt.title("Delta Embedding Space with Compositions")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/delta_space.png", dpi=150)
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="deltas")
    parser.add_argument("--output_dir", default="outputs/composition_viz")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    delta_cache = DeltaCache(args.cache_dir)
    experiment = CompositionExperiment(delta_cache)

    print("Visualizing additivity...")
    visualize_additivity(experiment, args.output_dir)

    print("Visualizing interpolation...")
    visualize_interpolation(experiment, args.output_dir)

    print("Visualizing delta space...")
    visualize_delta_space(experiment, args.output_dir)

    print(f"Saved visualizations to {args.output_dir}")


if __name__ == "__main__":
    main()
```

## File Structure After Phase 6

```
llgbm/
├── llgbm/
│   ├── compositionality.py           # Composition experiments
│   └── evaluation/
│       └── composition_eval.py       # Eval composed adapters
├── scripts/
│   └── visualize_composition.py      # Visualization
└── outputs/
    └── composition_viz/
        ├── additivity_heatmap.png
        ├── interpolation_*.png
        └── delta_space.png
```

## Key Experiments

| Experiment | What it Tests | Expected Result |
|------------|---------------|-----------------|
| Additivity | Δ₁ + Δ₂ ≈ Δ_mix | High cosine similarity |
| Subtraction | Δ_AB - Δ_B ≈ Δ_A | Should recover single domain |
| Interpolation | αΔ₁ + (1-α)Δ₂ | Smooth transition |
| Composition | Merge adapters by delta | Competitive with trained mix |

## Acceptance Criteria

- [ ] `delta_mix` closer to `delta_1 + delta_2` than to either component alone
- [ ] Mixed adapters do not catastrophically interfere
- [ ] Interpolation produces smooth behavioral transitions
- [ ] At least one composition method beats naive merging baselines

## Next Phase
Proceed to **Phase 7** for packaging, reproducibility, and scaling.
