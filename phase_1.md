# Phase 1 — Define Delta Targets (Teacher Signals) Offline

## Goal
Compute and cache delta embeddings for all teacher LoRA checkpoints. These will serve as supervision targets for behavioral matching.

## Prerequisites
- Phase 0 complete (baseline DnD working)
- Teacher LoRA checkpoints available in `data/teacher_checkpoints/`
- GPU with enough VRAM to load Qwen2.5-1.5B + LoRA adapter

## Delta Embedding Definition

For a probe set P, the delta embedding is:
```
v(M) = E[h^(L)_M(p, last_token)] for p in P
Delta(M, B) = v(M) - v(B)
```

Where:
- B = base Qwen2.5-1.5B model
- M = base + teacher LoRA adapter
- h^(L) = last layer hidden state
- last_token = final token position

## Implementation Steps

### Step 1: Create Probe Module

Create `llgbm/probes.py`:

```python
"""Probe templates for delta embedding computation."""
from typing import List

def create_generic_probes() -> List[str]:
    """
    Create the 5 generic probe templates from the Delta Activations paper.
    These are task-agnostic and designed to elicit general model behavior.
    """
    task = "Please provide a response."
    input_text = "Input."

    probe_templates = [
        f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{task} Input:{input_text}\n\n### Response:",
        f"The task described below requires a response that completes the request accurately.\n\n### Instruction:\n{task} Input:{input_text}\n\n### Response:",
        f"Below is a description of a task. Provide a response that aligns with the requirements.\n\n### Instruction:\n{task} Input:{input_text}\n\n### Response:",
        f"The following instruction outlines a task. Generate a response that meets the specified request.\n\n### Instruction:\n{task} Input:{input_text}\n\n### Response:",
        f"You are given an instruction and input. Write a response that completes the task as requested.\n\n### Instruction:\n{task} Input:{input_text}\n\n### Response:",
    ]

    return probe_templates


def create_domain_probes(domain: str) -> List[str]:
    """
    Create domain-specific probes for better signal.

    Args:
        domain: One of "math", "code", "commonsense"
    """
    if domain == "math":
        return [
            "Solve the following math problem step by step:\nQuestion: What is 2 + 2?\nAnswer:",
            "Calculate the result:\nProblem: Find the derivative of x^2\nSolution:",
            "Mathematical reasoning:\nGiven: A triangle has angles 30, 60, and 90 degrees.\nFind: The ratio of its sides.\nAnswer:",
            "Arithmetic:\n15 * 7 = ",
            "Word problem: If a train travels 60 miles per hour for 2 hours, how far does it travel?\nSolution:",
        ]
    elif domain == "code":
        return [
            "Write a Python function:\ndef factorial(n):\n    ",
            "Complete the code:\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    ",
            "Debug this code:\ndef add(a, b):\n    return a - b  # Bug: should be +\nFixed:",
            "Explain this code:\nfor i in range(10):\n    print(i)\nExplanation:",
            "Write a function to reverse a string:\ndef reverse_string(s):\n    ",
        ]
    elif domain == "commonsense":
        return [
            "Question: What happens when you drop an egg on the floor?\nAnswer:",
            "Complete the sentence: The sun rises in the ",
            "Common knowledge: Water freezes at what temperature in Celsius?\nAnswer:",
            "Reasoning: If all birds can fly, and a penguin is a bird, can a penguin fly?\nAnswer:",
            "What do people typically eat for breakfast?\nAnswer:",
        ]
    else:
        return create_generic_probes()
```

### Step 2: Create Delta Computation Module

Create `llgbm/delta.py`:

```python
"""Delta embedding computation for behavioral matching."""
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def get_average_activation(
    model: torch.nn.Module,
    texts: List[str],
    tokenizer,
    device: torch.device,
    max_length: int = 256,
    layer_idx: int = -1,  # -1 for last layer
) -> np.ndarray:
    """
    Compute the average activation from a model for given probe texts.

    Args:
        model: The model to analyze (must have output_hidden_states=True)
        texts: List of probe strings
        tokenizer: Model tokenizer
        device: Device to run inference on
        max_length: Maximum sequence length
        layer_idx: Which layer to extract (-1 = last layer)

    Returns:
        numpy.ndarray: Average activation vector of shape (hidden_size,)
    """
    model.eval()
    activations = []

    for text in tqdm(texts, desc="Computing activations", leave=False):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Extract hidden state from specified layer
        hidden = outputs.hidden_states[layer_idx].float()  # (1, seq_len, hidden_size)
        last_token_hidden = hidden[:, -1, :].squeeze(0)    # (hidden_size,)
        activations.append(last_token_hidden.cpu().numpy())

    # Average across all probes
    return np.mean(np.stack(activations), axis=0)


def compute_base_activation(
    base_model_name: str,
    probes: List[str],
    device: torch.device,
    max_length: int = 256,
    dtype: torch.dtype = torch.float16,
) -> Tuple[np.ndarray, AutoTokenizer]:
    """
    Compute activation for the base model (cached once).

    Returns:
        Tuple of (base_activation, tokenizer)
    """
    print(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.config.output_hidden_states = True

    base_activation = get_average_activation(model, probes, tokenizer, device, max_length)

    # Free memory
    del model
    torch.cuda.empty_cache()

    return base_activation, tokenizer


def compute_adapter_delta(
    base_model_name: str,
    adapter_path: str,
    probes: List[str],
    base_activation: np.ndarray,
    tokenizer,
    device: torch.device,
    max_length: int = 256,
    dtype: torch.dtype = torch.float16,
) -> np.ndarray:
    """
    Compute delta embedding for a single adapter.

    Args:
        base_model_name: HuggingFace model ID for base
        adapter_path: Path to the LoRA adapter
        probes: List of probe texts
        base_activation: Pre-computed base model activation
        tokenizer: Tokenizer (shared with base)
        device: Compute device
        max_length: Max sequence length for probes
        dtype: Model dtype

    Returns:
        Delta embedding (adapted - base) as numpy array
    """
    # Load base model with adapter
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.config.output_hidden_states = True
    model.eval()

    # Compute adapted activation
    adapted_activation = get_average_activation(model, probes, tokenizer, device, max_length)

    # Compute delta
    delta = adapted_activation - base_activation

    # Free memory
    del model
    torch.cuda.empty_cache()

    return delta


class DeltaCache:
    """
    Manages caching of delta embeddings.

    Cache structure:
        deltas/
        ├── manifest.json       # Maps adapter paths to delta files
        ├── base_activation.npy # Cached base activation
        └── {adapter_id}.npy    # Individual delta embeddings
    """

    def __init__(self, cache_dir: str = "deltas"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.manifest_path = self.cache_dir / "manifest.json"
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict:
        import json
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                return json.load(f)
        return {"adapters": {}, "config": {}}

    def _save_manifest(self):
        import json
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

    def _adapter_id(self, adapter_path: str) -> str:
        """Generate unique ID for adapter path."""
        import hashlib
        return hashlib.md5(adapter_path.encode()).hexdigest()[:12]

    def get_base_activation(self) -> Optional[np.ndarray]:
        """Load cached base activation if exists."""
        path = self.cache_dir / "base_activation.npy"
        if path.exists():
            return np.load(path)
        return None

    def save_base_activation(self, activation: np.ndarray, config: Dict):
        """Save base activation and config."""
        np.save(self.cache_dir / "base_activation.npy", activation)
        self.manifest["config"] = config
        self._save_manifest()

    def get_delta(self, adapter_path: str) -> Optional[np.ndarray]:
        """Load cached delta for adapter if exists."""
        adapter_id = self._adapter_id(adapter_path)
        if adapter_id in self.manifest["adapters"]:
            delta_path = self.cache_dir / f"{adapter_id}.npy"
            if delta_path.exists():
                return np.load(delta_path)
        return None

    def save_delta(self, adapter_path: str, delta: np.ndarray):
        """Save delta embedding for adapter."""
        adapter_id = self._adapter_id(adapter_path)
        delta_path = self.cache_dir / f"{adapter_id}.npy"
        np.save(delta_path, delta)
        self.manifest["adapters"][adapter_id] = {
            "path": adapter_path,
            "norm": float(np.linalg.norm(delta)),
            "shape": list(delta.shape),
        }
        self._save_manifest()

    def get_all_deltas(self) -> Dict[str, np.ndarray]:
        """Load all cached deltas."""
        deltas = {}
        for adapter_id, info in self.manifest["adapters"].items():
            delta_path = self.cache_dir / f"{adapter_id}.npy"
            if delta_path.exists():
                deltas[info["path"]] = np.load(delta_path)
        return deltas
```

### Step 3: Create Delta Computation Script

Create `scripts/compute_teacher_deltas.py`:

```python
"""Compute and cache delta embeddings for all teacher LoRA checkpoints."""
import argparse
import torch
import gc
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from llgbm.probes import create_generic_probes, create_domain_probes
from llgbm.delta import (
    DeltaCache,
    compute_base_activation,
    compute_adapter_delta,
)

def find_adapter_paths(checkpoint_dir: str) -> list:
    """Find all LoRA adapter paths in a directory."""
    checkpoint_dir = Path(checkpoint_dir)
    adapter_paths = []

    # Look for adapter_config.json or adapter_model.safetensors
    for path in checkpoint_dir.rglob("adapter_config.json"):
        adapter_paths.append(str(path.parent))

    return sorted(adapter_paths)


def main():
    parser = argparse.ArgumentParser(description="Compute delta embeddings for teacher LoRAs")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B",
                        help="Base model name or path")
    parser.add_argument("--checkpoint_dir", type=str, default="data/teacher_checkpoints",
                        help="Directory containing teacher LoRA checkpoints")
    parser.add_argument("--cache_dir", type=str, default="deltas",
                        help="Directory to cache delta embeddings")
    parser.add_argument("--probe_type", type=str, default="generic",
                        choices=["generic", "math", "code", "commonsense"],
                        help="Type of probes to use")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Maximum sequence length for probes")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Model dtype")
    parser.add_argument("--force_recompute", action="store_true",
                        help="Recompute even if cached")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # Initialize cache
    cache = DeltaCache(args.cache_dir)

    # Get probes
    if args.probe_type == "generic":
        probes = create_generic_probes()
    else:
        probes = create_domain_probes(args.probe_type)

    print(f"Using {len(probes)} {args.probe_type} probes")

    # Compute or load base activation
    base_activation = cache.get_base_activation()
    if base_activation is None or args.force_recompute:
        print("Computing base model activation...")
        base_activation, tokenizer = compute_base_activation(
            args.base_model, probes, device, args.max_length, dtype
        )
        cache.save_base_activation(base_activation, {
            "base_model": args.base_model,
            "probe_type": args.probe_type,
            "num_probes": len(probes),
            "max_length": args.max_length,
        })
        print(f"Base activation shape: {base_activation.shape}")
        print(f"Base activation norm: {np.linalg.norm(base_activation):.4f}")
    else:
        print("Loaded cached base activation")
        # Still need tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # Find all adapter paths
    adapter_paths = find_adapter_paths(args.checkpoint_dir)
    print(f"Found {len(adapter_paths)} teacher adapters")

    # Compute deltas
    stats = {"norms": [], "computed": 0, "cached": 0, "failed": 0}

    for adapter_path in tqdm(adapter_paths, desc="Computing deltas"):
        # Check cache
        if not args.force_recompute:
            cached_delta = cache.get_delta(adapter_path)
            if cached_delta is not None:
                stats["cached"] += 1
                stats["norms"].append(np.linalg.norm(cached_delta))
                continue

        try:
            delta = compute_adapter_delta(
                args.base_model,
                adapter_path,
                probes,
                base_activation,
                tokenizer,
                device,
                args.max_length,
                dtype,
            )
            cache.save_delta(adapter_path, delta)
            stats["computed"] += 1
            stats["norms"].append(np.linalg.norm(delta))

        except Exception as e:
            print(f"\nFailed to compute delta for {adapter_path}: {e}")
            stats["failed"] += 1

        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()

    # Print summary
    import numpy as np
    print("\n" + "="*50)
    print("Delta Computation Summary")
    print("="*50)
    print(f"Total adapters: {len(adapter_paths)}")
    print(f"Computed: {stats['computed']}")
    print(f"Cached: {stats['cached']}")
    print(f"Failed: {stats['failed']}")

    if stats["norms"]:
        norms = np.array(stats["norms"])
        print(f"\nDelta norm statistics:")
        print(f"  Min:    {norms.min():.4f}")
        print(f"  Max:    {norms.max():.4f}")
        print(f"  Mean:   {norms.mean():.4f}")
        print(f"  Std:    {norms.std():.4f}")
        print(f"  Median: {np.median(norms):.4f}")

    print(f"\nCache saved to: {args.cache_dir}/")

if __name__ == "__main__":
    import numpy as np
    main()
```

### Step 4: Delta Visualization

Create `scripts/visualize_deltas.py`:

```python
"""Visualize delta embeddings to verify clustering."""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from llgbm.delta import DeltaCache

def visualize_tsne(deltas: dict, output_path: str):
    """Create t-SNE visualization of delta embeddings."""
    # Prepare data
    names = list(deltas.keys())
    embeddings = np.stack([deltas[n] for n in names])

    # Extract domain from path (assuming structure like .../domain/...)
    domains = []
    for name in names:
        parts = Path(name).parts
        # Try to extract domain from path
        domain = "unknown"
        for part in parts:
            if part in ["math", "code", "commonsense", "legal", "medical"]:
                domain = part
                break
        domains.append(domain)

    unique_domains = list(set(domains))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_domains)))
    domain_to_color = {d: c for d, c in zip(unique_domains, colors)}

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=min(5, len(embeddings)-1), random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(10, 8))
    for i, (x, y) in enumerate(embeddings_2d):
        plt.scatter(x, y, c=[domain_to_color[domains[i]]], s=100, alpha=0.7)
        plt.annotate(Path(names[i]).stem[:10], (x, y), fontsize=8)

    # Legend
    for domain, color in domain_to_color.items():
        plt.scatter([], [], c=[color], label=domain)
    plt.legend()

    plt.title("Delta Embeddings t-SNE")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved t-SNE plot to {output_path}")


def compute_similarity_matrix(deltas: dict, output_path: str):
    """Compute and visualize cosine similarity matrix."""
    names = list(deltas.keys())
    embeddings = np.stack([deltas[n] for n in names])

    # Compute cosine similarity
    sim_matrix = cosine_similarity(embeddings)

    # Plot
    plt.figure(figsize=(12, 10))
    plt.imshow(sim_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(label='Cosine Similarity')

    # Labels
    short_names = [Path(n).stem[:15] for n in names]
    plt.xticks(range(len(names)), short_names, rotation=90, fontsize=8)
    plt.yticks(range(len(names)), short_names, fontsize=8)

    plt.title("Delta Embedding Similarity Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved similarity matrix to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="deltas")
    parser.add_argument("--output_dir", type=str, default="outputs/delta_analysis")
    args = parser.parse_args()

    # Load deltas
    cache = DeltaCache(args.cache_dir)
    deltas = cache.get_all_deltas()

    if len(deltas) < 2:
        print("Need at least 2 deltas for visualization")
        return

    print(f"Loaded {len(deltas)} delta embeddings")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Visualizations
    visualize_tsne(deltas, str(output_dir / "tsne.png"))
    compute_similarity_matrix(deltas, str(output_dir / "similarity.png"))

    # Compute clustering quality if we have domain labels
    names = list(deltas.keys())
    embeddings = np.stack([deltas[n] for n in names])

    # Try to extract domain labels
    domains = []
    for name in names:
        parts = Path(name).parts
        domain = "unknown"
        for part in parts:
            if part in ["math", "code", "commonsense", "legal", "medical"]:
                domain = part
                break
        domains.append(domain)

    unique_domains = list(set(domains))
    if len(unique_domains) > 1 and "unknown" not in unique_domains:
        domain_to_idx = {d: i for i, d in enumerate(unique_domains)}
        labels = [domain_to_idx[d] for d in domains]

        score = silhouette_score(embeddings, labels, metric='cosine')
        print(f"\nSilhouette score (by domain): {score:.4f}")

if __name__ == "__main__":
    main()
```

## File Structure After Phase 1

```
llgbm/
├── llgbm/
│   ├── __init__.py
│   ├── probes.py          # Probe templates
│   └── delta.py           # Delta computation
├── scripts/
│   ├── compute_teacher_deltas.py
│   └── visualize_deltas.py
├── deltas/                # Cache directory
│   ├── manifest.json
│   ├── base_activation.npy
│   └── {adapter_id}.npy
└── outputs/
    └── delta_analysis/
        ├── tsne.png
        └── similarity.png
```

## Acceptance Criteria

- [ ] Can compute base activation for Qwen2.5-1.5B
- [ ] Can compute delta for N adapters without memory leaks
- [ ] Cache manifest correctly maps adapter paths to delta files
- [ ] Delta norms are reasonable (not all zeros, not exploding)
- [ ] Deterministic output with fixed seed
- [ ] t-SNE shows some clustering by domain (if domain labels available)

## Usage

```bash
# Compute deltas for all teacher checkpoints
python scripts/compute_teacher_deltas.py \
    --base_model Qwen/Qwen2.5-1.5B \
    --checkpoint_dir data/teacher_checkpoints \
    --cache_dir deltas \
    --probe_type generic

# Visualize results
python scripts/visualize_deltas.py \
    --cache_dir deltas \
    --output_dir outputs/delta_analysis
```

## Next Phase
Proceed to **Phase 2** to add delta labels to the training dataset.
