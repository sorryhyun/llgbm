"""Delta embedding computation for behavioral matching.

This module implements the core delta embedding computation from the
"Delta Activations" approach: given a base model B and an adapted model M,
the delta embedding is:

    delta(M, B) = v(M) - v(B)

where v(M) is the average last-layer hidden state over a set of probe texts.
"""

import gc
import json
import hashlib
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm


def get_average_activation(
    model: torch.nn.Module,
    texts: List[str],
    tokenizer,
    device: torch.device,
    max_length: int = 256,
    layer_idx: int = -1,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Compute the average activation from a model for given probe texts.

    Extracts the hidden state at the last token position from the specified
    layer, then averages across all probe texts.

    Args:
        model: The model to analyze (must support output_hidden_states=True)
        texts: List of probe strings
        tokenizer: Model tokenizer
        device: Device to run inference on
        max_length: Maximum sequence length
        layer_idx: Which layer to extract (-1 = last layer)
        show_progress: Whether to show progress bar

    Returns:
        numpy.ndarray: Average activation vector of shape (hidden_size,)
    """
    model.eval()
    activations = []

    iterator = tqdm(texts, desc="Computing activations", leave=False) if show_progress else texts

    for text in iterator:
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
    show_progress: bool = True,
) -> Tuple[np.ndarray, "AutoTokenizer"]:
    """
    Compute activation for the base model.

    This should be cached and reused across all adapter delta computations.

    Args:
        base_model_name: HuggingFace model ID or path
        probes: List of probe texts
        device: Compute device
        max_length: Max sequence length for probes
        dtype: Model dtype for inference
        show_progress: Whether to show progress bar

    Returns:
        Tuple of (base_activation array, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map=device if isinstance(device, str) else {"": device},
        trust_remote_code=True,
    )
    model.config.output_hidden_states = True

    base_activation = get_average_activation(
        model, probes, tokenizer, device, max_length, show_progress=show_progress
    )

    # Free memory
    del model
    gc.collect()
    if torch.cuda.is_available():
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
    show_progress: bool = True,
) -> np.ndarray:
    """
    Compute delta embedding for a single adapter.

    Args:
        base_model_name: HuggingFace model ID for base
        adapter_path: Path to the LoRA adapter directory
        probes: List of probe texts
        base_activation: Pre-computed base model activation
        tokenizer: Tokenizer (shared with base)
        device: Compute device
        max_length: Max sequence length for probes
        dtype: Model dtype
        show_progress: Whether to show progress bar

    Returns:
        Delta embedding (adapted - base) as numpy array of shape (hidden_size,)
    """
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    # Load base model with adapter
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map=device if isinstance(device, str) else {"": device},
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.config.output_hidden_states = True
    model.eval()

    # Compute adapted activation
    adapted_activation = get_average_activation(
        model, probes, tokenizer, device, max_length, show_progress=show_progress
    )

    # Compute delta
    delta = adapted_activation - base_activation

    # Free memory
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return delta


class DeltaCache:
    """
    Manages caching of delta embeddings to disk.

    Cache structure:
        cache_dir/
        ├── manifest.json       # Maps adapter paths to delta files
        ├── base_activation.npy # Cached base activation
        └── {adapter_id}.npy    # Individual delta embeddings

    The manifest tracks:
    - Which adapters have been processed
    - Delta norms and shapes for quick inspection
    - Configuration used for computation
    """

    def __init__(self, cache_dir: str = "deltas"):
        """
        Initialize delta cache.

        Args:
            cache_dir: Directory to store cached embeddings
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.cache_dir / "manifest.json"
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict:
        """Load manifest from disk or create empty one."""
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                return json.load(f)
        return {"adapters": {}, "config": {}}

    def _save_manifest(self):
        """Save manifest to disk."""
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

    def _adapter_id(self, adapter_path: str) -> str:
        """Generate unique ID for adapter path."""
        return hashlib.md5(adapter_path.encode()).hexdigest()[:12]

    def get_base_activation(self) -> Optional[np.ndarray]:
        """Load cached base activation if exists."""
        path = self.cache_dir / "base_activation.npy"
        if path.exists():
            return np.load(path)
        return None

    def save_base_activation(self, activation: np.ndarray, config: Dict):
        """
        Save base activation and configuration.

        Args:
            activation: Base model activation array
            config: Configuration dict (base_model, probe_type, etc.)
        """
        np.save(self.cache_dir / "base_activation.npy", activation)
        self.manifest["config"] = config
        self._save_manifest()

    def get_delta(self, adapter_path: str) -> Optional[np.ndarray]:
        """
        Load cached delta for adapter if exists.

        Args:
            adapter_path: Path to the adapter directory

        Returns:
            Delta array if cached, None otherwise
        """
        adapter_id = self._adapter_id(adapter_path)
        if adapter_id in self.manifest["adapters"]:
            delta_path = self.cache_dir / f"{adapter_id}.npy"
            if delta_path.exists():
                return np.load(delta_path)
        return None

    def save_delta(self, adapter_path: str, delta: np.ndarray):
        """
        Save delta embedding for adapter.

        Args:
            adapter_path: Path to the adapter directory
            delta: Delta embedding array
        """
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
        """
        Load all cached deltas.

        Returns:
            Dict mapping adapter paths to delta arrays
        """
        deltas = {}
        for adapter_id, info in self.manifest["adapters"].items():
            delta_path = self.cache_dir / f"{adapter_id}.npy"
            if delta_path.exists():
                deltas[info["path"]] = np.load(delta_path)
        return deltas

    def get_delta_for_checkpoint(self, checkpoint_path: str) -> Optional[np.ndarray]:
        """
        Get delta for a checkpoint by matching path patterns.

        This is useful when checkpoint paths may have slight variations.

        Args:
            checkpoint_path: Path to search for (can be partial)

        Returns:
            Delta array if found, None otherwise
        """
        checkpoint_path = str(checkpoint_path)
        for adapter_id, info in self.manifest["adapters"].items():
            if checkpoint_path in info["path"] or info["path"] in checkpoint_path:
                delta_path = self.cache_dir / f"{adapter_id}.npy"
                if delta_path.exists():
                    return np.load(delta_path)
        return None

    def summary(self) -> Dict:
        """
        Get summary statistics of cached deltas.

        Returns:
            Dict with count, norm statistics, etc.
        """
        norms = [info["norm"] for info in self.manifest["adapters"].values()]
        if not norms:
            return {"count": 0}

        return {
            "count": len(norms),
            "config": self.manifest.get("config", {}),
            "norm_min": min(norms),
            "norm_max": max(norms),
            "norm_mean": np.mean(norms),
            "norm_std": np.std(norms),
        }

    def clear(self):
        """Clear all cached data."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = {"adapters": {}, "config": {}}
        self._save_manifest()
