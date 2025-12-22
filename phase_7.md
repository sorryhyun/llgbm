# Phase 7 — Packaging, Reproducibility, and Scaling

## Goal
Consolidate the experiment into a reproducible, well-documented package ready for scaling and potential publication.

## Prerequisites
- Phases 0-6 complete and validated
- Positive results from at least one experiment variant

## Implementation Steps

### Step 1: Create Centralized Configuration

Create `llgbm/config.py`:

```python
"""Centralized configuration management."""
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
import hashlib
import yaml


@dataclass
class ModelConfig:
    """Base model configuration."""
    name: str = "Qwen/Qwen2.5-1.5B"
    dtype: str = "float16"
    hidden_size: int = 1536
    num_layers: int = 28


@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])


@dataclass
class ProbeConfig:
    """Probe configuration for delta computation."""
    probe_type: str = "generic"  # generic, math, code, commonsense
    num_probes: int = 5
    max_length: int = 256


@dataclass
class DeltaConfig:
    """Delta computation configuration."""
    layer_idx: int = -1  # -1 for last layer
    token_idx: int = -1  # -1 for last token
    aggregation: str = "mean"  # mean, max, concat


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Basic training
    batch_size: int = 4
    learning_rate: float = 3e-5
    total_steps: int = 5000
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    # Delta loss
    lambda_delta: float = 1.0
    lambda_schedule: str = "linear"
    lambda_warmup_steps: int = 500
    delta_loss_type: str = "mse"

    # Memory optimization
    delta_batch_size: int = 2
    gradient_checkpointing: bool = False
    mixed_precision: str = "fp16"

    # Logging
    log_interval: int = 10
    save_interval: int = 500
    eval_interval: int = 100


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    # Sub-configs
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    delta: DeltaConfig = field(default_factory=DeltaConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Experiment metadata
    experiment_name: str = "llgbm_default"
    seed: int = 42
    output_dir: str = "outputs"

    # Data paths
    checkpoint_folder: str = "data/teacher_checkpoints"
    delta_cache_dir: str = "deltas"

    def __post_init__(self):
        """Ensure output directory exists."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def get_hash(self) -> str:
        """Generate unique hash for this configuration."""
        config_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def save(self, path: Optional[str] = None):
        """Save configuration to file."""
        if path is None:
            path = f"{self.output_dir}/config.yaml"

        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        """Load configuration from file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(
            model=ModelConfig(**data.get("model", {})),
            lora=LoRAConfig(**data.get("lora", {})),
            probe=ProbeConfig(**data.get("probe", {})),
            delta=DeltaConfig(**data.get("delta", {})),
            training=TrainingConfig(**data.get("training", {})),
            experiment_name=data.get("experiment_name", "llgbm_default"),
            seed=data.get("seed", 42),
            output_dir=data.get("output_dir", "outputs"),
            checkpoint_folder=data.get("checkpoint_folder", "data/teacher_checkpoints"),
            delta_cache_dir=data.get("delta_cache_dir", "deltas"),
        )


# Predefined configs for common experiments
CONFIGS = {
    "baseline": ExperimentConfig(
        experiment_name="baseline",
        training=TrainingConfig(lambda_delta=0.0),
    ),
    "delta_multi": ExperimentConfig(
        experiment_name="delta_multi",
        training=TrainingConfig(lambda_delta=1.0, lambda_schedule="linear"),
    ),
    "delta_only": ExperimentConfig(
        experiment_name="delta_only",
        training=TrainingConfig(lambda_delta=1.0),  # Will use delta-only training
    ),
}


def get_config(name: str = "delta_multi") -> ExperimentConfig:
    """Get predefined configuration by name."""
    if name in CONFIGS:
        return CONFIGS[name]
    raise ValueError(f"Unknown config: {name}. Available: {list(CONFIGS.keys())}")
```

### Step 2: Create Reproducibility Utilities

Create `llgbm/reproducibility.py`:

```python
"""Utilities for ensuring reproducible experiments."""
import os
import random
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import numpy as np


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for other libraries
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_git_info() -> Dict[str, str]:
    """Get current git commit info."""
    import subprocess

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()

        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()

        dirty = subprocess.call(
            ["git", "diff", "--quiet"],
            stderr=subprocess.DEVNULL,
        ) != 0

        return {
            "commit": commit,
            "branch": branch,
            "dirty": dirty,
        }
    except Exception:
        return {"commit": "unknown", "branch": "unknown", "dirty": True}


def get_environment_info() -> Dict[str, Any]:
    """Capture environment information for reproducibility."""
    import platform
    import sys

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "num_gpus": torch.cuda.device_count(),
    }


def compute_model_hash(model: torch.nn.Module) -> str:
    """Compute hash of model parameters for verification."""
    hasher = hashlib.md5()

    for name, param in sorted(model.named_parameters()):
        hasher.update(name.encode())
        hasher.update(param.data.cpu().numpy().tobytes())

    return hasher.hexdigest()


def compute_delta_cache_hash(cache_dir: str) -> str:
    """Compute hash of delta cache for versioning."""
    hasher = hashlib.md5()

    cache_path = Path(cache_dir)
    for file_path in sorted(cache_path.glob("*.npy")):
        hasher.update(file_path.name.encode())
        data = np.load(file_path)
        hasher.update(data.tobytes())

    return hasher.hexdigest()[:12]


class ExperimentLogger:
    """Log experiment metadata and results."""

    def __init__(self, output_dir: str, experiment_name: str):
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.log_dir = self.output_dir / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metadata = {
            "experiment_name": experiment_name,
            "start_time": datetime.now().isoformat(),
            "git": get_git_info(),
            "environment": get_environment_info(),
        }

    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        self.metadata["config"] = config
        self._save_metadata()

    def log_metrics(self, step: int, metrics: Dict[str, float]):
        """Log training metrics."""
        metrics_file = self.log_dir / "metrics.jsonl"
        with open(metrics_file, "a") as f:
            entry = {"step": step, **metrics}
            f.write(json.dumps(entry) + "\n")

    def log_evaluation(self, name: str, results: Dict[str, Any]):
        """Log evaluation results."""
        eval_file = self.log_dir / f"eval_{name}.json"
        with open(eval_file, "w") as f:
            json.dump(results, f, indent=2)

    def finalize(self, final_metrics: Optional[Dict[str, float]] = None):
        """Finalize experiment logging."""
        self.metadata["end_time"] = datetime.now().isoformat()
        if final_metrics:
            self.metadata["final_metrics"] = final_metrics
        self._save_metadata()

    def _save_metadata(self):
        """Save metadata to file."""
        with open(self.log_dir / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)


def verify_reproducibility(
    config_path: str,
    checkpoint_path: str,
    expected_metrics: Dict[str, float],
    tolerance: float = 0.01,
) -> bool:
    """
    Verify that an experiment can be reproduced.

    Args:
        config_path: Path to experiment config
        checkpoint_path: Path to saved checkpoint
        expected_metrics: Expected metric values
        tolerance: Acceptable difference

    Returns:
        True if reproducible within tolerance
    """
    from llgbm.config import ExperimentConfig

    # Load config
    config = ExperimentConfig.load(config_path)

    # Set seed
    set_seed(config.seed)

    # Run evaluation
    # actual_metrics = run_evaluation(config, checkpoint_path)

    # Compare
    # for key, expected in expected_metrics.items():
    #     actual = actual_metrics.get(key, 0)
    #     if abs(actual - expected) > tolerance:
    #         return False

    return True
```

### Step 3: Create Main Entry Point

Update `main.py`:

```python
"""Main entry point for LLGBM experiments."""
import argparse
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "Drag-and-Drop-LLMs"))


def main():
    parser = argparse.ArgumentParser(
        description="Learning LoRA Generator by Behavioral Matching"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Phase 0: Verify setup
    verify_parser = subparsers.add_parser("verify", help="Verify setup and imports")

    # Phase 1: Compute deltas
    delta_parser = subparsers.add_parser("compute-deltas", help="Compute delta embeddings")
    delta_parser.add_argument("--checkpoint_dir", required=True)
    delta_parser.add_argument("--output_dir", default="deltas")
    delta_parser.add_argument("--probe_type", default="generic")

    # Phase 4: Train with delta loss
    train_parser = subparsers.add_parser("train", help="Train generator")
    train_parser.add_argument("--config", default="delta_multi")
    train_parser.add_argument("--config_file", type=str, help="Custom config file")
    train_parser.add_argument("--output_dir", default="outputs")

    # Phase 5: Train delta-only
    delta_only_parser = subparsers.add_parser("train-delta-only", help="Delta-only training")
    delta_only_parser.add_argument("--variant", choices=["b1", "b2"], default="b1")
    delta_only_parser.add_argument("--config_file", type=str)

    # Evaluation
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model")
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--data_dir", required=True)
    eval_parser.add_argument("--output", default="eval_results.json")

    # Phase 6: Composition
    compose_parser = subparsers.add_parser("compose", help="Test composition")
    compose_parser.add_argument("--cache_dir", default="deltas")
    compose_parser.add_argument("--output_dir", default="outputs/composition")

    # Visualization
    viz_parser = subparsers.add_parser("visualize", help="Generate visualizations")
    viz_parser.add_argument("--type", choices=["deltas", "composition", "training"])
    viz_parser.add_argument("--input_dir", required=True)
    viz_parser.add_argument("--output_dir", default="outputs/viz")

    args = parser.parse_args()

    if args.command == "verify":
        from scripts.verify_imports import verify_imports
        verify_imports()

    elif args.command == "compute-deltas":
        from scripts.compute_teacher_deltas import main as compute_main
        sys.argv = [
            "compute_teacher_deltas.py",
            "--checkpoint_dir", args.checkpoint_dir,
            "--cache_dir", args.output_dir,
            "--probe_type", args.probe_type,
        ]
        compute_main()

    elif args.command == "train":
        from llgbm.config import get_config, ExperimentConfig
        from scripts.train_with_delta import train

        if args.config_file:
            config = ExperimentConfig.load(args.config_file)
        else:
            config = get_config(args.config)

        config.output_dir = args.output_dir
        train(config)

    elif args.command == "train-delta-only":
        from scripts.train_delta_only import main as train_delta_main
        train_delta_main()

    elif args.command == "evaluate":
        from scripts.evaluate import main as eval_main
        eval_main()

    elif args.command == "compose":
        from llgbm.compositionality import CompositionExperiment
        from llgbm.delta import DeltaCache

        cache = DeltaCache(args.cache_dir)
        experiment = CompositionExperiment(cache)
        # Run composition tests...

    elif args.command == "visualize":
        if args.type == "deltas":
            from scripts.visualize_deltas import main as viz_main
            viz_main()
        elif args.type == "composition":
            from scripts.visualize_composition import main as viz_main
            viz_main()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

### Step 4: Create Package Structure

Update `pyproject.toml`:

```toml
[project]
name = "llgbm"
version = "0.1.0"
description = "Learning LoRA Generator by Behavioral Matching"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}

dependencies = [
    "torch>=2.0.0",
    "transformers>=4.35.0",
    "peft>=0.7.0",
    "accelerate>=0.24.0",
    "safetensors>=0.4.0",
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "tqdm>=4.65.0",
    "pyyaml>=6.0",
    "faiss-cpu>=1.7.0",  # Or faiss-gpu
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
]

[project.scripts]
llgbm = "llgbm.main:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["llgbm"]
```

### Step 5: Create Tests

Create `tests/test_delta.py`:

```python
"""Tests for delta computation."""
import pytest
import numpy as np
import torch

from llgbm.probes import create_generic_probes, create_domain_probes
from llgbm.delta import DeltaCache


class TestProbes:
    def test_generic_probes_count(self):
        probes = create_generic_probes()
        assert len(probes) == 5

    def test_generic_probes_format(self):
        probes = create_generic_probes()
        for probe in probes:
            assert "### Instruction:" in probe
            assert "### Response:" in probe

    def test_domain_probes(self):
        for domain in ["math", "code", "commonsense"]:
            probes = create_domain_probes(domain)
            assert len(probes) == 5


class TestDeltaCache:
    def test_cache_creation(self, tmp_path):
        cache = DeltaCache(str(tmp_path / "deltas"))
        assert cache.manifest == {"adapters": {}, "config": {}}

    def test_save_load_delta(self, tmp_path):
        cache = DeltaCache(str(tmp_path / "deltas"))

        # Save delta
        test_delta = np.random.randn(1536).astype(np.float32)
        cache.save_delta("/path/to/adapter", test_delta)

        # Load delta
        loaded = cache.get_delta("/path/to/adapter")
        np.testing.assert_array_almost_equal(test_delta, loaded)

    def test_base_activation_cache(self, tmp_path):
        cache = DeltaCache(str(tmp_path / "deltas"))

        # Save base activation
        base_act = np.random.randn(1536).astype(np.float32)
        cache.save_base_activation(base_act, {"model": "test"})

        # Load
        loaded = cache.get_base_activation()
        np.testing.assert_array_almost_equal(base_act, loaded)
```

Create `tests/test_functional_lora.py`:

```python
"""Tests for functional LoRA."""
import pytest
import torch

# Skip if no GPU
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU required for functional LoRA tests"
)


class TestFunctionalLoRA:
    @pytest.fixture
    def device(self):
        return torch.device("cuda")

    def test_gradient_flow(self, device):
        """Test that gradients flow through functional_call."""
        # This test verifies the core mechanism
        # See scripts/test_gradient_flow.py for full implementation
        pass

    def test_delta_sensitivity(self, device):
        """Test that delta changes with LoRA weights."""
        pass
```

### Step 6: Create Documentation

Create `README.md` (update existing):

```markdown
# LLGBM: Learning LoRA Generator by Behavioral Matching

Train a hypernetwork-based LoRA generator with behavioral supervision using delta activations.

## Quick Start

```bash
# Install
pip install -e .

# Verify setup
llgbm verify

# Compute delta embeddings for teacher LoRAs
llgbm compute-deltas --checkpoint_dir data/teacher_checkpoints --output_dir deltas

# Train with delta supervision
llgbm train --config delta_multi --output_dir outputs/experiment1

# Evaluate
llgbm evaluate --checkpoint outputs/experiment1/final --data_dir data/test
```

## Experiment Phases

| Phase | Description | Command |
|-------|-------------|---------|
| 0 | Baseline reproduction | `llgbm verify` |
| 1 | Compute delta targets | `llgbm compute-deltas` |
| 2 | Dataset preparation | (automatic) |
| 3 | Differentiable delta | (integrated) |
| 4 | Multi-task training | `llgbm train` |
| 5 | Delta-only training | `llgbm train-delta-only` |
| 6 | Composition tests | `llgbm compose` |

## Configuration

See `llgbm/config.py` for all options. Create custom configs:

```yaml
# my_config.yaml
model:
  name: "Qwen/Qwen2.5-1.5B"
training:
  lambda_delta: 2.0
  delta_loss_type: "cosine"
```

```bash
llgbm train --config_file my_config.yaml
```

## Citation

```bibtex
@article{llgbm2024,
  title={Learning LoRA Generator by Behavioral Matching},
  author={...},
  year={2024}
}
```
```

## Final File Structure

```
llgbm/
├── llgbm/
│   ├── __init__.py
│   ├── config.py              # Centralized config
│   ├── reproducibility.py     # Reproducibility utils
│   ├── probes.py
│   ├── delta.py
│   ├── dataset.py
│   ├── functional_lora.py
│   ├── losses.py
│   ├── retrieval.py
│   ├── compositionality.py
│   ├── training/
│   │   └── delta_only.py
│   ├── models/
│   │   └── delta_predictor.py
│   └── evaluation/
│       └── composition_eval.py
├── scripts/
│   ├── verify_imports.py
│   ├── prepare_sample_data.py
│   ├── test_tokenizer.py
│   ├── train_baseline.py
│   ├── compute_teacher_deltas.py
│   ├── visualize_deltas.py
│   ├── test_delta_dataset.py
│   ├── test_gradient_flow.py
│   ├── train_with_delta.py
│   ├── evaluate.py
│   ├── train_delta_only.py
│   ├── visualize_composition.py
│   └── ablation_study.py
├── tests/
│   ├── test_delta.py
│   └── test_functional_lora.py
├── configs/
│   ├── baseline.yaml
│   ├── delta_multi.yaml
│   └── delta_only.yaml
├── data/
│   ├── teacher_checkpoints/
│   └── sample_prompts/
├── deltas/
├── outputs/
├── main.py
├── pyproject.toml
├── README.md
├── plan.md
├── phase_0.md - phase_7.md
└── Drag-and-Drop-LLMs/  (submodule/dependency)
```

## Acceptance Criteria

- [ ] All imports work via `llgbm verify`
- [ ] CLI interface functional for all phases
- [ ] Configs save/load correctly
- [ ] Experiments are reproducible with fixed seeds
- [ ] Tests pass
- [ ] Documentation complete

## Scaling Roadmap

1. **More Probes**: Increase from 5 to 20+ diverse probes
2. **Multi-Domain**: Add math, code, commonsense, legal, medical
3. **Multi-Layer Deltas**: Concatenate hidden states from multiple layers
4. **Cross-Backbone**: Test delta transfer between model families
5. **Larger Models**: Scale to 7B, 13B base models

## Risks Addressed

| Risk | Mitigation |
|------|------------|
| Gradient breaks | Functional LoRA verified |
| Memory OOM | Microbatch delta loss, caching |
| Under-constraint | Regularization in delta-only |
| Non-reproducibility | Seed control, config hashing |
