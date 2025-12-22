# Phase 0 — Baseline Reproduction (DnD on Qwen2.5-1.5B)

## Goal
Verify the existing DnD framework works end-to-end for Qwen2.5-1.5B LoRA generation before adding delta supervision.

## Prerequisites
- CUDA-capable GPU with sufficient VRAM (16GB+ recommended)
- Python 3.12+
- Access to Qwen2.5-1.5B model weights

## Implementation Steps

### Step 1: Environment Setup

```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install torch transformers accelerate peft bitsandbytes
uv pip install safetensors wandb scikit-learn seaborn matplotlib
uv pip install huggingface_hub tqdm numpy
```

### Step 2: Verify DnD Imports

Create `scripts/verify_imports.py`:

```python
"""Verify all DnD components can be imported."""
import sys
sys.path.insert(0, "Drag-and-Drop-LLMs")

def verify_imports():
    # Model components
    from workspace.dnd.model.decoderonly import (
        HyperConvDecoderModel,
        HyperConvDecoderModel_FullCond,
        HyperConvDecoderModel_SuperLarge
    )
    print("[OK] Model imports")

    # Tokenizer
    from workspace.dnd.tokenizer.register import Qwen2515LoRA_Tokenizer2D
    print("[OK] Tokenizer imports")

    # Dataset
    from workspace.dnd.dataset.register import (
        Text2Qwen25LoRA_FullCondDataset,
        Text2Qwen25LoRA_CondQ_ADataset
    )
    print("[OK] Dataset imports")

    # Modules
    from workspace.dnd.module.hyperconv import HyperConvDecoder
    print("[OK] HyperConv imports")

    print("\nAll imports successful!")

if __name__ == "__main__":
    verify_imports()
```

### Step 3: Prepare Sample Data

Create `scripts/prepare_sample_data.py`:

```python
"""Create minimal sample data for sanity check."""
import os
import torch
from safetensors.torch import save_file
from pathlib import Path

def create_dummy_lora_checkpoint(output_dir: str, rank: int = 16):
    """
    Create a dummy LoRA checkpoint matching Qwen2.5-1.5B structure.

    LoRA targets for Qwen2.5-1.5B:
    - q_proj, k_proj, v_proj, o_proj (attention)
    - gate_proj, up_proj, down_proj (MLP)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Qwen2.5-1.5B config
    hidden_size = 1536
    intermediate_size = 8960
    num_layers = 28
    num_heads = 12
    head_dim = hidden_size // num_heads

    lora_weights = {}

    for layer_idx in range(num_layers):
        prefix = f"base_model.model.model.layers.{layer_idx}"

        # Attention LoRA (q, k, v, o)
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            if proj in ["q_proj", "o_proj"]:
                in_feat, out_feat = hidden_size, hidden_size
            else:  # k_proj, v_proj might have different sizes for GQA
                in_feat, out_feat = hidden_size, hidden_size

            lora_weights[f"{prefix}.self_attn.{proj}.lora_A.weight"] = torch.randn(rank, in_feat) * 0.01
            lora_weights[f"{prefix}.self_attn.{proj}.lora_B.weight"] = torch.zeros(out_feat, rank)

        # MLP LoRA
        lora_weights[f"{prefix}.mlp.gate_proj.lora_A.weight"] = torch.randn(rank, hidden_size) * 0.01
        lora_weights[f"{prefix}.mlp.gate_proj.lora_B.weight"] = torch.zeros(intermediate_size, rank)

        lora_weights[f"{prefix}.mlp.up_proj.lora_A.weight"] = torch.randn(rank, hidden_size) * 0.01
        lora_weights[f"{prefix}.mlp.up_proj.lora_B.weight"] = torch.zeros(intermediate_size, rank)

        lora_weights[f"{prefix}.mlp.down_proj.lora_A.weight"] = torch.randn(rank, intermediate_size) * 0.01
        lora_weights[f"{prefix}.mlp.down_proj.lora_B.weight"] = torch.zeros(hidden_size, rank)

    # Save checkpoint
    save_file(lora_weights, os.path.join(output_dir, "adapter_model.safetensors"))

    # Create adapter_config.json
    import json
    config = {
        "base_model_name_or_path": "Qwen/Qwen2.5-1.5B",
        "r": rank,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        "bias": "none",
        "task_type": "CAUSAL_LM"
    }
    with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Created dummy checkpoint at {output_dir}")
    print(f"Total parameters: {sum(p.numel() for p in lora_weights.values()):,}")

def create_sample_dataset(output_dir: str, num_samples: int = 10):
    """Create sample prompts dataset."""
    os.makedirs(output_dir, exist_ok=True)

    sample_prompts = [
        "Solve the equation: 2x + 5 = 15",
        "What is the derivative of x^2?",
        "Calculate the area of a circle with radius 5",
        "Simplify: (3x + 2)(x - 4)",
        "Find the roots of x^2 - 5x + 6 = 0",
        "What is 15% of 80?",
        "Convert 3/4 to a decimal",
        "What is the sum of angles in a triangle?",
        "Calculate: 125 / 5 + 3 * 4",
        "Find the GCD of 24 and 36"
    ]

    import json
    with open(os.path.join(output_dir, "prompts.json"), "w") as f:
        json.dump(sample_prompts[:num_samples], f, indent=2)

    print(f"Created {num_samples} sample prompts at {output_dir}")

if __name__ == "__main__":
    # Create sample data directory structure
    create_dummy_lora_checkpoint("data/sample_checkpoints/math_lora_001")
    create_dummy_lora_checkpoint("data/sample_checkpoints/math_lora_002")
    create_sample_dataset("data/sample_prompts/math")
```

### Step 4: Tokenizer Sanity Check

Create `scripts/test_tokenizer.py`:

```python
"""Test LoRA tokenization and detokenization."""
import sys
sys.path.insert(0, "Drag-and-Drop-LLMs")

import torch
from safetensors.torch import load_file
from workspace.dnd.tokenizer.register import Qwen2515LoRA_Tokenizer2D

def test_tokenizer_roundtrip(checkpoint_path: str):
    """Test that tokenize -> detokenize recovers original weights."""

    # Load checkpoint
    weights = load_file(checkpoint_path)
    print(f"Loaded {len(weights)} tensors from checkpoint")

    # Initialize tokenizer
    tokenizer = Qwen2515LoRA_Tokenizer2D()

    # Tokenize
    tokens, scales = tokenizer.tokenize(weights)
    print(f"Tokenized shape: {tokens.shape}")  # Expected: (num_tokens, H, W)
    print(f"Scales shape: {scales.shape}")

    # Detokenize
    reconstructed = tokenizer.detokenize(tokens, scales)
    print(f"Reconstructed {len(reconstructed)} tensors")

    # Compare
    total_error = 0.0
    for key in weights:
        if key in reconstructed:
            error = torch.abs(weights[key] - reconstructed[key]).mean().item()
            total_error += error
            if error > 1e-5:
                print(f"  {key}: error = {error:.6f}")

    avg_error = total_error / len(weights)
    print(f"\nAverage reconstruction error: {avg_error:.8f}")

    if avg_error < 1e-5:
        print("[PASS] Tokenizer roundtrip test")
    else:
        print("[WARN] Reconstruction error above threshold")

    return tokens, scales

if __name__ == "__main__":
    test_tokenizer_roundtrip("data/sample_checkpoints/math_lora_001/adapter_model.safetensors")
```

### Step 5: Minimal Training Loop

Create `scripts/train_baseline.py`:

```python
"""Minimal DnD training sanity check for Qwen2.5-1.5B."""
import sys
sys.path.insert(0, "Drag-and-Drop-LLMs")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from accelerate import Accelerator

from workspace.dnd.model.decoderonly import HyperConvDecoderModel_SuperLarge
from workspace.dnd.tokenizer.register import Qwen2515LoRA_Tokenizer2D
from workspace.dnd.dataset.register import Text2Qwen25LoRA_FullCondDataset

def get_config():
    """Training configuration for Qwen2.5-1.5B baseline."""
    return {
        # Data
        "token_size": (18, 258),  # Qwen2.5-1.5B LoRA token dimensions
        "max_text_length": 512,
        "batch_size": 4,

        # Training
        "total_steps": 100,
        "learning_rate": 3e-5,
        "warmup_steps": 10,
        "log_interval": 10,
        "save_interval": 50,

        # Model
        "extractor_type": "BERT",
        "condition_model": "bert-base-uncased",

        # Paths
        "checkpoint_folders": ["data/sample_checkpoints/math_lora_001"],
        "prompt_folders": ["data/sample_prompts/math"],
        "output_dir": "outputs/baseline_test",
    }

def train():
    config = get_config()
    accelerator = Accelerator()
    device = accelerator.device

    # Initialize tokenizer
    lora_tokenizer = Qwen2515LoRA_Tokenizer2D()

    # Initialize text tokenizer and condition model
    text_tokenizer = AutoTokenizer.from_pretrained(config["condition_model"])
    condition_model = AutoModel.from_pretrained(config["condition_model"])

    # Initialize generator model
    model = HyperConvDecoderModel_SuperLarge(
        features=[
            (32, config["token_size"][0], config["token_size"][1]),
            (64, config["token_size"][0], config["token_size"][1]),
            (128, config["token_size"][0], config["token_size"][1]),
        ],
        condition_dim=(768, 16, 16),  # BERT hidden size
        extra_condition_module=condition_model,
        extractor_type=config["extractor_type"],
    )

    # Initialize dataset
    # Note: This is a simplified version - actual implementation needs proper data loading
    dataset = Text2Qwen25LoRA_FullCondDataset(
        checkpoint_folder=config["checkpoint_folders"][0],
        tokenizer=lora_tokenizer,
        text_tokenizer=text_tokenizer,
        max_text_length=config["max_text_length"],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=dataset.collate_fn_train,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

    # Prepare for distributed training
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # Training loop
    model.train()
    global_step = 0

    for epoch in range(100):  # Loop until steps reached
        for batch in dataloader:
            tokens, condition, _ = batch

            # Forward pass
            loss = model(tokens, condition)

            # Backward pass
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

            if global_step % config["log_interval"] == 0:
                print(f"Step {global_step}: loss = {loss.item():.4f}")

            if global_step >= config["total_steps"]:
                break

        if global_step >= config["total_steps"]:
            break

    print(f"\nTraining complete! Final loss: {loss.item():.4f}")

    # Save checkpoint
    accelerator.save_state(config["output_dir"])
    print(f"Saved checkpoint to {config['output_dir']}")

if __name__ == "__main__":
    train()
```

### Step 6: Generation Test

Create `scripts/test_generation.py`:

```python
"""Test generating a LoRA checkpoint from the trained model."""
import sys
sys.path.insert(0, "Drag-and-Drop-LLMs")

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer, AutoModel

from workspace.dnd.model.decoderonly import HyperConvDecoderModel_SuperLarge
from workspace.dnd.tokenizer.register import Qwen2515LoRA_Tokenizer2D

def generate_lora(model, text_condition: str, lora_tokenizer, text_tokenizer, device):
    """Generate a LoRA checkpoint from a text condition."""
    model.eval()

    # Tokenize condition
    inputs = text_tokenizer(
        text_condition,
        return_tensors="pt",
        padding="max_length",
        max_length=512,
        truncation=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate tokens
    with torch.no_grad():
        tokens_pred, scales_pred = model.generate(
            condition=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

    # Detokenize to weights
    lora_weights = lora_tokenizer.detokenize(tokens_pred[0], scales_pred[0])

    return lora_weights

def test_generation(checkpoint_path: str, output_path: str):
    """Test end-to-end generation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load components
    lora_tokenizer = Qwen2515LoRA_Tokenizer2D()
    text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    condition_model = AutoModel.from_pretrained("bert-base-uncased")

    # Load trained model
    model = HyperConvDecoderModel_SuperLarge(
        features=[(32, 18, 258), (64, 18, 258), (128, 18, 258)],
        condition_dim=(768, 16, 16),
        extra_condition_module=condition_model,
        extractor_type="BERT",
    )
    model.load_state_dict(torch.load(checkpoint_path))
    model = model.to(device)

    # Generate
    test_condition = "Solve mathematical equations step by step"
    lora_weights = generate_lora(
        model, test_condition, lora_tokenizer, text_tokenizer, device
    )

    # Save
    save_file(lora_weights, output_path)
    print(f"Generated LoRA checkpoint saved to {output_path}")
    print(f"Total tensors: {len(lora_weights)}")
    print(f"Total parameters: {sum(p.numel() for p in lora_weights.values()):,}")

if __name__ == "__main__":
    test_generation(
        "outputs/baseline_test/model.pt",
        "outputs/generated_lora.safetensors"
    )
```

## File Structure After Phase 0

```
llgbm/
├── scripts/
│   ├── verify_imports.py
│   ├── prepare_sample_data.py
│   ├── test_tokenizer.py
│   ├── train_baseline.py
│   └── test_generation.py
├── data/
│   ├── sample_checkpoints/
│   │   ├── math_lora_001/
│   │   └── math_lora_002/
│   └── sample_prompts/
│       └── math/
├── outputs/
│   └── baseline_test/
└── Drag-and-Drop-LLMs/  (existing)
```

## Acceptance Criteria

- [ ] All imports in `verify_imports.py` succeed
- [ ] Tokenizer roundtrip error < 1e-5
- [ ] Training loop runs for 100 steps without OOM
- [ ] Loss decreases over training
- [ ] Can generate and save a LoRA checkpoint

## Troubleshooting

### OOM Errors
- Reduce batch_size to 1-2
- Use gradient checkpointing
- Enable mixed precision with `accelerator = Accelerator(mixed_precision="fp16")`

### Import Errors
- Ensure `Drag-and-Drop-LLMs` is in the Python path
- Check for missing dependencies in `workspace/dnd/__init__.py`

### Tokenizer Dimension Mismatch
- Verify `token_size` matches the actual Qwen2.5-1.5B LoRA structure
- Check `Qwen2515LoRA_Tokenizer2D` configuration

## Next Phase
Once baseline is verified, proceed to **Phase 1** to compute delta embeddings for teacher LoRAs.
