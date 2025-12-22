# Phase 3 — Differentiable Delta Computation for Generated LoRA (Core Engineering)

## Goal
Implement differentiable delta computation that allows gradients to flow from the delta loss back through the generated LoRA weights to the generator network.

## Prerequisites
- Phase 2 complete (dataset returns delta labels)
- Understanding of `torch.func.functional_call`
- PEFT library installed

## The Challenge

The standard approach of applying LoRA (using `load_state_dict` or `model.merge_and_unload()`) **breaks gradients** because:
1. `state_dict` operations are not differentiable
2. In-place weight modifications don't track gradients

We need to compute `v(B + θ̂)` where `θ̂` is the generated LoRA output, while maintaining the gradient path from the loss back to the generator.

## Solution: `torch.func.functional_call`

PyTorch's `functional_call` allows us to run a model's forward pass with **externally provided parameters**, maintaining differentiability.

```python
# Instead of: model.load_state_dict(new_params); output = model(x)
# We use:     output = functional_call(model, new_params, (x,))
```

## Implementation Steps

### Step 1: Create Functional LoRA Module

Create `llgbm/functional_lora.py`:

```python
"""Differentiable LoRA application using functional_call."""
import torch
import torch.nn as nn
from torch.func import functional_call
from typing import Dict, Tuple, Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import copy


class FunctionalLoRAModel:
    """
    Wrapper for applying LoRA weights differentiably using functional_call.

    The key insight: We create a PEFT model structure once, then use
    functional_call to swap in generated LoRA parameters while maintaining
    gradient flow.
    """

    def __init__(
        self,
        base_model_name: str,
        lora_config: LoraConfig,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ):
        """
        Args:
            base_model_name: HuggingFace model name
            lora_config: PEFT LoRA configuration
            device: Compute device
            dtype: Model dtype
        """
        self.device = device
        self.dtype = dtype
        self.lora_config = lora_config

        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.base_model.config.output_hidden_states = True

        # Create PEFT model structure (for parameter naming)
        self.peft_model = get_peft_model(self.base_model, lora_config)
        self.peft_model.eval()

        # Freeze all parameters
        for param in self.peft_model.parameters():
            param.requires_grad = False

        # Cache parameter names for LoRA layers
        self.lora_param_names = self._get_lora_param_names()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _get_lora_param_names(self) -> List[str]:
        """Get all LoRA parameter names in the PEFT model."""
        lora_names = []
        for name, _ in self.peft_model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                lora_names.append(name)
        return lora_names

    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get current LoRA parameters as a state dict."""
        return {
            name: param.clone()
            for name, param in self.peft_model.named_parameters()
            if "lora_A" in name or "lora_B" in name
        }

    def build_full_params(
        self,
        lora_weights: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Build complete parameter dict by merging generated LoRA weights
        with frozen base model parameters.

        Args:
            lora_weights: Generated LoRA A and B matrices

        Returns:
            Complete parameter dict for functional_call
        """
        # Start with all current parameters (frozen)
        full_params = dict(self.peft_model.named_parameters())

        # Replace LoRA parameters with generated ones
        for name in self.lora_param_names:
            if name in lora_weights:
                full_params[name] = lora_weights[name]

        return full_params

    def forward_with_lora(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        lora_weights: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass with externally provided LoRA weights.

        This is the core differentiable operation!

        Args:
            input_ids: Input token IDs (B, seq_len)
            attention_mask: Attention mask (B, seq_len)
            lora_weights: Generated LoRA parameters (differentiable)

        Returns:
            Hidden states from last layer, last token (B, hidden_size)
        """
        # Build full parameter dict
        full_params = self.build_full_params(lora_weights)

        # Use functional_call for differentiable forward
        outputs = functional_call(
            self.peft_model,
            full_params,
            kwargs={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "output_hidden_states": True,
            },
        )

        # Extract last layer, last token hidden state
        hidden_states = outputs.hidden_states[-1]  # (B, seq_len, hidden_size)

        # Get last token for each sequence
        # Handle variable length sequences properly
        seq_lengths = attention_mask.sum(dim=1) - 1  # Last valid position
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_hidden = hidden_states[batch_indices, seq_lengths]  # (B, hidden_size)

        return last_hidden.float()  # Ensure float32 for loss stability


class DifferentiableDeltaCompute(nn.Module):
    """
    Module that computes delta embeddings differentiably.

    Given generated LoRA weights, computes:
        delta = v(base + lora) - v(base)

    where v() is the average probe activation.
    """

    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-1.5B",
        lora_config: Optional[LoraConfig] = None,
        probes: Optional[List[str]] = None,
        device: torch.device = None,
        dtype: torch.dtype = torch.float16,
        max_length: int = 256,
    ):
        super().__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        # Default LoRA config for Qwen2.5-1.5B
        if lora_config is None:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
            )

        # Initialize functional model
        self.functional_model = FunctionalLoRAModel(
            base_model_name, lora_config, self.device, dtype
        )

        # Set up probes
        if probes is None:
            from llgbm.probes import create_generic_probes
            probes = create_generic_probes()
        self.probes = probes

        # Pre-tokenize probes
        self.probe_inputs = self._tokenize_probes()

        # Cache base activation (computed once)
        self._base_activation = None

    def _tokenize_probes(self) -> Dict[str, torch.Tensor]:
        """Tokenize all probes."""
        encoded = self.functional_model.tokenizer(
            self.probes,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device),
        }

    @torch.no_grad()
    def compute_base_activation(self) -> torch.Tensor:
        """
        Compute and cache base model activation on probes.
        This only needs to be done once.
        """
        if self._base_activation is not None:
            return self._base_activation

        # Get zero LoRA weights (equivalent to base model)
        zero_lora = {
            name: torch.zeros_like(param)
            for name, param in self.functional_model.get_lora_state_dict().items()
        }

        # Compute activation
        hidden = self.functional_model.forward_with_lora(
            self.probe_inputs["input_ids"],
            self.probe_inputs["attention_mask"],
            zero_lora,
        )

        # Average over probes
        self._base_activation = hidden.mean(dim=0)  # (hidden_size,)
        return self._base_activation

    def forward(
        self,
        lora_weights: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute delta embedding for generated LoRA weights.

        Args:
            lora_weights: Generated LoRA parameters (must be differentiable!)

        Returns:
            Delta embedding (hidden_size,) with gradients
        """
        # Ensure base activation is computed
        base_act = self.compute_base_activation()

        # Compute activation with LoRA applied
        adapted_hidden = self.functional_model.forward_with_lora(
            self.probe_inputs["input_ids"],
            self.probe_inputs["attention_mask"],
            lora_weights,
        )

        # Average over probes
        adapted_act = adapted_hidden.mean(dim=0)  # (hidden_size,)

        # Compute delta
        delta = adapted_act - base_act

        return delta

    def forward_batch(
        self,
        lora_weights_batch: List[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Compute delta embeddings for a batch of generated LoRAs.

        Args:
            lora_weights_batch: List of LoRA weight dicts, one per sample

        Returns:
            Delta embeddings (B, hidden_size)
        """
        deltas = []
        for lora_weights in lora_weights_batch:
            delta = self.forward(lora_weights)
            deltas.append(delta)

        return torch.stack(deltas)
```

### Step 2: Create LoRA Detokenization Bridge

Create `llgbm/detokenize.py`:

```python
"""Bridge between DnD tokens and PEFT parameter names."""
import sys
sys.path.insert(0, "Drag-and-Drop-LLMs")

import torch
from typing import Dict, Tuple
from workspace.dnd.tokenizer.register import Qwen2515LoRA_Tokenizer2D


class DifferentiableDetokenizer:
    """
    Detokenizes generator output to LoRA weight dicts while preserving gradients.

    The DnD tokenizer uses numpy internally, which breaks gradients.
    This class reimplements detokenization in pure PyTorch.
    """

    def __init__(self, lora_tokenizer: Qwen2515LoRA_Tokenizer2D = None):
        """
        Args:
            lora_tokenizer: DnD tokenizer (used for shape/config reference)
        """
        self.lora_tokenizer = lora_tokenizer or Qwen2515LoRA_Tokenizer2D()

        # Build mapping from token indices to parameter names
        self.token_to_param = self._build_token_mapping()

    def _build_token_mapping(self) -> Dict[int, Dict]:
        """
        Build mapping from token indices to parameter info.

        This maps DnD's token representation to PEFT's parameter naming.
        """
        # This needs to be implemented based on the specific tokenizer config
        # For Qwen2.5-1.5B, we need to map each token to:
        # - Parameter name in PEFT model
        # - Slice indices within that parameter
        # - Scale factors for denormalization

        # Placeholder - actual implementation depends on tokenizer internals
        mapping = {}

        # Example structure:
        # mapping[token_idx] = {
        #     "param_name": "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
        #     "row_start": 0,
        #     "row_end": 16,
        #     "col_start": 0,
        #     "col_end": 256,
        # }

        return mapping

    def detokenize(
        self,
        tokens: torch.Tensor,
        scales: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Convert generator output tokens to LoRA weight dict.

        Args:
            tokens: Generated tokens (num_tokens, H, W) - must have gradients!
            scales: Normalization scales for reconstruction

        Returns:
            Dict mapping PEFT parameter names to weight tensors (with gradients)
        """
        # For now, use the original tokenizer but ensure we maintain gradients
        # by using the detokenized shapes as targets

        # Option 1: Simple approach - reshape tokens directly
        # This assumes tokens can be directly mapped to LoRA params

        lora_weights = {}

        # Get reference shapes from tokenizer
        # This is a simplified version - actual implementation needs
        # to match the tokenizer's exact mapping

        # Example for a single layer:
        # tokens shape: (num_tokens, 18, 258)
        # We need to extract and reshape into LoRA A and B matrices

        return lora_weights


def create_detokenizer_from_peft(
    peft_model,
    lora_tokenizer: Qwen2515LoRA_Tokenizer2D,
) -> Dict[str, Tuple[int, int, int, int]]:
    """
    Create a mapping from tokens to PEFT parameters by analyzing both.

    Args:
        peft_model: A PEFT model with LoRA adapters
        lora_tokenizer: DnD tokenizer

    Returns:
        Mapping from token indices to (param_name, row_slice, col_slice)
    """
    # Get all LoRA param names and shapes from PEFT model
    peft_params = {}
    for name, param in peft_model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            peft_params[name] = param.shape

    # Get tokenizer's internal mapping
    # This requires understanding the tokenizer's _layer_types and _chunks

    # Build bidirectional mapping
    mapping = {}

    return mapping


class SimpleLoRAReconstructor(torch.nn.Module):
    """
    Simple learnable mapping from generator tokens to LoRA weights.

    Instead of exact reconstruction, learn a linear mapping that
    produces valid LoRA weights while preserving gradients.
    """

    def __init__(
        self,
        token_shape: Tuple[int, int, int],  # (num_tokens, H, W)
        lora_shapes: Dict[str, Tuple[int, int]],  # param_name -> (rows, cols)
    ):
        super().__init__()

        self.token_shape = token_shape
        self.lora_shapes = lora_shapes

        # Flatten token dimension
        token_dim = token_shape[0] * token_shape[1] * token_shape[2]

        # Create projection for each LoRA parameter
        self.projections = torch.nn.ModuleDict()
        for name, shape in lora_shapes.items():
            safe_name = name.replace(".", "_")
            param_dim = shape[0] * shape[1]
            self.projections[safe_name] = torch.nn.Linear(token_dim, param_dim)

        self._name_mapping = {name.replace(".", "_"): name for name in lora_shapes}

    def forward(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Convert tokens to LoRA weights.

        Args:
            tokens: (num_tokens, H, W) or (B, num_tokens, H, W)

        Returns:
            Dict of LoRA parameters with proper shapes
        """
        # Handle batch dimension
        if tokens.dim() == 4:
            batch_size = tokens.size(0)
            tokens_flat = tokens.view(batch_size, -1)  # (B, token_dim)
        else:
            tokens_flat = tokens.view(1, -1)  # (1, token_dim)
            batch_size = 1

        lora_weights = {}
        for safe_name, proj in self.projections.items():
            original_name = self._name_mapping[safe_name]
            shape = self.lora_shapes[original_name]

            # Project and reshape
            weight = proj(tokens_flat)  # (B, param_dim)
            weight = weight.view(batch_size, *shape)  # (B, rows, cols)

            if batch_size == 1:
                weight = weight.squeeze(0)  # (rows, cols)

            lora_weights[original_name] = weight

        return lora_weights
```

### Step 3: Create Gradient Sanity Test

Create `scripts/test_gradient_flow.py`:

```python
"""Test that gradients flow from delta loss to generator."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "Drag-and-Drop-LLMs")

import torch
import torch.nn as nn

from llgbm.functional_lora import DifferentiableDeltaCompute
from llgbm.probes import create_generic_probes


def test_gradient_flow_simple():
    """
    Simple test: verify gradients flow through functional_call.
    """
    print("Testing gradient flow through functional_call...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create delta compute module
    delta_compute = DifferentiableDeltaCompute(
        base_model_name="Qwen/Qwen2.5-1.5B",
        device=device,
        dtype=torch.float16,
    )

    # Get reference LoRA shapes
    ref_lora = delta_compute.functional_model.get_lora_state_dict()
    print(f"LoRA parameters: {len(ref_lora)}")

    # Create dummy "generated" LoRA weights with gradients
    generated_lora = {}
    for name, ref_tensor in ref_lora.items():
        # Random initialization with requires_grad=True
        generated_lora[name] = torch.randn_like(ref_tensor, requires_grad=True) * 0.01

    # Compute delta
    delta = delta_compute(generated_lora)
    print(f"Delta shape: {delta.shape}")

    # Create dummy target
    target_delta = torch.randn_like(delta)

    # Compute loss
    loss = nn.functional.mse_loss(delta, target_delta)
    print(f"Loss: {loss.item():.6f}")

    # Backward pass
    loss.backward()

    # Check gradients
    has_grad = []
    for name, param in generated_lora.items():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            has_grad.append((name, grad_norm))

    print(f"\nParameters with gradients: {len(has_grad)}/{len(generated_lora)}")

    if has_grad:
        print("\nSample gradient norms:")
        for name, norm in has_grad[:5]:
            print(f"  {name}: {norm:.6f}")
        print("[PASS] Gradients flow through functional_call!")
    else:
        print("[FAIL] No gradients detected!")

    return len(has_grad) == len(generated_lora)


def test_gradient_flow_with_generator():
    """
    Test gradient flow from delta loss through a mock generator.
    """
    print("\n" + "="*50)
    print("Testing gradient flow with mock generator...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Mock generator (simple MLP)
    class MockGenerator(nn.Module):
        def __init__(self, condition_dim, lora_shapes):
            super().__init__()
            self.lora_shapes = lora_shapes

            # Calculate total output size
            total_params = sum(s[0] * s[1] for s in lora_shapes.values())

            self.net = nn.Sequential(
                nn.Linear(condition_dim, 512),
                nn.ReLU(),
                nn.Linear(512, total_params),
            )

            # Store split sizes for reconstruction
            self.split_sizes = [s[0] * s[1] for s in lora_shapes.values()]
            self.param_names = list(lora_shapes.keys())

        def forward(self, condition):
            flat_output = self.net(condition)
            splits = torch.split(flat_output, self.split_sizes, dim=-1)

            lora_weights = {}
            for name, split, shape in zip(
                self.param_names, splits, self.lora_shapes.values()
            ):
                lora_weights[name] = split.view(*shape)

            return lora_weights

    # Create delta compute
    delta_compute = DifferentiableDeltaCompute(
        base_model_name="Qwen/Qwen2.5-1.5B",
        device=device,
        dtype=torch.float16,
    )

    # Get LoRA shapes
    ref_lora = delta_compute.functional_model.get_lora_state_dict()
    lora_shapes = {name: tuple(param.shape) for name, param in ref_lora.items()}

    # Reduce to first few layers for testing
    lora_shapes = dict(list(lora_shapes.items())[:8])

    # Create mock generator
    generator = MockGenerator(condition_dim=768, lora_shapes=lora_shapes).to(device)

    # Create condition
    condition = torch.randn(768, device=device)

    # Forward through generator
    generated_lora = generator(condition)

    # Fill in remaining LoRA params with zeros (non-differentiable)
    full_lora = {name: torch.zeros_like(param) for name, param in ref_lora.items()}
    full_lora.update(generated_lora)

    # Compute delta
    delta = delta_compute(full_lora)

    # Dummy target
    target_delta = torch.randn_like(delta)

    # Loss and backward
    loss = nn.functional.mse_loss(delta, target_delta)
    loss.backward()

    # Check generator gradients
    generator_grads = []
    for name, param in generator.named_parameters():
        if param.grad is not None:
            generator_grads.append((name, param.grad.norm().item()))

    print(f"Generator parameters with gradients: {len(generator_grads)}")
    for name, norm in generator_grads:
        print(f"  {name}: {norm:.6f}")

    if generator_grads:
        print("\n[PASS] Gradients reach generator parameters!")
        return True
    else:
        print("\n[FAIL] Gradients do not reach generator!")
        return False


def test_delta_changes_with_lora():
    """
    Sanity check: verify that delta actually changes when LoRA weights change.
    """
    print("\n" + "="*50)
    print("Testing delta sensitivity to LoRA weights...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    delta_compute = DifferentiableDeltaCompute(
        base_model_name="Qwen/Qwen2.5-1.5B",
        device=device,
        dtype=torch.float16,
    )

    ref_lora = delta_compute.functional_model.get_lora_state_dict()

    # Compute delta with zero LoRA (should be ~0)
    zero_lora = {name: torch.zeros_like(param) for name, param in ref_lora.items()}
    delta_zero = delta_compute(zero_lora)
    print(f"Delta with zero LoRA - norm: {delta_zero.norm().item():.6f}")

    # Compute delta with random LoRA
    random_lora = {name: torch.randn_like(param) * 0.1 for name, param in ref_lora.items()}
    delta_random = delta_compute(random_lora)
    print(f"Delta with random LoRA - norm: {delta_random.norm().item():.6f}")

    # Compute delta with scaled random LoRA
    scaled_lora = {name: param * 2 for name, param in random_lora.items()}
    delta_scaled = delta_compute(scaled_lora)
    print(f"Delta with 2x LoRA - norm: {delta_scaled.norm().item():.6f}")

    # Check that deltas are different
    diff_zero_random = (delta_zero - delta_random).norm().item()
    diff_random_scaled = (delta_random - delta_scaled).norm().item()

    print(f"\nDifference (zero vs random): {diff_zero_random:.6f}")
    print(f"Difference (random vs scaled): {diff_random_scaled:.6f}")

    if diff_zero_random > 0.01 and diff_random_scaled > 0.01:
        print("\n[PASS] Delta is sensitive to LoRA weights!")
        return True
    else:
        print("\n[WARN] Delta may not be sensitive enough to LoRA weights")
        return False


if __name__ == "__main__":
    results = []
    results.append(("Gradient flow (simple)", test_gradient_flow_simple()))
    results.append(("Gradient flow (generator)", test_gradient_flow_with_generator()))
    results.append(("Delta sensitivity", test_delta_changes_with_lora()))

    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: [{status}]")
```

### Step 4: Alternative Approach - Custom Functional LoRA

Create `llgbm/functional_lora_custom.py`:

```python
"""
Custom functional LoRA injection without PEFT dependency.

Backup approach if functional_call + PEFT proves brittle.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM


class FunctionalLoRALinear:
    """
    Functional implementation of LoRA for a single linear layer.

    Computes: y = xW^T + s * (x @ A^T) @ B^T

    where:
    - W is the frozen base weight
    - A is LoRA down projection (rank x in_features)
    - B is LoRA up projection (out_features x rank)
    - s is the scaling factor (alpha / rank)
    """

    @staticmethod
    def forward(
        x: torch.Tensor,
        base_weight: torch.Tensor,
        base_bias: Optional[torch.Tensor],
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
        scaling: float = 1.0,
    ) -> torch.Tensor:
        """
        Forward pass with LoRA applied.

        Args:
            x: Input tensor (..., in_features)
            base_weight: Frozen base weight (out_features, in_features)
            base_bias: Optional bias (out_features,)
            lora_A: LoRA A matrix (rank, in_features)
            lora_B: LoRA B matrix (out_features, rank)
            scaling: LoRA scaling factor (alpha / rank)

        Returns:
            Output tensor (..., out_features)
        """
        # Base linear
        result = F.linear(x, base_weight, base_bias)

        # LoRA delta: x @ A^T @ B^T
        lora_output = x @ lora_A.T @ lora_B.T
        result = result + scaling * lora_output

        return result


class FunctionalLoRAModel(nn.Module):
    """
    Custom functional LoRA model that intercepts specific linear layers.

    This approach gives more control than using PEFT + functional_call.
    """

    def __init__(
        self,
        base_model_name: str,
        target_modules: List[str],
        lora_rank: int = 16,
        lora_alpha: int = 32,
        device: torch.device = None,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()

        self.device = device or torch.device("cuda")
        self.target_modules = target_modules
        self.scaling = lora_alpha / lora_rank

        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.base_model.config.output_hidden_states = True

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Identify target layers and cache their info
        self.target_layers = self._identify_target_layers()

    def _identify_target_layers(self) -> Dict[str, Dict]:
        """Identify layers that will receive LoRA."""
        layers = {}

        for name, module in self.base_model.named_modules():
            for target in self.target_modules:
                if target in name and isinstance(module, nn.Linear):
                    layers[name] = {
                        "module": module,
                        "in_features": module.in_features,
                        "out_features": module.out_features,
                        "weight": module.weight,
                        "bias": module.bias,
                    }

        return layers

    def get_lora_param_shapes(self) -> Dict[str, Tuple[int, int]]:
        """Get expected shapes for LoRA parameters."""
        shapes = {}
        for name, info in self.target_layers.items():
            # A: (rank, in_features), B: (out_features, rank)
            shapes[f"{name}.lora_A"] = (16, info["in_features"])  # Default rank=16
            shapes[f"{name}.lora_B"] = (info["out_features"], 16)
        return shapes

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        lora_weights: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass with custom LoRA injection.

        This uses hooks to intercept and modify layer outputs.
        """
        # Register forward hooks to apply LoRA
        handles = []
        lora_outputs = {}

        def make_hook(layer_name):
            def hook(module, input, output):
                lora_A = lora_weights.get(f"{layer_name}.lora_A")
                lora_B = lora_weights.get(f"{layer_name}.lora_B")

                if lora_A is not None and lora_B is not None:
                    x = input[0]
                    lora_delta = x @ lora_A.T @ lora_B.T
                    return output + self.scaling * lora_delta
                return output
            return hook

        # Register hooks
        for name, info in self.target_layers.items():
            handle = info["module"].register_forward_hook(make_hook(name))
            handles.append(handle)

        try:
            # Forward pass
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        finally:
            # Remove hooks
            for handle in handles:
                handle.remove()

        # Extract last layer, last token
        hidden_states = outputs.hidden_states[-1]
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_hidden = hidden_states[batch_indices, seq_lengths]

        return last_hidden.float()
```

## File Structure After Phase 3

```
llgbm/
├── llgbm/
│   ├── __init__.py
│   ├── probes.py
│   ├── delta.py
│   ├── dataset.py
│   ├── dataset_wrapper.py
│   ├── functional_lora.py        # New: Main implementation
│   ├── functional_lora_custom.py # New: Backup approach
│   └── detokenize.py             # New: Token to weight bridge
├── scripts/
│   └── test_gradient_flow.py     # New: Gradient sanity tests
└── ...
```

## Acceptance Criteria

- [ ] `functional_call` works with PEFT model
- [ ] Gradients flow from MSE loss on delta to LoRA parameters
- [ ] Gradients reach generator parameters (not just LoRA params)
- [ ] Delta changes meaningfully when LoRA weights change
- [ ] No GPU memory leaks during repeated forward passes
- [ ] Works with mixed precision (float16/bfloat16)

## Memory Optimization Tips

```python
# Use gradient checkpointing
model.gradient_checkpointing_enable()

# Only compute delta on subset of batch
delta_batch_size = 2  # Even if main batch is larger
delta_loss = compute_delta_loss(batch[:delta_batch_size])

# Cache base activation (computed once)
base_act = delta_compute.compute_base_activation()  # No gradients

# Use smaller probe set
probes = create_generic_probes()[:3]  # 3 instead of 5
```

## Next Phase
Proceed to **Phase 4** to integrate delta loss into the training loop.
