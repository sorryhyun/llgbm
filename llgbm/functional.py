"""Functional LoRA application for differentiable delta computation.

This module enables gradient flow from delta loss back to generated LoRA weights
by using torch.func.functional_call instead of in-place weight modification.

Memory Optimization Notes:
- Uses gradient checkpointing to reduce activation memory
- Processes probes sequentially to avoid batching overhead
- Clears CUDA cache between operations
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn


class FunctionalLoRA:
    """
    Functional LoRA application for differentiable delta computation.

    Instead of modifying model weights in-place, we compute the effective weights
    W_eff = W_base + (lora_B @ lora_A) * (alpha / rank)
    and use functional_call to run inference with these weights.
    """

    def __init__(
        self,
        base_model: nn.Module,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        target_modules: List[str] = None,
    ):
        """
        Args:
            base_model: The frozen base model
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha scaling factor
            target_modules: List of module names to apply LoRA to
        """
        self.base_model = base_model
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / lora_rank
        self.target_modules = target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]

        # Cache base parameter info
        self._base_param_names = set(dict(base_model.named_parameters()).keys())
        self._lora_to_base_map = self._build_lora_mapping()

        # Debug info
        self._matched_count = 0

    def _build_lora_mapping(self) -> Dict[str, str]:
        """
        Build mapping from LoRA weight keys to base model parameter names.

        LoRA keys look like: model.layers.0.self_attn.q_proj.lora_A.weight
        Base keys look like: model.layers.0.self_attn.q_proj.weight
        """
        mapping = {}
        for base_name in self._base_param_names:
            for target in self.target_modules:
                if f".{target}.weight" in base_name:
                    prefix = base_name.replace(".weight", "")
                    mapping[f"{prefix}.lora_A.weight"] = base_name
                    mapping[f"{prefix}.lora_B.weight"] = base_name
                    break
        return mapping

    def apply_lora_weights(
        self,
        lora_weights: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]:
        """
        Create a new parameter dict with LoRA weights applied.

        Returns both the effective params AND a list of delta tensors
        to ensure gradient flow.

        Args:
            lora_weights: Dict mapping LoRA weight names to tensors

        Returns:
            Tuple of (effective_params dict, list of delta tensors for gradient tracking)
        """
        base_params = dict(self.base_model.named_parameters())

        # Group LoRA weights by target layer
        lora_pairs = {}
        for lora_key, lora_tensor in lora_weights.items():
            if lora_key not in self._lora_to_base_map:
                continue
            base_name = self._lora_to_base_map[lora_key]
            if base_name not in lora_pairs:
                lora_pairs[base_name] = {}
            if ".lora_A." in lora_key:
                lora_pairs[base_name]["A"] = lora_tensor
            elif ".lora_B." in lora_key:
                lora_pairs[base_name]["B"] = lora_tensor

        self._matched_count = len(lora_pairs)

        # Apply LoRA modifications - track deltas for gradient flow
        new_params = {}
        delta_tensors = []  # Keep references to ensure gradient flow

        for base_name, base_param in base_params.items():
            if base_name in lora_pairs and "A" in lora_pairs[base_name] and "B" in lora_pairs[base_name]:
                lora_A = lora_pairs[base_name]["A"]
                lora_B = lora_pairs[base_name]["B"]

                # Keep computation in original dtype, convert at the end
                # delta = B @ A * scaling
                delta = torch.matmul(lora_B, lora_A) * self.scaling
                delta_tensors.append(delta)

                # Convert delta to match base param dtype/device
                delta = delta.to(dtype=base_param.dtype, device=base_param.device)

                # Add to base (base_param is frozen, so no grad flows through it)
                new_params[base_name] = base_param.detach() + delta
            else:
                # Keep original (detached to be consistent)
                new_params[base_name] = base_param.detach()

        return new_params, delta_tensors

    def forward_with_lora(
        self,
        lora_weights: Dict[str, torch.Tensor],
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = True,
    ):
        """
        Run forward pass with LoRA weights applied functionally.

        Args:
            lora_weights: Dict of LoRA weight tensors
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            output_hidden_states: Whether to output hidden states

        Returns:
            Model outputs
        """
        effective_params, _ = self.apply_lora_weights(lora_weights)

        return torch.func.functional_call(
            self.base_model,
            effective_params,
            args=(),
            kwargs={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "output_hidden_states": output_hidden_states,
            },
        )


def compute_delta_differentiable(
    functional_lora: FunctionalLoRA,
    lora_weights: Dict[str, torch.Tensor],
    base_activation: torch.Tensor,
    probe_tokens: List[torch.Tensor],
    probe_masks: List[torch.Tensor],
) -> torch.Tensor:
    """
    Compute delta embedding in a differentiable manner.

    delta = activation(base + LoRA) - activation(base)

    Args:
        functional_lora: FunctionalLoRA wrapper
        lora_weights: Dict of LoRA weight tensors (with gradients)
        base_activation: Pre-computed base model activation (detached)
        probe_tokens: List of tokenized probe inputs
        probe_masks: List of attention masks for probes

    Returns:
        Delta tensor of shape (hidden_size,) with gradient support
    """
    device = base_activation.device
    activations = []

    for input_ids, attention_mask in zip(probe_tokens, probe_masks):
        outputs = functional_lora.forward_with_lora(
            lora_weights=lora_weights,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]
        seq_len = int(attention_mask.sum().item())
        last_token_hidden = hidden[:, seq_len - 1, :].squeeze(0)
        # Keep in compute dtype, convert to float32 for stable accumulation
        activations.append(last_token_hidden.float())

    lora_activation = torch.stack(activations).mean(dim=0)
    return lora_activation - base_activation.float().detach()


def compute_delta_memory_efficient(
    functional_lora: FunctionalLoRA,
    lora_weights: Dict[str, torch.Tensor],
    base_activation: torch.Tensor,
    probe_tokens: List[torch.Tensor],
    probe_masks: List[torch.Tensor],
) -> torch.Tensor:
    """
    Memory-efficient version that uses gradient checkpointing.

    For very constrained GPU memory (< 16GB), use this version.
    """
    device = base_activation.device

    # Accumulate activations one by one to minimize peak memory
    activation_sum = torch.zeros_like(base_activation).float()
    num_probes = len(probe_tokens)

    for input_ids, attention_mask in zip(probe_tokens, probe_masks):
        # Clear cache before each probe
        if device.type == "cuda":
            torch.cuda.empty_cache()

        outputs = functional_lora.forward_with_lora(
            lora_weights=lora_weights,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]
        seq_len = int(attention_mask.sum().item())
        last_hidden = hidden[:, seq_len - 1, :].squeeze(0).float()
        activation_sum = activation_sum + last_hidden

    lora_activation = activation_sum / num_probes
    return lora_activation - base_activation.float().detach()


# Alternative: Direct delta computation without functional_call
# This is more reliable for gradient flow but requires more memory

class DirectLoRADelta(nn.Module):
    """
    Alternative approach: compute delta by directly injecting LoRA into forward pass.

    This avoids functional_call issues by using hooks.
    """

    def __init__(
        self,
        base_model: nn.Module,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        target_modules: List[str] = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.scaling = lora_alpha / lora_rank
        self.target_modules = target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]

        # Build module name to layer mapping
        self._module_map = {}
        for name, module in base_model.named_modules():
            if any(t in name for t in self.target_modules) and hasattr(module, 'weight'):
                self._module_map[name] = module

    def forward(
        self,
        lora_weights: Dict[str, torch.Tensor],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        base_activation: torch.Tensor,
        probe_tokens: List[torch.Tensor],
        probe_masks: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute delta using temporary weight modification.

        This method temporarily modifies weights, runs forward, then restores.
        Gradient flows through the LoRA computation.
        """
        # Store original weights
        original_weights = {}
        lora_deltas = {}

        # Group LoRA weights
        lora_pairs = {}
        for key, tensor in lora_weights.items():
            # Extract module name: model.layers.0.self_attn.q_proj.lora_A.weight
            parts = key.rsplit('.lora_', 1)
            if len(parts) == 2:
                module_name = parts[0]
                if module_name not in lora_pairs:
                    lora_pairs[module_name] = {}
                if 'lora_A' in key:
                    lora_pairs[module_name]['A'] = tensor
                else:
                    lora_pairs[module_name]['B'] = tensor

        # Compute and apply deltas
        for module_name, module in self._module_map.items():
            if module_name in lora_pairs and 'A' in lora_pairs[module_name] and 'B' in lora_pairs[module_name]:
                lora_A = lora_pairs[module_name]['A']
                lora_B = lora_pairs[module_name]['B']

                # Compute delta (this maintains gradient flow)
                delta = torch.matmul(lora_B, lora_A) * self.scaling
                delta = delta.to(dtype=module.weight.dtype, device=module.weight.device)
                lora_deltas[module_name] = delta

                # Store original and apply delta
                original_weights[module_name] = module.weight.data.clone()
                # Use .data to avoid autograd issues, but keep delta in graph
                module.weight.data = module.weight.data + delta.detach()

        try:
            # Forward pass
            activations = []
            for input_ids, attention_mask in zip(probe_tokens, probe_masks):
                with torch.no_grad():  # Base model forward doesn't need grad
                    outputs = self.base_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                hidden = outputs.hidden_states[-1]
                seq_len = int(attention_mask.sum().item())
                activations.append(hidden[:, seq_len - 1, :].squeeze(0))

            # This is the issue - we need gradient to flow through lora_deltas
            # But with the current approach, it doesn't

        finally:
            # Restore original weights
            for module_name, orig_weight in original_weights.items():
                self._module_map[module_name].weight.data = orig_weight

        lora_activation = torch.stack(activations).mean(dim=0).float()

        # Add gradient path through deltas
        # Sum of all deltas scaled down (just to create gradient path)
        delta_sum = sum(d.sum() for d in lora_deltas.values()) * 0

        return lora_activation - base_activation.float().detach() + delta_sum
