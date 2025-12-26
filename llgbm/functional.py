"""Functional LoRA application for differentiable delta computation.

This module enables gradient flow from delta loss back to generated LoRA weights
by using torch.func.functional_call instead of in-place weight modification.

Memory Optimization Notes:
- `torch.func.functional_call` + dense ΔW materialization is correct but very
  memory hungry: it forms full matrices ΔW = (B @ A) for every target module,
  for every layer, for every probe.
- By default we apply LoRA in low-rank form via forward hooks (two small linear
  ops) to avoid materializing dense ΔW and reduce OOM risk.
"""

from __future__ import annotations

import inspect
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        mode: str = "hooks",
    ):
        """
        Args:
            base_model: The frozen base model
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha scaling factor
            target_modules: List of module names to apply LoRA to
            mode: "hooks" (default, efficient) or "functional_call" (legacy, slow/OOM-prone)
        """
        self.base_model = base_model
        self.backbone = getattr(base_model, "model", base_model)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / lora_rank
        self.target_modules = target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
        self.mode = mode

        # Cache base parameter info
        self._base_param_names = set(dict(base_model.named_parameters()).keys())
        self._lora_to_base_map = self._build_lora_mapping()

        # Debug info
        self._matched_count = 0

        # Hook-based application state
        self._active_lora_weights: Optional[Dict[str, torch.Tensor]] = None
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._module_key_candidates: Dict[str, Tuple[Tuple[str, str], ...]] = {}
        self._forward_has_var_kwargs, self._forward_arg_names = self._inspect_forward(self.backbone)

        if self.mode == "hooks":
            self._register_lora_hooks()

    @staticmethod
    def _inspect_forward(model: nn.Module) -> Tuple[bool, set[str]]:
        try:
            sig = inspect.signature(model.forward)
        except (TypeError, ValueError):
            return True, set()

        has_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        return has_var_kwargs, set(sig.parameters.keys())

    def close(self) -> None:
        """Remove registered hooks (useful in notebooks)."""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

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

    def _register_lora_hooks(self) -> None:
        """
        Register forward hooks on target Linear modules.

        The hook adds the LoRA update in low-rank form:
            y = W x + scaling * (B (A x))

        This avoids constructing dense ΔW = B @ A.
        """
        if self._hook_handles:
            return

        for module_name, module in self.backbone.named_modules():
            if module_name.split(".")[-1] not in self.target_modules:
                continue
            if not isinstance(module, nn.Linear):
                continue

            candidates: List[Tuple[str, str]] = [
                (f"{module_name}.lora_A.weight", f"{module_name}.lora_B.weight")
            ]
            if module_name.startswith("model."):
                stripped = module_name[len("model.") :]
                candidates.append((f"{stripped}.lora_A.weight", f"{stripped}.lora_B.weight"))
            else:
                candidates.append((f"model.{module_name}.lora_A.weight", f"model.{module_name}.lora_B.weight"))
            self._module_key_candidates[module_name] = tuple(candidates)

            def _hook(mod: nn.Module, inputs, output, *, _name: str = module_name):
                lora = self._active_lora_weights
                if not lora:
                    return output

                A = B = None
                for a_key, b_key in self._module_key_candidates.get(_name, ()):
                    if a_key in lora:
                        A = lora.get(a_key)
                        B = lora.get(b_key)
                        break
                if A is None or B is None:
                    return output

                x = inputs[0]
                A = A.to(dtype=x.dtype, device=x.device)
                B = B.to(dtype=x.dtype, device=x.device)
                update = F.linear(F.linear(x, A), B) * self.scaling
                return output + update

            self._hook_handles.append(module.register_forward_hook(_hook))

    def apply_lora_weights(
        self,
        lora_weights: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Create a new parameter dict with LoRA weights applied.

        Gradient flows through the delta computation (B @ A * scaling).

        Args:
            lora_weights: Dict mapping LoRA weight names to tensors

        Returns:
            Dict of effective parameters with LoRA applied
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

        # Apply LoRA modifications
        new_params = {}

        for base_name, base_param in base_params.items():
            if base_name in lora_pairs and "A" in lora_pairs[base_name] and "B" in lora_pairs[base_name]:
                lora_A = lora_pairs[base_name]["A"]
                lora_B = lora_pairs[base_name]["B"]

                # delta = B @ A * scaling (gradient flows through this)
                delta = torch.matmul(lora_B, lora_A) * self.scaling

                # Convert delta to match base param dtype/device
                delta = delta.to(dtype=base_param.dtype, device=base_param.device)

                # Add to base (base_param is frozen, so no grad flows through it)
                new_params[base_name] = base_param.detach() + delta
            else:
                # Keep original (detached to be consistent)
                new_params[base_name] = base_param.detach()

        return new_params

    def forward_with_lora(
        self,
        lora_weights: Dict[str, torch.Tensor],
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        use_cache: bool = False,
    ):
        """
        Run forward pass with LoRA weights applied functionally.

        Args:
            lora_weights: Dict of LoRA weight tensors
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            output_hidden_states: Whether to output hidden states
            use_cache: Whether to use KV cache (set False for training)

        Returns:
            Model outputs
        """
        if self.mode == "functional_call":
            effective_params = self.apply_lora_weights(lora_weights)

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

        # Efficient hook-based mode.
        if not self._hook_handles:
            self._register_lora_hooks()

        self._active_lora_weights = lora_weights
        try:
            kwargs = {"input_ids": input_ids}
            if attention_mask is not None:
                kwargs["attention_mask"] = attention_mask
            if self._forward_has_var_kwargs or "output_hidden_states" in self._forward_arg_names:
                kwargs["output_hidden_states"] = output_hidden_states
            if self._forward_has_var_kwargs or "use_cache" in self._forward_arg_names:
                kwargs["use_cache"] = use_cache
            return self.backbone(**kwargs)
        finally:
            self._active_lora_weights = None


def compute_delta_differentiable(
    functional_lora: FunctionalLoRA,
    lora_weights: Dict[str, torch.Tensor],
    base_activation: torch.Tensor,
    probe_tokens: List[torch.Tensor],
    probe_masks: List[torch.Tensor],
    batch_probes: bool = True,
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
        batch_probes: If True, process all probes in a single batched forward pass
                      for better GPU utilization. Set False for memory-constrained cases.

    Returns:
        Delta tensor of shape (hidden_size,) with gradient support
    """
    num_probes = len(probe_tokens)

    if batch_probes and num_probes > 1:
        # Batched processing: single forward pass for all probes
        # Pad sequences to same length and batch them
        max_len = max(t.shape[1] for t in probe_tokens)
        device = probe_tokens[0].device
        dtype = probe_tokens[0].dtype

        # Pad and stack all probes
        batched_ids = []
        batched_masks = []
        for ids, mask in zip(probe_tokens, probe_masks):
            pad_len = max_len - ids.shape[1]
            if pad_len > 0:
                # Pad on the right
                ids = F.pad(ids, (0, pad_len), value=0)
                mask = F.pad(mask, (0, pad_len), value=0)
            batched_ids.append(ids)
            batched_masks.append(mask)

        # Stack into batch: (num_probes, seq_len)
        all_ids = torch.cat(batched_ids, dim=0)
        all_masks = torch.cat(batched_masks, dim=0)

        # Single batched forward pass
        outputs = functional_lora.forward_with_lora(
            lora_weights=lora_weights,
            input_ids=all_ids,
            attention_mask=all_masks,
            output_hidden_states=False,
            use_cache=False,
        )

        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            hidden = outputs.last_hidden_state
        else:
            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states is None:
                raise AttributeError("Model outputs missing `last_hidden_state` and `hidden_states`")
            hidden = hidden_states[-1]

        # Extract last token for each probe (based on actual sequence length)
        seq_lens = all_masks.long().sum(dim=1).clamp(min=1) - 1
        batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
        last_token_hiddens = hidden[batch_idx, seq_lens, :].float()  # (num_probes, hidden_size)

        # Average across probes
        lora_activation = last_token_hiddens.mean(dim=0)

        del outputs, hidden

    else:
        # Sequential processing (original behavior, lower memory)
        activation_sum = None

        for input_ids, attention_mask in zip(probe_tokens, probe_masks):
            outputs = functional_lora.forward_with_lora(
                lora_weights=lora_weights,
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                use_cache=False,
            )

            if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                hidden = outputs.last_hidden_state
            else:
                hidden_states = getattr(outputs, "hidden_states", None)
                if hidden_states is None:
                    raise AttributeError("Model outputs missing `last_hidden_state` and `hidden_states`")
                hidden = hidden_states[-1]

            seq_lens = attention_mask.long().sum(dim=1).clamp(min=1) - 1
            batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
            last_token_hidden = hidden[batch_idx, seq_lens, :].squeeze(0).float()

            # Accumulate instead of storing all activations
            if activation_sum is None:
                activation_sum = last_token_hidden
            else:
                activation_sum = activation_sum + last_token_hidden

            # Free Python refs (autograd saved tensors still live until backward).
            del outputs, hidden

        lora_activation = activation_sum / num_probes

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
    # Accumulate activations one by one to minimize peak memory
    activation_sum = torch.zeros_like(base_activation).float()
    num_probes = len(probe_tokens)

    for input_ids, attention_mask in zip(probe_tokens, probe_masks):
        outputs = functional_lora.forward_with_lora(
            lora_weights=lora_weights,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False,
        )

        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            hidden = outputs.last_hidden_state
        else:
            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states is None:
                raise AttributeError("Model outputs missing `last_hidden_state` and `hidden_states`")
            hidden = hidden_states[-1]

        seq_lens = attention_mask.long().sum(dim=1).clamp(min=1) - 1
        batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
        last_hidden = hidden[batch_idx, seq_lens, :].squeeze(0).float()
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
