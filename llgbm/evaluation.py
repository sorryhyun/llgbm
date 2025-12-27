"""Evaluation utilities for LLGBM.

This module provides functions for evaluating generated LoRAs on
task-specific data using eval loss and accuracy metrics.
"""

import gc
import json
import re
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from llgbm.functional import FunctionalLoRA


# =============================================================================
# Answer Extraction Helpers
# =============================================================================

def extract_mcq_answer(text: str) -> Optional[str]:
    """Extract multiple choice answer [A], [B], [C], [D] from generated text."""
    # Try bracketed format first: [A], [B], etc.
    match = re.search(r'\[([A-Da-d])\]', text)
    if match:
        return match.group(1).upper()

    # Try standalone letter at start or after newline
    match = re.search(r'(?:^|\n)\s*([A-Da-d])(?:\s|$|\.|\)|:)', text)
    if match:
        return match.group(1).upper()

    # Try "answer is X" pattern
    match = re.search(r'answer\s+is\s+([A-Da-d])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return None


def extract_bool_answer(text: str) -> Optional[bool]:
    """Extract True/False answer from generated text."""
    text_lower = text.lower()

    # Try bracketed format: [True], [False]
    if '[true]' in text_lower:
        return True
    if '[false]' in text_lower:
        return False

    # Try standalone true/false
    if re.search(r'\btrue\b', text_lower):
        return True
    if re.search(r'\bfalse\b', text_lower):
        return False

    # Try yes/no
    if re.search(r'\byes\b', text_lower):
        return True
    if re.search(r'\bno\b', text_lower):
        return False

    return None


def extract_gsm8k_answer(text: str) -> Optional[float]:
    """Extract numeric answer from GSM8K response (after #### or final number)."""
    # Try #### format first
    match = re.search(r'####\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)', text)
    if match:
        return float(match.group(1).replace(',', ''))

    # Try "answer is X" pattern
    match = re.search(r'answer\s+is\s+([-+]?\d+(?:,\d{3})*(?:\.\d+)?)', text, re.IGNORECASE)
    if match:
        return float(match.group(1).replace(',', ''))

    # Try last number in text
    numbers = re.findall(r'[-+]?\d+(?:,\d{3})*(?:\.\d+)?', text)
    if numbers:
        return float(numbers[-1].replace(',', ''))

    return None


def extract_ground_truth(sample: Dict, task_type: str) -> Optional[str | bool | float]:
    """Extract ground truth answer from sample based on task type."""
    response = sample.get("response", "")

    if task_type == "mcq":
        match = re.search(r'\[([A-Da-d])\]', response)
        return match.group(1).upper() if match else None
    elif task_type == "bool":
        if '[True]' in response or '[true]' in response:
            return True
        elif '[False]' in response or '[false]' in response:
            return False
        return None
    elif task_type == "gsm8k":
        return extract_gsm8k_answer(response)

    return None


def format_chat_for_eval(example: Dict) -> str:
    """Format example as Qwen chat template."""
    system = example.get("system", "You are a helpful assistant.")
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{example['prompt']}<|im_end|>\n"
        f"<|im_start|>assistant\n{example['response']}<|im_end|>"
    )


def compute_eval_loss_with_lora(
    generator: nn.Module,
    functional_lora: FunctionalLoRA,
    condition_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    eval_samples: List[Dict],
    tokenizer,
    max_samples: int = 50,
    max_length: int = 384,
    device: torch.device = None,
) -> float:
    """
    Generate LoRA weights and compute eval loss on held-out data.

    Args:
        generator: Trained LoRA generator
        functional_lora: FunctionalLoRA wrapper for the base model
        condition_ids: Condition input IDs for the generator
        attention_mask: Attention mask for condition
        eval_samples: List of eval examples with prompt/response
        tokenizer: Tokenizer for the base model
        max_samples: Maximum number of samples to evaluate
        max_length: Maximum sequence length
        device: Device for computation

    Returns:
        Average cross-entropy loss on eval samples
    """
    if device is None:
        device = next(generator.parameters()).device

    generator.eval()

    # Generate LoRA weights
    with torch.no_grad():
        lora_weights_batch = generator(
            condition_ids.unsqueeze(0).to(device),
            attention_mask.unsqueeze(0).to(device)
        )
        lora_weights = lora_weights_batch[0]

    # Compute loss on eval samples
    total_loss = 0.0
    total_tokens = 0

    for sample in eval_samples[:max_samples]:
        text = format_chat_for_eval(sample)

        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        input_ids = inputs["input_ids"].to(device)
        attn_mask = inputs["attention_mask"].to(device)

        if input_ids.shape[1] < 2:
            continue

        # Forward pass with generated LoRA
        with torch.no_grad():
            outputs = functional_lora.forward_with_lora(
                lora_weights=lora_weights,
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=False,
                use_cache=False,
            )

        # Compute cross-entropy loss (shifted labels)
        logits = outputs.logits[:, :-1, :].contiguous()
        labels = input_ids[:, 1:].contiguous()

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction="sum"
        )

        num_tokens = labels.numel()
        total_loss += loss.item()
        total_tokens += num_tokens

    return total_loss / max(total_tokens, 1)


def compute_base_eval_loss(
    base_model: nn.Module,
    eval_samples: List[Dict],
    tokenizer,
    max_samples: int = 50,
    max_length: int = 384,
    device: torch.device = None,
) -> float:
    """
    Compute eval loss using base model without any LoRA.

    Args:
        base_model: Base language model
        eval_samples: List of eval examples
        tokenizer: Tokenizer for the model
        max_samples: Maximum samples to evaluate
        max_length: Maximum sequence length
        device: Device for computation

    Returns:
        Average cross-entropy loss
    """
    if device is None:
        device = next(base_model.parameters()).device

    base_model.eval()
    total_loss = 0.0
    total_tokens = 0

    for sample in eval_samples[:max_samples]:
        text = format_chat_for_eval(sample)

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        input_ids = inputs["input_ids"].to(device)
        attn_mask = inputs["attention_mask"].to(device)

        if input_ids.shape[1] < 2:
            continue

        with torch.no_grad():
            outputs = base_model(input_ids=input_ids, attention_mask=attn_mask)

        logits = outputs.logits[:, :-1, :].contiguous()
        labels = input_ids[:, 1:].contiguous()

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction="sum"
        )

        total_loss += loss.item()
        total_tokens += labels.numel()

    return total_loss / max(total_tokens, 1)


def evaluate_task_performance(
    generator: nn.Module,
    functional_lora: FunctionalLoRA,
    dataset: Dataset,
    eval_data: Dict[str, List[Dict]],
    tokenizer,
    max_eval_samples: int = 50,
) -> Dict[str, float]:
    """
    Evaluate generator on all tasks using actual eval loss.

    Args:
        generator: Trained LoRA generator
        functional_lora: FunctionalLoRA wrapper
        dataset: Dataset with task samples for conditioning
        eval_data: Dict mapping task names to eval examples
        tokenizer: Tokenizer for the base model
        max_eval_samples: Max samples per task

    Returns:
        Dict mapping task names to eval losses
    """
    generator.eval()
    results = {}

    for task_name, task_eval_data in eval_data.items():
        # Find a sample with this task to use as condition
        task_samples = [i for i, s in enumerate(dataset.samples) if s["task"] == task_name]

        if not task_samples:
            print(f"  [SKIP] {task_name}: no conditioning samples")
            continue

        # Use first sample of this task as condition
        sample = dataset[task_samples[0]]
        condition_ids = sample["condition_ids"]
        attention_mask = sample["attention_mask"]

        # Compute eval loss
        eval_loss = compute_eval_loss_with_lora(
            generator=generator,
            functional_lora=functional_lora,
            condition_ids=condition_ids,
            attention_mask=attention_mask,
            eval_samples=task_eval_data,
            tokenizer=tokenizer,
            max_samples=max_eval_samples,
        )

        results[task_name] = eval_loss
        print(f"  {task_name}: eval_loss={eval_loss:.4f}")

    return results


def evaluate_teacher_adapter_loss(
    base_model_name: str,
    adapter_path: str,
    eval_samples: List[Dict],
    tokenizer,
    torch_dtype: torch.dtype = torch.bfloat16,
    max_samples: int = 50,
    device: torch.device = None,
) -> float:
    """
    Evaluate a real (teacher) LoRA adapter on task using eval loss.

    Args:
        base_model_name: Name/path of base model
        adapter_path: Path to LoRA adapter
        eval_samples: Eval examples
        tokenizer: Tokenizer for the model
        torch_dtype: Data type for model
        max_samples: Max samples to evaluate
        device: Device for computation

    Returns:
        Average eval loss
    """
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model with adapter
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    # Compute eval loss
    total_loss = 0.0
    total_tokens = 0

    for sample in eval_samples[:max_samples]:
        text = format_chat_for_eval(sample)

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=384,
            padding=False,
        )
        input_ids = inputs["input_ids"].to(device)
        attn_mask = inputs["attention_mask"].to(device)

        if input_ids.shape[1] < 2:
            continue

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attn_mask)

        logits = outputs.logits[:, :-1, :].contiguous()
        labels = input_ids[:, 1:].contiguous()

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction="sum"
        )

        total_loss += loss.item()
        total_tokens += labels.numel()

    # Cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return total_loss / max(total_tokens, 1)


# =============================================================================
# Accuracy-based Evaluation
# =============================================================================

def format_prompt_for_generation(sample: Dict) -> str:
    """Format sample as Qwen chat template for generation (no response)."""
    system = sample.get("system", "You are a helpful assistant.")
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{sample['prompt']}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def generate_with_lora(
    functional_lora: FunctionalLoRA,
    lora_weights: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
) -> torch.Tensor:
    """
    Generate tokens using FunctionalLoRA with greedy/temperature sampling.

    Args:
        functional_lora: FunctionalLoRA wrapper
        lora_weights: Generated LoRA weights
        input_ids: Input token IDs [1, seq_len]
        attention_mask: Attention mask [1, seq_len]
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 = greedy)

    Returns:
        Generated token IDs [1, seq_len + new_tokens]
    """
    generated = input_ids.clone()
    current_mask = attention_mask.clone()

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = functional_lora.forward_with_lora(
                lora_weights=lora_weights,
                input_ids=generated,
                attention_mask=current_mask,
                output_hidden_states=False,
                use_cache=False,
            )

        # Get next token logits
        next_logits = outputs.logits[:, -1, :]

        if temperature == 0:
            next_token = next_logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(next_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        # Append token
        generated = torch.cat([generated, next_token], dim=1)
        current_mask = torch.cat([current_mask, torch.ones_like(next_token)], dim=1)

        # Check for EOS (Qwen uses 151643 or 151645)
        if next_token.item() in [151643, 151645, 2]:  # eos tokens
            break

    return generated


def compute_accuracy_with_lora(
    generator: nn.Module,
    functional_lora: FunctionalLoRA,
    condition_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    eval_samples: List[Dict],
    tokenizer,
    task_type: Literal["mcq", "bool", "gsm8k"],
    max_samples: int = 100,
    max_new_tokens: int = 64,
    device: torch.device = None,
    show_progress: bool = True,
) -> Dict[str, float]:
    """
    Generate LoRA weights and compute accuracy on test data.

    Args:
        generator: Trained LoRA generator
        functional_lora: FunctionalLoRA wrapper for the base model
        condition_ids: Condition input IDs for the generator
        attention_mask: Attention mask for condition
        eval_samples: List of eval examples with prompt/response
        tokenizer: Tokenizer for the base model
        task_type: Type of task ("mcq", "bool", "gsm8k")
        max_samples: Maximum number of samples to evaluate
        max_new_tokens: Maximum tokens to generate per sample
        device: Device for computation
        show_progress: Show progress bar

    Returns:
        Dict with accuracy, correct count, total count, and parse failures
    """
    if device is None:
        device = next(generator.parameters()).device

    generator.eval()

    # Generate LoRA weights
    with torch.no_grad():
        lora_weights_batch = generator(
            condition_ids.unsqueeze(0).to(device),
            attention_mask.unsqueeze(0).to(device)
        )
        lora_weights = lora_weights_batch[0]

    correct = 0
    total = 0
    parse_failures = 0

    samples = eval_samples[:max_samples]
    iterator = tqdm(samples, desc="Evaluating", disable=not show_progress)

    for sample in iterator:
        prompt_text = format_prompt_for_generation(sample)

        # Tokenize prompt
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=384,
            padding=False,
        )
        input_ids = inputs["input_ids"].to(device)
        attn_mask = inputs["attention_mask"].to(device)

        # Generate response
        generated_ids = generate_with_lora(
            functional_lora=functional_lora,
            lora_weights=lora_weights,
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
        )

        # Decode generated text (only new tokens)
        new_tokens = generated_ids[0, input_ids.shape[1]:]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Extract predicted answer
        if task_type == "mcq":
            pred = extract_mcq_answer(generated_text)
        elif task_type == "bool":
            pred = extract_bool_answer(generated_text)
        else:  # gsm8k
            pred = extract_gsm8k_answer(generated_text)

        # Extract ground truth
        gt = extract_ground_truth(sample, task_type)

        if pred is None:
            parse_failures += 1
            total += 1
            continue

        # Compare
        if task_type == "gsm8k":
            # Allow small float tolerance
            is_correct = gt is not None and abs(pred - gt) < 0.01
        else:
            is_correct = pred == gt

        if is_correct:
            correct += 1
        total += 1

        if show_progress:
            iterator.set_postfix(acc=f"{correct/max(total,1):.2%}")

    accuracy = correct / max(total, 1)
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "parse_failures": parse_failures,
    }


def compute_teacher_accuracy(
    base_model_name: str,
    adapter_path: str,
    eval_samples: List[Dict],
    tokenizer,
    task_type: Literal["mcq", "bool", "gsm8k"],
    torch_dtype: torch.dtype = torch.bfloat16,
    max_samples: int = 100,
    max_new_tokens: int = 64,
    device: torch.device = None,
    show_progress: bool = True,
) -> Dict[str, float]:
    """
    Evaluate a real (teacher) LoRA adapter on task using accuracy.

    Args:
        base_model_name: Name/path of base model
        adapter_path: Path to LoRA adapter
        eval_samples: Eval examples with prompt/response
        tokenizer: Tokenizer for the model
        task_type: Type of task ("mcq", "bool", "gsm8k")
        torch_dtype: Data type for model
        max_samples: Max samples to evaluate
        max_new_tokens: Max tokens to generate
        device: Device for computation
        show_progress: Show progress bar

    Returns:
        Dict with accuracy, correct count, total count, and parse failures
    """
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model with adapter
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    correct = 0
    total = 0
    parse_failures = 0

    samples = eval_samples[:max_samples]
    iterator = tqdm(samples, desc="Teacher eval", disable=not show_progress)

    for sample in iterator:
        prompt_text = format_prompt_for_generation(sample)

        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=384,
            padding=False,
        )
        input_ids = inputs["input_ids"].to(device)
        attn_mask = inputs["attention_mask"].to(device)

        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_tokens = generated_ids[0, input_ids.shape[1]:]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Extract predicted answer
        if task_type == "mcq":
            pred = extract_mcq_answer(generated_text)
        elif task_type == "bool":
            pred = extract_bool_answer(generated_text)
        else:
            pred = extract_gsm8k_answer(generated_text)

        gt = extract_ground_truth(sample, task_type)

        if pred is None:
            parse_failures += 1
            total += 1
            continue

        if task_type == "gsm8k":
            is_correct = gt is not None and abs(pred - gt) < 0.01
        else:
            is_correct = pred == gt

        if is_correct:
            correct += 1
        total += 1

        if show_progress:
            iterator.set_postfix(acc=f"{correct/max(total,1):.2%}")

    # Cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "accuracy": correct / max(total, 1),
        "correct": correct,
        "total": total,
        "parse_failures": parse_failures,
    }


def compute_base_accuracy(
    base_model: nn.Module,
    eval_samples: List[Dict],
    tokenizer,
    task_type: Literal["mcq", "bool", "gsm8k"],
    max_samples: int = 100,
    max_new_tokens: int = 64,
    device: torch.device = None,
    show_progress: bool = True,
) -> Dict[str, float]:
    """
    Evaluate base model (no LoRA) on task using accuracy.

    Args:
        base_model: Base language model
        eval_samples: Eval examples
        tokenizer: Tokenizer for the model
        task_type: Type of task
        max_samples: Max samples to evaluate
        max_new_tokens: Max tokens to generate
        device: Device for computation
        show_progress: Show progress bar

    Returns:
        Dict with accuracy, correct, total, parse_failures
    """
    if device is None:
        device = next(base_model.parameters()).device

    base_model.eval()

    correct = 0
    total = 0
    parse_failures = 0

    samples = eval_samples[:max_samples]
    iterator = tqdm(samples, desc="Base eval", disable=not show_progress)

    for sample in iterator:
        prompt_text = format_prompt_for_generation(sample)

        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=384,
            padding=False,
        )
        input_ids = inputs["input_ids"].to(device)
        attn_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            generated_ids = base_model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_tokens = generated_ids[0, input_ids.shape[1]:]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        if task_type == "mcq":
            pred = extract_mcq_answer(generated_text)
        elif task_type == "bool":
            pred = extract_bool_answer(generated_text)
        else:
            pred = extract_gsm8k_answer(generated_text)

        gt = extract_ground_truth(sample, task_type)

        if pred is None:
            parse_failures += 1
            total += 1
            continue

        if task_type == "gsm8k":
            is_correct = gt is not None and abs(pred - gt) < 0.01
        else:
            is_correct = pred == gt

        if is_correct:
            correct += 1
        total += 1

        if show_progress:
            iterator.set_postfix(acc=f"{correct/max(total,1):.2%}")

    return {
        "accuracy": correct / max(total, 1),
        "correct": correct,
        "total": total,
        "parse_failures": parse_failures,
    }


# =============================================================================
# Data Loading
# =============================================================================

def load_test_data(data_dir: str | Path) -> Dict[str, Tuple[List[Dict], str]]:
    """
    Load test datasets from data directory.

    Args:
        data_dir: Path to data directory containing test JSON files

    Returns:
        Dict mapping task name to (samples, task_type) tuple
    """
    data_dir = Path(data_dir)
    datasets = {}

    # ARC-Easy
    arc_e_path = data_dir / "ARC-e_test.json"
    if arc_e_path.exists():
        with open(arc_e_path) as f:
            datasets["arc_e"] = (json.load(f), "mcq")

    # ARC-Challenge
    arc_c_path = data_dir / "ARC-c_test.json"
    if arc_c_path.exists():
        with open(arc_c_path) as f:
            datasets["arc_c"] = (json.load(f), "mcq")

    # BoolQ
    boolq_path = data_dir / "BoolQ_test.json"
    if boolq_path.exists():
        with open(boolq_path) as f:
            datasets["boolq"] = (json.load(f), "bool")

    # GSM8K
    gsm8k_path = data_dir / "GSM8K.json"
    if gsm8k_path.exists():
        with open(gsm8k_path) as f:
            datasets["gsm8k"] = (json.load(f), "gsm8k")

    return datasets


def evaluate_all_tasks(
    generator: nn.Module,
    functional_lora: FunctionalLoRA,
    dataset,  # RealAdapterDataset
    test_data: Dict[str, Tuple[List[Dict], str]],
    tokenizer,
    max_samples_per_task: int = 100,
    device: torch.device = None,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate generator on all tasks using accuracy.

    Args:
        generator: Trained LoRA generator
        functional_lora: FunctionalLoRA wrapper
        dataset: Training dataset (to get conditioning samples)
        test_data: Dict from load_test_data()
        tokenizer: Tokenizer for base model
        max_samples_per_task: Max test samples per task
        device: Device for computation

    Returns:
        Dict mapping task names to accuracy results
    """
    results = {}

    for task_name, (samples, task_type) in test_data.items():
        print(f"\n[{task_name.upper()}] Evaluating ({len(samples)} samples, type={task_type})")

        # Find conditioning sample for this task
        task_samples = [i for i, s in enumerate(dataset.samples) if s["task"] == task_name]

        if not task_samples:
            print(f"  [SKIP] No conditioning samples for {task_name}")
            continue

        # Get condition from first matching sample
        sample = dataset[task_samples[0]]
        condition_ids = sample["condition_ids"]
        attention_mask = sample["attention_mask"]

        # Compute accuracy
        task_results = compute_accuracy_with_lora(
            generator=generator,
            functional_lora=functional_lora,
            condition_ids=condition_ids,
            attention_mask=attention_mask,
            eval_samples=samples,
            tokenizer=tokenizer,
            task_type=task_type,
            max_samples=max_samples_per_task,
            device=device,
        )

        results[task_name] = task_results
        print(f"  Accuracy: {task_results['accuracy']:.2%} "
              f"({task_results['correct']}/{task_results['total']}, "
              f"{task_results['parse_failures']} parse failures)")

    return results
