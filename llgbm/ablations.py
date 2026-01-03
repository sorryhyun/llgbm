"""Ablation study runner for LLGBM.

This module provides high-level functions to run ablation experiments
comparing different training configurations (multi-task, delta-only, weight-only).
"""

import gc
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import AutoModelForCausalLM, AutoTokenizer

from llgbm.probes import create_generic_probes
from llgbm.functional import FunctionalLoRA
from llgbm.dataset import RealAdapterDataset
from llgbm.generator import create_generator
from llgbm.text_encoder import create_text_encoder
from llgbm.training import (
    TrainingConfig,
    MultiTaskLoss,
    DeltaOnlyLoss,
    WeightLoss,
    train,
    evaluate,
)


def _precompute_embedding_cache(
    dataset: RealAdapterDataset,
    text_encoder,
    cache_dir: Path,
    *,
    overwrite: bool = False,
    max_length: int = 256,
    variants: int = 1,
    seed: int = 42,
) -> None:
    import numpy as np
    import random
    import hashlib
    from tqdm.auto import tqdm

    cache_dir.mkdir(parents=True, exist_ok=True)
    device = next(text_encoder.model.parameters()).device
    variants = max(1, int(variants))

    written = 0
    skipped = 0
    for sample in tqdm(dataset.samples, desc="Precomputing condition embeddings"):
        adapter_name = sample["name"]
        safe_name = adapter_name.replace("/", "_").replace("\\", "_")
        cache_path = cache_dir / f"{safe_name}.npy"

        if cache_path.exists() and not overwrite:
            skipped += 1
            continue

        if dataset.shuffle_task_prompts:
            all_texts = dataset.task_prompt_pools.get(sample["task"], [adapter_name])
        else:
            all_texts = dataset._load_prompts(sample)

        # Stable per-adapter seed (avoid Python's randomized hash()).
        adapter_seed_bytes = hashlib.sha256(adapter_name.encode("utf-8")).digest()[:8]
        adapter_seed = int.from_bytes(adapter_seed_bytes, byteorder="little", signed=False)

        pooled_variants = []
        for v in range(variants):
            rng = random.Random(adapter_seed + int(seed) + v)
            if len(all_texts) >= dataset.num_prompts:
                selected_texts = rng.sample(all_texts, dataset.num_prompts)
            else:
                repeated = (all_texts * ((dataset.num_prompts // max(1, len(all_texts))) + 1))
                selected_texts = repeated[: dataset.num_prompts]

            emb = text_encoder.encode_texts(selected_texts, max_length=max_length, device=device)
            pooled = emb.mean(dim=0)
            pooled_variants.append(pooled.detach().cpu())

        pooled_stack = torch.stack(pooled_variants, dim=0)  # (V, embed_dim)
        if variants == 1:
            np.save(cache_path, pooled_stack[0].numpy())
        else:
            np.save(cache_path, pooled_stack.numpy())
        written += 1

    print(f"[OK] Embedding cache: wrote={written}, skipped={skipped}, dir={cache_dir}")
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.empty_cache()


@dataclass
class AblationConfig:
    """Configuration for ablation experiments."""

    # Experiment settings
    configs: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "multitask": {"lambda_weight": 1.0, "lambda_delta": 0.1},
        "multitask2": {"lambda_weight": 0.5, "lambda_delta": 0.5},
        "delta_only": {"lambda_weight": 0.0, "lambda_delta": 1.0},
        "weight_only": {"lambda_weight": 1.0, "lambda_delta": 0.0},
    })
    num_trials: int = 3
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    num_steps: int = 100

    # Paths
    checkpoint_dir: str = "./checkpoints"
    deltas_dir: str = "./llgbm/deltas"
    output_dir: str = "outputs/phase4_5_ablations"

    # Model settings
    use_small_model: bool = True
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    warmup_steps: int = 50
    probes_per_task: int = 10
    # Cap total probes used for delta computation (VRAM scales ~ linearly with this).
    num_probes: int = 10
    # Max length for probe tokenization (VRAM scales ~ linearly with this).
    max_probe_length: int = 256
    # If True, batch all probes into a single forward per adapter (faster, potentially higher peak VRAM).
    delta_batch_probes: bool = True
    # Enable HF gradient checkpointing on the base model (reduces VRAM, increases compute).
    enable_gradient_checkpointing: bool = False

    # Text encoder settings
    text_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    freeze_text_encoder: bool = True
    num_prompts_per_adapter: int = 8
    shuffle_task_prompts: bool = False  # Sample from task-wide prompt pool for generalization
    # Cache condition embeddings per adapter to avoid repeated tokenization + text encoder forward.
    # NOTE: if `shuffle_task_prompts=True`, caching makes conditioning deterministic per-adapter.
    embedding_cache_dir: Optional[str] = None
    precompute_embeddings: bool = False
    overwrite_embedding_cache: bool = False
    # If >1, precompute multiple pooled embeddings per adapter (different prompt subsets)
    # and sample a variant each time the adapter is loaded.
    embedding_cache_variants: int = 1
    embedding_cache_seed: int = 42

    # Colab support
    in_colab: bool = False
    drive_output_dir: Optional[str] = None


def setup_base_components(
    config: AblationConfig,
    base_config: TrainingConfig,
) -> Dict[str, Any]:
    """
    Set up shared components for ablation experiments.

    Args:
        config: Ablation configuration
        base_config: Base training configuration

    Returns:
        Dict with base_model, tokenizer, functional_lora, probe_tokens, etc.
    """
    TORCH_DTYPE = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }[base_config.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_dir = Path(config.checkpoint_dir)
    deltas_dir = Path(config.deltas_dir)

    print(f"Model: {base_config.base_model}")
    print(f"Device: {device}")

    # Load base model
    tokenizer = AutoTokenizer.from_pretrained(base_config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_config.base_model,
        torch_dtype=TORCH_DTYPE,
        device_map=device,
        trust_remote_code=True
    )
    base_model.config.output_hidden_states = False
    base_model.config.use_cache = False
    if config.enable_gradient_checkpointing and hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()
        # HF gradient checkpointing is typically gated on `model.training`.
        base_model.train()
        print("[OK] Gradient checkpointing enabled for base model")
    else:
        base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    print("[OK] Base model loaded")

    # Load probes
    delta_manifest_path = deltas_dir / "delta_manifest.json"
    manifest_path = checkpoint_dir / "manifest.json"

    all_probes = []
    if delta_manifest_path.exists() and manifest_path.exists():
        with open(delta_manifest_path) as f:
            delta_manifest = json.load(f)
        with open(manifest_path) as f:
            adapter_manifest = json.load(f)

        adapter_paths = {a["name"]: a["path"] for a in adapter_manifest.get("adapters", [])}
        tasks_seen = set()

        for adapter_name, adapter_info in delta_manifest["adapters"].items():
            task = adapter_info.get("task", "unknown")
            if task not in tasks_seen:
                remaining = max(0, base_config.num_probes - len(all_probes))
                if remaining == 0:
                    break
                adapter_path = adapter_paths.get(adapter_name)
                if adapter_path:
                    prompts_file = Path(adapter_path) / "prompts.json"
                    if prompts_file.exists():
                        with open(prompts_file) as f:
                            prompts_data = json.load(f)
                        per_task = min(config.probes_per_task, remaining)
                        probes = prompts_data.get("prompts", [])[:per_task]
                        if probes:
                            all_probes.extend(probes)
                            tasks_seen.add(task)
                            print(f"  Loaded {len(probes)} probes for {task}")

    if not all_probes:
        print("[WARN] No task-specific probes, using generic")
        all_probes = create_generic_probes()[:base_config.num_probes]
    elif len(all_probes) > base_config.num_probes:
        # Should be prevented by `remaining`, but keep as a safety net.
        print(f"[INFO] Limiting probes: {len(all_probes)} -> {base_config.num_probes}")
        all_probes = all_probes[:base_config.num_probes]

    # Tokenize probes
    probe_tokens, probe_masks = [], []
    for p in all_probes:
        enc = tokenizer(p, return_tensors="pt", truncation=True, max_length=base_config.max_probe_length)
        probe_tokens.append(enc["input_ids"].to(device))
        probe_masks.append(enc["attention_mask"].to(device))

    # Compute base activation
    with torch.no_grad():
        base_acts = []
        for ids, mask in zip(probe_tokens, probe_masks):
            backbone = getattr(base_model, "model", None)
            if backbone is not None:
                out = backbone(input_ids=ids, attention_mask=mask, use_cache=False)
                hidden = out.last_hidden_state
            else:
                out = base_model(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
                hidden = out.hidden_states[-1]
            seq_lens = mask.long().sum(dim=1).clamp(min=1) - 1
            batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
            h = hidden[batch_idx, seq_lens, :].squeeze(0)
            base_acts.append(h)
        base_activation = torch.stack(base_acts).mean(dim=0)

    functional_lora = FunctionalLoRA(base_model, base_config.lora_rank, base_config.lora_alpha)
    print(f"[OK] Probes: {len(all_probes)}, FunctionalLoRA ready")

    # Create pretrained text encoder
    print(f"[INFO] Loading text encoder: {config.text_encoder_name}")
    text_encoder = create_text_encoder(
        model_name=config.text_encoder_name,
        freeze=config.freeze_text_encoder,
        device=device,
    )

    # Create dataset with prompt batches
    # Use text encoder's tokenizer for consistency
    if config.precompute_embeddings and not config.embedding_cache_dir:
        config.embedding_cache_dir = str(Path(config.output_dir) / "embedding_cache")
    dataset = RealAdapterDataset(
        checkpoint_dir=str(checkpoint_dir),
        deltas_dir=str(deltas_dir),
        tokenizer=text_encoder.tokenizer,
        config=base_config,
        num_prompts=config.num_prompts_per_adapter,
        embedding_cache_dir=config.embedding_cache_dir,
        shuffle_task_prompts=config.shuffle_task_prompts,
    )
    shuffle_mode = "task-shuffled" if config.shuffle_task_prompts else "adapter-specific"
    print(f"[OK] Dataset: {len(dataset)} samples, {config.num_prompts_per_adapter} prompts per adapter ({shuffle_mode})")

    if config.precompute_embeddings and config.embedding_cache_dir:
        _precompute_embedding_cache(
            dataset=dataset,
            text_encoder=text_encoder,
            cache_dir=Path(config.embedding_cache_dir),
            overwrite=config.overwrite_embedding_cache,
            variants=config.embedding_cache_variants,
            seed=config.embedding_cache_seed,
        )

    # Create weight loss criterion
    weight_criterion = WeightLoss()
    print("[OK] WeightLoss criterion ready")

    return {
        "base_model": base_model,
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "functional_lora": functional_lora,
        "base_activation": base_activation,
        "probe_tokens": probe_tokens,
        "probe_masks": probe_masks,
        "dataset": dataset,
        "device": device,
        "torch_dtype": TORCH_DTYPE,
        "weight_criterion": weight_criterion,
    }


def run_trial(
    config_name: str,
    lambda_weight: float,
    lambda_delta: float,
    seed: int,
    trial_idx: int,
    num_steps: int,
    base_config: TrainingConfig,
    components: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """Run a single ablation trial."""
    print(f"\n{'='*60}")
    print(f"Config: {config_name} | Trial {trial_idx+1} | Seed: {seed}")
    print(f"lambda_w={lambda_weight}, lambda_d={lambda_delta}")
    print(f"{'='*60}")

    device = components["device"]
    TORCH_DTYPE = components["torch_dtype"]

    # Create config for this trial
    config = TrainingConfig(
        use_small_model=base_config.use_small_model,
        batch_size=base_config.batch_size,
        gradient_accumulation_steps=base_config.gradient_accumulation_steps,
        learning_rate=base_config.learning_rate,
        num_steps=num_steps,
        warmup_steps=base_config.warmup_steps,
        lambda_weight=lambda_weight,
        lambda_delta=lambda_delta,
        num_probes=base_config.num_probes,
        max_probe_length=base_config.max_probe_length,
        delta_batch_probes=base_config.delta_batch_probes,
        output_dir=str(output_dir / f"{config_name}_trial{trial_idx}"),
    )
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # Fresh generator with pretrained text encoder
    generator = create_generator(
        config,
        seed,
        device,
        text_encoder=components["text_encoder"],
    )

    # Loss function
    if lambda_weight == 0:
        criterion = DeltaOnlyLoss()
    else:
        criterion = MultiTaskLoss(lambda_weight=lambda_weight, lambda_delta=lambda_delta)

    # Get weight criterion for direct weight supervision
    weight_criterion = components.get("weight_criterion")

    # Optimizer & scheduler
    optimizer = AdamW(generator.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    warmup_steps = min(config.warmup_steps, num_steps // 10)
    cosine_steps = max(1, num_steps - warmup_steps)
    scheduler = SequentialLR(
        optimizer,
        [LinearLR(optimizer, 0.1, 1.0, warmup_steps),
         CosineAnnealingLR(optimizer, cosine_steps, config.learning_rate * 0.01)],
        [warmup_steps]
    )

    # Dataloader
    dataloader = DataLoader(
        components["dataset"],
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=components["dataset"].collate_fn,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    # Train
    start_time = time.time()
    state = train(
        generator=generator,
        dataloader=dataloader,
        functional_lora=components["functional_lora"],
        base_activation=components["base_activation"],
        probe_tokens=components["probe_tokens"],
        probe_masks=components["probe_masks"],
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        compute_dtype=TORCH_DTYPE,
        weight_criterion=weight_criterion,  # Pass weight criterion for direct weight supervision
    )
    train_time = time.time() - start_time

    # Evaluate
    eval_dataloader = DataLoader(
        components["dataset"],
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=components["dataset"].collate_fn
    )
    eval_results = evaluate(
        generator=generator,
        dataloader=eval_dataloader,
        functional_lora=components["functional_lora"],
        base_activation=components["base_activation"],
        probe_tokens=components["probe_tokens"],
        probe_masks=components["probe_masks"],
        criterion=criterion,
    )

    # Cleanup
    del generator, optimizer, scheduler, dataloader, eval_dataloader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = {
        "config_name": config_name,
        "trial": trial_idx,
        "seed": seed,
        "lambda_weight": lambda_weight,
        "lambda_delta": lambda_delta,
        "num_steps": num_steps,
        "final_loss": state.loss_history[-1] if state.loss_history else None,
        "best_loss": state.best_loss,
        "train_time": train_time,
        **eval_results,
    }

    cosine = result.get('mean_cosine', None)
    cosine_str = f"{cosine:.4f}" if cosine is not None else "N/A"
    print(f"Result: loss={result['final_loss']:.4f}, mean_cosine={cosine_str}, time={train_time:.1f}s")

    return result


def run_ablations(config: AblationConfig) -> Dict[str, Any]:
    """
    Run complete ablation study.

    Args:
        config: AblationConfig with experiment settings

    Returns:
        Dict with all results, summary, and paths to saved outputs
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Base training config
    base_config = TrainingConfig(
        use_small_model=config.use_small_model,
        batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_steps=config.num_steps,
        warmup_steps=config.warmup_steps,
        num_probes=config.num_probes,
        max_probe_length=config.max_probe_length,
        delta_batch_probes=config.delta_batch_probes,
    )

    print(f"Configurations: {list(config.configs.keys())}")
    print(f"Trials per config: {config.num_trials}")
    print(f"Steps per trial: {config.num_steps}")
    print(f"Total runs: {len(config.configs) * config.num_trials}")

    # Setup shared components
    components = setup_base_components(config, base_config)

    # Run all trials
    all_results = []
    for config_name, params in config.configs.items():
        for trial_idx in range(config.num_trials):
            seed = config.seeds[trial_idx] if trial_idx < len(config.seeds) else 42 + trial_idx
            result = run_trial(
                config_name=config_name,
                lambda_weight=params["lambda_weight"],
                lambda_delta=params["lambda_delta"],
                seed=seed,
                trial_idx=trial_idx,
                num_steps=config.num_steps,
                base_config=base_config,
                components=components,
                output_dir=output_dir,
            )
            all_results.append(result)

    print(f"\n\nCompleted {len(all_results)} trials!")

    # Build summary
    import pandas as pd
    df = pd.DataFrame(all_results)

    summary = {}
    for c in config.configs.keys():
        c_df = df[df['config_name'] == c]
        summary[c] = {
            "final_loss_mean": float(c_df['final_loss'].mean()),
            "final_loss_std": float(c_df['final_loss'].std()),
            "mean_cosine_mean": float(c_df['mean_cosine'].mean()) if 'mean_cosine' in c_df.columns else None,
            "mean_cosine_std": float(c_df['mean_cosine'].std()) if 'mean_cosine' in c_df.columns else None,
        }

    # Save results
    df.to_csv(output_dir / "all_trials.csv", index=False)

    final_results = {
        "configs": config.configs,
        "num_trials": config.num_trials,
        "seeds": config.seeds[:config.num_trials],
        "summary": summary,
        "all_trials": all_results,
    }

    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\nSaved to {output_dir}/")

    return {
        "results": all_results,
        "summary": summary,
        "dataframe": df,
        "output_dir": output_dir,
        "components": components,
    }


def plot_ablation_results(
    df,
    output_dir: Path,
    configs: List[str] = None,
):
    """
    Plot ablation comparison charts.

    Args:
        df: DataFrame with trial results
        output_dir: Directory to save plots
        configs: List of config names to plot
    """
    import matplotlib.pyplot as plt

    if configs is None:
        configs = df['config_name'].unique().tolist()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6'][:len(configs)]

    # Final Loss comparison
    means = [df[df['config_name'] == c]['final_loss'].mean() for c in configs]
    stds = [df[df['config_name'] == c]['final_loss'].std() for c in configs]
    axes[0].bar(configs, means, yerr=stds, color=colors, capsize=5, alpha=0.8)
    axes[0].set_ylabel('Final Loss')
    axes[0].set_title('Final Loss by Configuration')
    axes[0].grid(axis='y', alpha=0.3)

    # Cosine similarity comparison
    if 'mean_cosine' in df.columns and df['mean_cosine'].notna().any():
        means = [df[df['config_name'] == c]['mean_cosine'].mean() for c in configs]
        stds = [df[df['config_name'] == c]['mean_cosine'].std() for c in configs]
        axes[1].bar(configs, means, yerr=stds, color=colors, capsize=5, alpha=0.8)
        axes[1].set_ylabel('Mean Cosine Similarity')
        axes[1].set_title('Delta Cosine Similarity by Configuration')
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].set_ylim(-1, 1)
    else:
        axes[1].text(0.5, 0.5, 'Cosine similarity not available',
                     ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Delta Cosine Similarity')

    plt.tight_layout()
    plt.savefig(output_dir / "ablation_comparison.png", dpi=150)
    plt.show()

    return fig
