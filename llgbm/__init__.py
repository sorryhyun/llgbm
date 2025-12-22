"""LLGBM - LoRA Learning with Guided Behavioral Matching."""

from llgbm.probes import create_generic_probes, create_domain_probes, create_mixed_probes
from llgbm.delta import (
    get_average_activation,
    compute_base_activation,
    compute_adapter_delta,
    DeltaCache,
)
from llgbm.dataset import (
    DeltaAugmentedDataset,
    Text2Qwen25LoRA_DeltaDataset,
    create_dataloader,
)
from llgbm.functional import (
    FunctionalLoRA,
    compute_delta_differentiable,
    compute_delta_memory_efficient,
)
from llgbm.training import (
    TrainingConfig,
    TrainingState,
    MultiTaskLoss,
    DeltaOnlyLoss,
    compute_delta_for_batch,
    save_checkpoint,
    load_checkpoint,
    train_step,
    train,
    evaluate,
)

__version__ = "0.1.0"
__all__ = [
    # Probes
    "create_generic_probes",
    "create_domain_probes",
    "create_mixed_probes",
    # Delta computation
    "get_average_activation",
    "compute_base_activation",
    "compute_adapter_delta",
    "DeltaCache",
    # Datasets
    "DeltaAugmentedDataset",
    "Text2Qwen25LoRA_DeltaDataset",
    "create_dataloader",
    # Functional (differentiable)
    "FunctionalLoRA",
    "compute_delta_differentiable",
    "compute_delta_memory_efficient",
    # Training
    "TrainingConfig",
    "TrainingState",
    "MultiTaskLoss",
    "DeltaOnlyLoss",
    "compute_delta_for_batch",
    "save_checkpoint",
    "load_checkpoint",
    "train_step",
    "train",
    "evaluate",
]
