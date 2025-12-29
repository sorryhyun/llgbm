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
    RealAdapterDataset,
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
    WeightLoss,
    compute_delta_for_batch,
    save_checkpoint,
    load_checkpoint,
    train_step,
    train,
    evaluate,
)
from llgbm.text_encoder import (
    PretrainedTextEncoder,
    EmbeddingCache,
    create_text_encoder,
)
from llgbm.generator import (
    LoRAGenerator,
    create_generator,
)
from llgbm.evaluation import (
    format_chat_for_eval,
    compute_eval_loss_with_lora,
    compute_base_eval_loss,
    evaluate_task_performance,
    evaluate_teacher_adapter_loss,
    # Accuracy-based evaluation
    extract_mcq_answer,
    extract_bool_answer,
    extract_gsm8k_answer,
    compute_accuracy_with_lora,
    compute_accuracy_with_lora_batched,
    compute_teacher_accuracy,
    compute_base_accuracy,
    load_test_data,
    evaluate_all_tasks,
)
from llgbm.ablations import (
    AblationConfig,
    run_ablations,
    plot_ablation_results,
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
    "RealAdapterDataset",
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
    "WeightLoss",
    "compute_delta_for_batch",
    "save_checkpoint",
    "load_checkpoint",
    "train_step",
    "train",
    "evaluate",
    # Text encoder
    "PretrainedTextEncoder",
    "EmbeddingCache",
    "create_text_encoder",
    # Generator
    "LoRAGenerator",
    "create_generator",
    # Evaluation (loss-based)
    "format_chat_for_eval",
    "compute_eval_loss_with_lora",
    "compute_base_eval_loss",
    "evaluate_task_performance",
    "evaluate_teacher_adapter_loss",
    # Evaluation (accuracy-based)
    "extract_mcq_answer",
    "extract_bool_answer",
    "extract_gsm8k_answer",
    "compute_accuracy_with_lora",
    "compute_accuracy_with_lora_batched",
    "compute_teacher_accuracy",
    "compute_base_accuracy",
    "load_test_data",
    "evaluate_all_tasks",
    # Ablations
    "AblationConfig",
    "run_ablations",
    "plot_ablation_results",
]
