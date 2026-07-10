"""
Training subsystem for the multimodal AI pipeline.

This package owns learning orchestration only.
"""

from training.train_config import (
    TrainConfig,
    TrainConfigError,
    ConfigFrozenError,
    ConfigState,
    OVERRIDABLE_FIELDS,
    build_train_config,
)

from training.run_context import (
    RunContext,
    RunContextError,
    build_run_context,
)

from training.optimizer import (
    OptimizerError,
    OptimizerMetadata,
    build_optimizer,
    validate_optimizer_inputs,
    summarize_optimizer,
    optimizer_to_dict,
    get_optimizer_metadata,
)

from training.scheduler import (
    SchedulerError,
    SchedulerMetadata,
    build_scheduler,
    validate_scheduler_inputs,
    summarize_scheduler,
    scheduler_to_dict,
    get_scheduler_metadata,
)

from training.evaluation import (
    EvaluationError,
    EvaluationMetadata,
    EvaluationRuntimeState,
    EvaluationResult,
    Evaluator,
    build_evaluator,
    compute_loss,
    compute_metrics,
    extract_prediction,
)

__all__ = [
    "TrainConfig",
    "TrainConfigError",
    "ConfigFrozenError",
    "ConfigState",
    "OVERRIDABLE_FIELDS",
    "build_train_config",
    "RunContext",
    "RunContextError",
    "build_run_context",
    "OptimizerError",
    "OptimizerMetadata",
    "build_optimizer",
    "validate_optimizer_inputs",
    "summarize_optimizer",
    "optimizer_to_dict",
    "get_optimizer_metadata",
    "SchedulerError",
    "SchedulerMetadata",
    "build_scheduler",
    "validate_scheduler_inputs",
    "summarize_scheduler",
    "scheduler_to_dict",
    "get_scheduler_metadata",
    "EvaluationError",
    "EvaluationMetadata",
    "EvaluationRuntimeState",
    "EvaluationResult",
    "Evaluator",
    "build_evaluator",
    "compute_loss",
    "compute_metrics",
    "extract_prediction",
]
