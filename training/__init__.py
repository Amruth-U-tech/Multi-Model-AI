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
]
