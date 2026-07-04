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
]
