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

__all__ = [
    "TrainConfig",
    "TrainConfigError",
    "ConfigFrozenError",
    "ConfigState",
    "OVERRIDABLE_FIELDS",
    "build_train_config",
]
