# =============================================================================
# training/train_config.py
# Training Configuration Authority -- Multimodal AI Pipeline
# =============================================================================
#
# Purpose:
#   Single source of truth for all training configuration. Defines,
#   normalizes, validates, controls overrides, and freezes the config
#   that future training files will consume.
#
# Responsibilities:
#   1. Configuration Authority -- owns every training parameter
#   2. Validation -- rejects invalid values early and loudly
#   3. Normalization -- canonicalizes user input
#   4. Dependency Rules -- enforces inter-field consistency
#   5. Controlled Override -- accepts/rejects experiment changes
#   6. Freeze -- locks config for reproducibility once training starts
#
# What this file does NOT do:
#   - Train models
#   - Build datasets or dataloaders
#   - Create optimizers or schedulers
#   - Save checkpoints or logs
#   - Import torch, models, or data_pipeline
#
# Usage:
#   from training.train_config import TrainConfig, build_train_config
# =============================================================================

import sys
import copy
import math
import logging
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Sequence, Set, FrozenSet, Tuple, Union

# -- Project root bootstrap ----------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _THIS_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from configs.paths import PROJECT_ROOT, CHECKPOINT_DIR, EXPERIMENT_DIR, LOG_DIR

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

VALID_OPTIMIZERS: FrozenSet[str] = frozenset({"adamw", "adam", "sgd"})
VALID_SCHEDULERS: FrozenSet[str] = frozenset({"none", "cosine", "step", "plateau"})
VALID_LOSSES: FrozenSet[str] = frozenset({"mse", "mae", "huber"})
VALID_DEVICES: FrozenSet[str] = frozenset({"auto", "cpu", "cuda"})

OVERRIDABLE_FIELDS: FrozenSet[str] = frozenset({
    "experiment_name", "description", "epochs", "batch_size",
    "learning_rate", "weight_decay", "optimizer", "scheduler",
    "warmup_epochs", "loss_name", "gradient_clip", "mixed_precision",
    "validation_frequency", "logging_frequency", "checkpoint_frequency",
    "num_workers", "device",
})

_PROTECTED_FIELDS: FrozenSet[str] = frozenset({
    "_state", "_frozen",
})


# =============================================================================
# Errors
# =============================================================================

class TrainConfigError(Exception):
    """Structured training configuration error."""

    def __init__(self, stage: str, field_name: str, received: Any,
                 expected: str, resolution: str = ""):
        self.stage = stage
        self.field_name = field_name
        self.received = received
        self.expected = expected
        self.resolution = resolution
        lines = [
            "[TRAIN CONFIG ERROR]",
            f"  Stage     : {stage}",
            f"  Field     : {field_name}",
            f"  Received  : {received!r}",
            f"  Expected  : {expected}",
        ]
        if resolution:
            lines.append(f"  Resolution: {resolution}")
        super().__init__("\n".join(lines))


class ConfigFrozenError(Exception):
    """Raised when attempting to modify a frozen config."""

    def __init__(self, field_name: str):
        super().__init__(
            f"[TRAIN CONFIG FROZEN] Cannot modify '{field_name}' -- "
            f"config is frozen for training reproducibility."
        )


# =============================================================================
# Lifecycle State
# =============================================================================

class ConfigState(Enum):
    CREATED = "CREATED"
    VALIDATED = "VALIDATED"
    OVERRIDDEN = "OVERRIDDEN"
    FROZEN = "FROZEN"


# =============================================================================
# TrainConfig
# =============================================================================

@dataclass
class TrainConfig:
    """
    Central training configuration authority.

    Lifecycle: CREATED -> validate() -> VALIDATED -> apply_overrides() ->
               OVERRIDDEN -> freeze() -> FROZEN (immutable)
    """

    # -- General ---------------------------------------------------------------
    experiment_name: str = "default_experiment"
    seed: int = 42
    description: str = ""

    # -- Dataset ---------------------------------------------------------------
    dataset_name: str = "sample_100"
    validation_dataset_name: Optional[str] = None
    num_workers: Optional[int] = None

    # -- Dataset Selection & Splitting -----------------------------------------
    validation_split: float = 0.2
    train_all: bool = False
    train_datasets: Tuple[str, ...] = ()

    # -- Training --------------------------------------------------------------
    epochs: int = 50
    batch_size: int = 16
    validation_frequency: int = 1
    logging_frequency: int = 10

    # -- Optimizer -------------------------------------------------------------
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    backbone_lr_multiplier: Optional[float] = None

    # -- Scheduler -------------------------------------------------------------
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    step_size: int = 10
    step_gamma: float = 0.1
    plateau_patience: int = 5
    plateau_factor: float = 0.5

    # -- Loss ------------------------------------------------------------------
    loss_name: str = "mse"

    # -- Checkpointing ---------------------------------------------------------
    checkpoint_frequency: int = 5
    save_best: bool = True
    save_latest: bool = True
    resume: bool = False
    resume_checkpoint: Optional[str] = None

    # -- Runtime ---------------------------------------------------------------
    device: str = "auto"
    mixed_precision: bool = False
    gradient_clip: Optional[float] = 1.0
    gradient_accumulation_steps: int = 1

    # -- Reproducibility -------------------------------------------------------
    deterministic: bool = True
    log_level: str = "INFO"

    # -- Internal (not user-facing) --------------------------------------------
    _state: ConfigState = field(default=ConfigState.CREATED, repr=False)
    _frozen: bool = field(default=False, repr=False)

    def __post_init__(self):
        # Normalize on construction but do not validate yet
        self._normalize()

    def __setattr__(self, name: str, value: Any):
        try:
            frozen = object.__getattribute__(self, "_frozen")
        except AttributeError:
            frozen = False
        if frozen:
            raise ConfigFrozenError(name)
        object.__setattr__(self, name, value)

    # -- Normalization ---------------------------------------------------------

    def _normalize(self):
        """Canonicalize string fields to lowercase/stripped forms."""
        if isinstance(self.experiment_name, str):
            self.experiment_name = self.experiment_name.strip()
        if isinstance(self.description, str):
            self.description = self.description.strip()
        if isinstance(self.dataset_name, str):
            self.dataset_name = self.dataset_name.strip()
        if isinstance(self.optimizer, str):
            self.optimizer = self.optimizer.strip().lower()
        if isinstance(self.scheduler, str):
            self.scheduler = self.scheduler.strip().lower()
        if isinstance(self.loss_name, str):
            self.loss_name = self.loss_name.strip().lower()
        if isinstance(self.device, str):
            self.device = self.device.strip().lower()
        if isinstance(self.log_level, str):
            self.log_level = self.log_level.strip().upper()
        if isinstance(self.validation_dataset_name, str):
            self.validation_dataset_name = self.validation_dataset_name.strip() or None
        if isinstance(self.resume_checkpoint, str):
            self.resume_checkpoint = self.resume_checkpoint.strip() or None

        # Normalize train_datasets: accept list/tuple of strings -> tuple of stripped strings
        if isinstance(self.train_datasets, (list, tuple)):
            self.train_datasets = tuple(
                s.strip() if isinstance(s, str) else s
                for s in self.train_datasets
            )
        elif isinstance(self.train_datasets, str):
            self.train_datasets = (self.train_datasets.strip(),) if self.train_datasets.strip() else ()

    # -- Validation ------------------------------------------------------------

    def _err(self, fld: str, received: Any, expected: str,
             resolution: str = "") -> TrainConfigError:
        return TrainConfigError("validation", fld, received, expected, resolution)

    def _validate_positive_int(self, name: str, val: Any, allow_zero: bool = False):
        if isinstance(val, bool) or not isinstance(val, int):
            raise self._err(name, val, f"{name} must be int, not {type(val).__name__}")
        lo = 0 if allow_zero else 1
        if val < lo:
            raise self._err(name, val, f"{name} >= {lo}")

    def _validate_positive_float(self, name: str, val: Any):
        if isinstance(val, bool):
            raise self._err(name, val, f"{name} must be numeric, not bool")
        if not isinstance(val, (int, float)):
            raise self._err(name, val, f"{name} must be numeric")
        if math.isnan(val) or math.isinf(val):
            raise self._err(name, val, f"{name} must be finite")
        if val <= 0:
            raise self._err(name, val, f"{name} > 0")

    def _validate_non_negative_float(self, name: str, val: Any):
        if isinstance(val, bool):
            raise self._err(name, val, f"{name} must be numeric, not bool")
        if not isinstance(val, (int, float)):
            raise self._err(name, val, f"{name} must be numeric")
        if math.isnan(val) or math.isinf(val):
            raise self._err(name, val, f"{name} must be finite")
        if val < 0:
            raise self._err(name, val, f"{name} >= 0")

    def validate(self) -> "TrainConfig":
        """
        Normalize and validate all fields. Sets state to VALIDATED.

        Returns self for chaining.

        Raises:
            TrainConfigError: on any invalid field.
            ConfigFrozenError: if config is already frozen.
        """
        if self._frozen:
            raise ConfigFrozenError("validate")
        self._normalize()

        # -- Strings -----------------------------------------------------------
        if not isinstance(self.experiment_name, str) or not self.experiment_name:
            raise self._err("experiment_name", self.experiment_name,
                            "non-empty string")
        if not isinstance(self.dataset_name, str) or not self.dataset_name:
            raise self._err("dataset_name", self.dataset_name, "non-empty string")
        if not isinstance(self.description, str):
            raise self._err("description", self.description, "string")

        # -- Validation dataset name -------------------------------------------
        if self.validation_dataset_name is not None:
            if isinstance(self.validation_dataset_name, bool):
                raise self._err("validation_dataset_name",
                                self.validation_dataset_name, "None or non-empty string")
            if not isinstance(self.validation_dataset_name, str):
                raise self._err("validation_dataset_name",
                                self.validation_dataset_name,
                                "None or non-empty string",
                                f"Got {type(self.validation_dataset_name).__name__}.")
            if not self.validation_dataset_name:
                raise self._err("validation_dataset_name",
                                self.validation_dataset_name,
                                "None or non-empty string",
                                "Use None instead of empty string.")

        # -- Seed --------------------------------------------------------------
        self._validate_positive_int("seed", self.seed, allow_zero=True)

        # -- Training integers -------------------------------------------------
        self._validate_positive_int("epochs", self.epochs)
        self._validate_positive_int("batch_size", self.batch_size)
        self._validate_positive_int("validation_frequency", self.validation_frequency)
        self._validate_positive_int("logging_frequency", self.logging_frequency)
        self._validate_positive_int("checkpoint_frequency", self.checkpoint_frequency)
        self._validate_positive_int("gradient_accumulation_steps",
                                    self.gradient_accumulation_steps)

        # -- Workers -----------------------------------------------------------
        if self.num_workers is not None:
            self._validate_positive_int("num_workers", self.num_workers,
                                        allow_zero=True)

        # -- Optimizer floats --------------------------------------------------
        self._validate_positive_float("learning_rate", self.learning_rate)
        self._validate_non_negative_float("weight_decay", self.weight_decay)

        if self.backbone_lr_multiplier is not None:
            self._validate_positive_float("backbone_lr_multiplier",
                                          self.backbone_lr_multiplier)

        # -- Enum-like fields --------------------------------------------------
        if self.optimizer not in VALID_OPTIMIZERS:
            raise self._err("optimizer", self.optimizer,
                            f"one of {sorted(VALID_OPTIMIZERS)}")
        if self.scheduler not in VALID_SCHEDULERS:
            raise self._err("scheduler", self.scheduler,
                            f"one of {sorted(VALID_SCHEDULERS)}")
        if self.loss_name not in VALID_LOSSES:
            raise self._err("loss_name", self.loss_name,
                            f"one of {sorted(VALID_LOSSES)}")
        if self.device not in VALID_DEVICES:
            raise self._err("device", self.device,
                            f"one of {sorted(VALID_DEVICES)}")

        # -- Scheduler params --------------------------------------------------
        self._validate_positive_int("warmup_epochs", self.warmup_epochs,
                                    allow_zero=True)
        self._validate_positive_int("step_size", self.step_size)
        self._validate_positive_float("step_gamma", self.step_gamma)
        self._validate_positive_int("plateau_patience", self.plateau_patience)
        self._validate_positive_float("plateau_factor", self.plateau_factor)

        # -- Gradient clip -----------------------------------------------------
        if self.gradient_clip is not None:
            if isinstance(self.gradient_clip, bool):
                raise self._err("gradient_clip", self.gradient_clip,
                                "None or positive float")
            if not isinstance(self.gradient_clip, (int, float)):
                raise self._err("gradient_clip", self.gradient_clip,
                                "None or positive float")
            if math.isnan(self.gradient_clip) or math.isinf(self.gradient_clip):
                raise self._err("gradient_clip", self.gradient_clip, "finite")
            if self.gradient_clip <= 0:
                raise self._err("gradient_clip", self.gradient_clip,
                                "None or positive float")

        # -- Booleans ----------------------------------------------------------
        for bname in ("save_best", "save_latest", "resume", "mixed_precision",
                      "deterministic", "train_all"):
            val = getattr(self, bname)
            if not isinstance(val, bool):
                raise self._err(bname, val, "bool")

        # -- validation_split --------------------------------------------------
        if isinstance(self.validation_split, bool):
            raise self._err("validation_split", self.validation_split,
                            "float in (0.0, 1.0), not bool")
        if not isinstance(self.validation_split, (int, float)):
            raise self._err("validation_split", self.validation_split,
                            "float in (0.0, 1.0)")
        if math.isnan(self.validation_split) or math.isinf(self.validation_split):
            raise self._err("validation_split", self.validation_split,
                            "finite float in (0.0, 1.0)")
        if not (0.0 < self.validation_split < 1.0):
            raise self._err("validation_split", self.validation_split,
                            "0.0 < validation_split < 1.0",
                            "Typical values: 0.1, 0.15, 0.2")

        # -- train_datasets ----------------------------------------------------
        if not isinstance(self.train_datasets, (tuple, list)):
            raise self._err("train_datasets", self.train_datasets,
                            "tuple or list of non-empty strings")
        for i, name in enumerate(self.train_datasets):
            if not isinstance(name, str) or not name.strip():
                raise self._err("train_datasets", self.train_datasets,
                                f"all entries must be non-empty strings (index {i})")
        # Reject duplicates
        seen = set()
        for name in self.train_datasets:
            if name in seen:
                raise self._err("train_datasets", self.train_datasets,
                                f"no duplicate entries (duplicate: '{name}')")
            seen.add(name)

        # -- Log level ---------------------------------------------------------
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level not in valid_levels:
            raise self._err("log_level", self.log_level,
                            f"one of {sorted(valid_levels)}")

        # -- Dependency rules --------------------------------------------------
        self._apply_dependency_rules()

        object.__setattr__(self, "_state", ConfigState.VALIDATED)
        return self

    def _apply_dependency_rules(self):
        """Enforce inter-field consistency."""
        # scheduler=none -> warmup must be 0
        if self.scheduler == "none" and self.warmup_epochs != 0:
            logger.info("Dependency rule: scheduler='none' -> warmup_epochs=0")
            self.warmup_epochs = 0

        # warmup < epochs
        if self.warmup_epochs >= self.epochs:
            raise self._err(
                "warmup_epochs", self.warmup_epochs,
                f"warmup_epochs ({self.warmup_epochs}) < epochs ({self.epochs})",
                "Reduce warmup_epochs or increase epochs."
            )

        # resume dependencies
        if self.resume:
            if not self.resume_checkpoint:
                raise self._err(
                    "resume_checkpoint", self.resume_checkpoint,
                    "non-empty checkpoint reference when resume=True",
                    "Set resume_checkpoint to a valid filename."
                )
        else:
            if self.resume_checkpoint is not None:
                logger.info("Dependency rule: resume=False -> resume_checkpoint=None")
                self.resume_checkpoint = None

        # resume_checkpoint path safety
        if self.resume_checkpoint is not None:
            ckpt = Path(self.resume_checkpoint)
            ckpt_root = CHECKPOINT_DIR.resolve()
            if ckpt.is_absolute():
                resolved = ckpt.resolve()
            else:
                resolved = (CHECKPOINT_DIR / ckpt).resolve()
            # Real parent-chain containment check
            if resolved != ckpt_root and ckpt_root not in resolved.parents:
                raise self._err(
                    "resume_checkpoint", self.resume_checkpoint,
                    f"path contained within {CHECKPOINT_DIR}",
                    "Use a filename or relative path inside checkpoints/."
                )

    # -- Override --------------------------------------------------------------

    def apply_overrides(self, overrides: Dict[str, Any]) -> "TrainConfig":
        """
        Apply experiment overrides from a dictionary.

        Only fields in OVERRIDABLE_FIELDS are accepted. Revalidates after
        applying. Sets state to OVERRIDDEN.

        Args:
            overrides: field_name -> value mapping

        Returns:
            self for chaining.

        Raises:
            TrainConfigError: on protected/unknown fields or validation failure.
            ConfigFrozenError: if config is already frozen.
        """
        if self._frozen:
            raise ConfigFrozenError("apply_overrides")

        if not isinstance(overrides, dict):
            raise TrainConfigError(
                "override", "overrides", type(overrides).__name__,
                "dict", "Pass a dictionary of field_name -> value."
            )

        for key, value in overrides.items():
            if key in _PROTECTED_FIELDS:
                raise TrainConfigError(
                    "override", key, value,
                    "not a protected field",
                    f"'{key}' cannot be overridden."
                )
            if key not in OVERRIDABLE_FIELDS:
                raise TrainConfigError(
                    "override", key, value,
                    f"one of OVERRIDABLE_FIELDS",
                    f"'{key}' is not in the override allowlist."
                )
            setattr(self, key, value)

        self._normalize()
        # Re-run full validation
        object.__setattr__(self, "_state", ConfigState.CREATED)
        self.validate()
        object.__setattr__(self, "_state", ConfigState.OVERRIDDEN)
        return self

    # -- Freeze ----------------------------------------------------------------

    def freeze(self) -> "TrainConfig":
        """
        Lock config for training. No further modifications allowed.

        Raises:
            TrainConfigError: if config has not been validated.
            ConfigFrozenError: if already frozen.
        """
        if self._frozen:
            raise ConfigFrozenError("freeze")
        if self._state == ConfigState.CREATED:
            raise TrainConfigError(
                "freeze", "_state", self._state.value,
                "VALIDATED or OVERRIDDEN",
                "Call validate() before freeze()."
            )
        object.__setattr__(self, "_frozen", True)
        object.__setattr__(self, "_state", ConfigState.FROZEN)
        return self

    # -- Query -----------------------------------------------------------------

    @property
    def state(self) -> ConfigState:
        return self._state

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def summary(self) -> str:
        """Human-readable summary for logging."""
        lines = [
            "=" * 60,
            "  TRAINING CONFIGURATION",
            "=" * 60,
            f"  State            : {self._state.value}",
            f"  Experiment       : {self.experiment_name}",
            f"  Dataset          : {self.dataset_name}",
            f"  Train All        : {self.train_all}",
            f"  Train Datasets   : {self.train_datasets if self.train_datasets else '(default)'}",
            f"  Val Split        : {self.validation_split}",
            f"  Epochs           : {self.epochs}",
            f"  Batch Size       : {self.batch_size}",
            f"  Optimizer        : {self.optimizer} (lr={self.learning_rate})",
            f"  Scheduler        : {self.scheduler}",
            f"  Loss             : {self.loss_name}",
            f"  Device           : {self.device}",
            f"  Mixed Precision  : {self.mixed_precision}",
            f"  Gradient Clip    : {self.gradient_clip}",
            f"  Grad Accumulation: {self.gradient_accumulation_steps}",
            f"  Resume           : {self.resume}",
            f"  Deterministic    : {self.deterministic}",
            f"  Seed             : {self.seed}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serializable dict of all public fields."""
        result = {}
        for f in fields(self):
            if f.name.startswith("_"):
                continue
            result[f.name] = getattr(self, f.name)
        result["_state"] = self._state.value
        result["_frozen"] = self._frozen
        return result

    def as_dict(self) -> Dict[str, Any]:
        """Alias for to_dict(). Ergonomic compatibility."""
        return self.to_dict()


# =============================================================================
# Factory
# =============================================================================

def build_train_config(**kwargs: Any) -> TrainConfig:
    """
    Build and validate a TrainConfig from keyword arguments.

    Returns a validated (but not frozen) TrainConfig.
    """
    cfg = TrainConfig(**kwargs)
    cfg.validate()
    return cfg


# =============================================================================
# Smoke Test
# =============================================================================

if __name__ == "__main__":
    import os

    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] [%(levelname)s] %(name)s -- %(message)s",
        datefmt="%H:%M:%S",
    )

    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        global passed, failed
        if condition:
            passed += 1
            print(f"    [PASS]     {name}")
        else:
            failed += 1
            msg = f"    [FAIL]     {name}"
            if detail:
                msg += f"  -- {detail}"
            print(msg)

    def expect_error(name, exc_type, fn):
        global passed, failed
        try:
            fn()
            failed += 1
            print(f"    [FAIL]     {name}  -- no error raised")
        except exc_type:
            passed += 1
            print(f"    [PASS]     {name}")
        except Exception as e:
            failed += 1
            print(f"    [FAIL]     {name}  -- wrong error: {type(e).__name__}: {e}")

    print("=" * 60)
    print("  training/train_config.py -- smoke test")
    print("=" * 60)

    # -- 1. Default construction -----------------------------------------------
    print("\n  1. Default construction...")
    cfg = TrainConfig()
    check("default creates", cfg is not None)
    check("state is CREATED", cfg.state == ConfigState.CREATED)

    # -- 2. Validation ---------------------------------------------------------
    print("\n  2. Validation...")
    cfg.validate()
    check("validate() succeeds", cfg.state == ConfigState.VALIDATED)
    check("optimizer normalized", cfg.optimizer == "adamw")
    check("device normalized", cfg.device == "auto")

    # -- 3. Normalization ------------------------------------------------------
    print("\n  3. Normalization...")
    cfg2 = TrainConfig(optimizer=" AdamW ", scheduler=" Cosine ", device=" CPU ")
    cfg2.validate()
    check("optimizer ' AdamW ' -> 'adamw'", cfg2.optimizer == "adamw")
    check("scheduler ' Cosine ' -> 'cosine'", cfg2.scheduler == "cosine")
    check("device ' CPU ' -> 'cpu'", cfg2.device == "cpu")

    # -- 4. Invalid field rejection --------------------------------------------
    print("\n  4. Invalid field rejection...")
    expect_error("epochs=0", TrainConfigError,
                 lambda: TrainConfig(epochs=0).validate())
    expect_error("epochs=True", TrainConfigError,
                 lambda: TrainConfig(epochs=True).validate())
    expect_error("batch_size=-1", TrainConfigError,
                 lambda: TrainConfig(batch_size=-1).validate())
    expect_error("batch_size=True", TrainConfigError,
                 lambda: TrainConfig(batch_size=True).validate())
    expect_error("lr=-0.01", TrainConfigError,
                 lambda: TrainConfig(learning_rate=-0.01).validate())
    expect_error("lr=NaN", TrainConfigError,
                 lambda: TrainConfig(learning_rate=float("nan")).validate())
    expect_error("weight_decay=-1", TrainConfigError,
                 lambda: TrainConfig(weight_decay=-1).validate())
    expect_error("optimizer='magic'", TrainConfigError,
                 lambda: TrainConfig(optimizer="magic").validate())
    expect_error("scheduler='turbo'", TrainConfigError,
                 lambda: TrainConfig(scheduler="turbo").validate())
    expect_error("loss='cross_entropy'", TrainConfigError,
                 lambda: TrainConfig(loss_name="cross_entropy").validate())
    expect_error("device='tpu'", TrainConfigError,
                 lambda: TrainConfig(device="tpu").validate())
    expect_error("gradient_clip=-1", TrainConfigError,
                 lambda: TrainConfig(gradient_clip=-1.0).validate())
    expect_error("gradient_clip=True", TrainConfigError,
                 lambda: TrainConfig(gradient_clip=True).validate())
    expect_error("experiment_name=''", TrainConfigError,
                 lambda: TrainConfig(experiment_name="").validate())
    expect_error("grad_accum=0", TrainConfigError,
                 lambda: TrainConfig(gradient_accumulation_steps=0).validate())

    # -- 5. Dependency rules ---------------------------------------------------
    print("\n  5. Dependency rules...")
    d1 = TrainConfig(scheduler="none", warmup_epochs=5)
    d1.validate()
    check("scheduler=none -> warmup=0", d1.warmup_epochs == 0)

    d2 = TrainConfig(resume=False, resume_checkpoint="some.pt")
    d2.validate()
    check("resume=False -> checkpoint=None", d2.resume_checkpoint is None)

    expect_error("resume=True, no checkpoint", TrainConfigError,
                 lambda: TrainConfig(resume=True, resume_checkpoint=None).validate())

    expect_error("warmup >= epochs", TrainConfigError,
                 lambda: TrainConfig(epochs=10, warmup_epochs=10).validate())

    # -- 6. Override -----------------------------------------------------------
    print("\n  6. Override...")
    o1 = TrainConfig()
    o1.validate()
    o1.apply_overrides({"learning_rate": 5e-4, "epochs": 100})
    check("override applied", o1.learning_rate == 5e-4 and o1.epochs == 100)
    check("state is OVERRIDDEN", o1.state == ConfigState.OVERRIDDEN)

    expect_error("protected override '_state'", TrainConfigError,
                 lambda: TrainConfig().validate().apply_overrides({"_state": "x"}))
    expect_error("unknown override 'magic'", TrainConfigError,
                 lambda: TrainConfig().validate().apply_overrides({"magic": 1}))

    # -- 7. Freeze -------------------------------------------------------------
    print("\n  7. Freeze...")
    f1 = TrainConfig()
    f1.validate().freeze()
    check("state is FROZEN", f1.state == ConfigState.FROZEN)
    check("is_frozen=True", f1.is_frozen)

    expect_error("modify after freeze", ConfigFrozenError,
                 lambda: setattr(f1, "learning_rate", 0.1))
    expect_error("override after freeze", ConfigFrozenError,
                 lambda: f1.apply_overrides({"epochs": 99}))
    expect_error("freeze before validate", TrainConfigError,
                 lambda: TrainConfig().freeze())

    # -- 8. State transitions --------------------------------------------------
    print("\n  8. State transitions...")
    s = TrainConfig()
    check("initial CREATED", s.state == ConfigState.CREATED)
    s.validate()
    check("after validate: VALIDATED", s.state == ConfigState.VALIDATED)
    s.apply_overrides({"epochs": 20})
    check("after override: OVERRIDDEN", s.state == ConfigState.OVERRIDDEN)
    s.freeze()
    check("after freeze: FROZEN", s.state == ConfigState.FROZEN)

    # -- 9. Copy safety --------------------------------------------------------
    print("\n  9. Copy safety...")
    c1 = TrainConfig(epochs=25)
    c1.validate()
    c2 = copy.deepcopy(c1)
    object.__setattr__(c2, "_frozen", False)
    object.__setattr__(c2, "_state", ConfigState.VALIDATED)
    c2.epochs = 99
    c2.validate()
    check("deepcopy independent", c1.epochs == 25 and c2.epochs == 99)

    # -- 10. Factory -----------------------------------------------------------
    print("\n  10. Factory function...")
    fc = build_train_config(epochs=30, learning_rate=2e-4)
    check("factory returns validated", fc.state == ConfigState.VALIDATED)
    check("factory values correct", fc.epochs == 30 and fc.learning_rate == 2e-4)

    # -- 11. Summary -----------------------------------------------------------
    print("\n  11. Summary output...")
    s_out = build_train_config().summary()
    check("summary does not crash", isinstance(s_out, str) and len(s_out) > 50)

    # -- 12. to_dict + as_dict -------------------------------------------------
    print("\n  12. Serialization...")
    d = build_train_config().to_dict()
    check("to_dict returns dict", isinstance(d, dict))
    check("to_dict has epochs", "epochs" in d)
    check("to_dict has _state", d["_state"] == "VALIDATED")
    ad = build_train_config().as_dict()
    check("as_dict matches to_dict", ad == d)

    # -- 13. Checkpoint traversal ----------------------------------------------
    print("\n  13. Checkpoint path containment...")
    expect_error("ckpt ../checkpoints_evil/m.pt", TrainConfigError,
                 lambda: TrainConfig(resume=True,
                     resume_checkpoint="../checkpoints_evil/model.pt").validate())
    expect_error("ckpt ../checkpoints2/m.pt", TrainConfigError,
                 lambda: TrainConfig(resume=True,
                     resume_checkpoint="../checkpoints2/model.pt").validate())
    expect_error("ckpt ../../outside/m.pt", TrainConfigError,
                 lambda: TrainConfig(resume=True,
                     resume_checkpoint="../../outside/model.pt").validate())
    # Valid checkpoint references (do not need to exist on disk)
    v1 = TrainConfig(resume=True, resume_checkpoint="best.pt")
    v1.validate()
    check("ckpt best.pt accepted", v1.resume_checkpoint == "best.pt")

    # -- 14. Frozen internal mutation ------------------------------------------
    print("\n  14. Frozen internal immutability...")
    fi = TrainConfig()
    fi.validate().freeze()
    expect_error("frozen _state mutation", ConfigFrozenError,
                 lambda: setattr(fi, "_state", ConfigState.CREATED))
    expect_error("frozen _frozen mutation", ConfigFrozenError,
                 lambda: setattr(fi, "_frozen", False))
    expect_error("frozen validate()", ConfigFrozenError,
                 lambda: fi.validate())
    expect_error("frozen apply_overrides", ConfigFrozenError,
                 lambda: fi.apply_overrides({"epochs": 99}))

    # -- 15. Illegal state transitions -----------------------------------------
    print("\n  15. Illegal state transitions...")
    expect_error("CREATED -> freeze", TrainConfigError,
                 lambda: TrainConfig().freeze())

    # -- 16. validation_dataset_name -------------------------------------------
    print("\n  16. validation_dataset_name...")
    vn1 = TrainConfig(validation_dataset_name=None)
    vn1.validate()
    check("val_ds None accepted", vn1.validation_dataset_name is None)
    vn2 = TrainConfig(validation_dataset_name="sample_100")
    vn2.validate()
    check("val_ds 'sample_100' accepted", vn2.validation_dataset_name == "sample_100")
    # Empty/whitespace strings normalize to None (valid)
    vn3 = TrainConfig(validation_dataset_name="")
    vn3.validate()
    check("val_ds '' normalizes to None", vn3.validation_dataset_name is None)
    vn4 = TrainConfig(validation_dataset_name="   ")
    vn4.validate()
    check("val_ds '   ' normalizes to None", vn4.validation_dataset_name is None)
    # Invalid types are rejected
    expect_error("val_ds bool", TrainConfigError,
                 lambda: TrainConfig(validation_dataset_name=True).validate())
    expect_error("val_ds list", TrainConfigError,
                 lambda: TrainConfig(validation_dataset_name=[]).validate())

    # -- 17. validation_split --------------------------------------------------
    print("\n  17. validation_split...")
    vs1 = TrainConfig(validation_split=0.15)
    vs1.validate()
    check("val_split 0.15 accepted", vs1.validation_split == 0.15)
    expect_error("val_split=0", TrainConfigError,
                 lambda: TrainConfig(validation_split=0.0).validate())
    expect_error("val_split=1", TrainConfigError,
                 lambda: TrainConfig(validation_split=1.0).validate())
    expect_error("val_split=-0.1", TrainConfigError,
                 lambda: TrainConfig(validation_split=-0.1).validate())
    expect_error("val_split=True", TrainConfigError,
                 lambda: TrainConfig(validation_split=True).validate())
    expect_error("val_split=NaN", TrainConfigError,
                 lambda: TrainConfig(validation_split=float('nan')).validate())
    expect_error("val_split='0.2'", TrainConfigError,
                 lambda: TrainConfig(validation_split='0.2').validate())

    # -- 18. train_all + train_datasets ----------------------------------------
    print("\n  18. train_all + train_datasets...")
    ta1 = TrainConfig(train_all=True)
    ta1.validate()
    check("train_all=True accepted", ta1.train_all is True)
    td1 = TrainConfig(train_datasets=("ds1", "ds2"))
    td1.validate()
    check("train_datasets accepted", td1.train_datasets == ("ds1", "ds2"))
    # List normalizes to tuple
    td2 = TrainConfig(train_datasets=[" ds1 ", "ds2"])
    td2.validate()
    check("list normalizes to tuple", td2.train_datasets == ("ds1", "ds2"))
    # Duplicates rejected
    expect_error("duplicate train_datasets", TrainConfigError,
                 lambda: TrainConfig(train_datasets=("a", "b", "a")).validate())
    # Non-string entry
    expect_error("non-string train_datasets", TrainConfigError,
                 lambda: TrainConfig(train_datasets=(123,)).validate())
    # Empty string entry
    expect_error("empty string train_datasets", TrainConfigError,
                 lambda: TrainConfig(train_datasets=("",)).validate())
    expect_error("train_all=1 (non-bool)", TrainConfigError,
                 lambda: TrainConfig(train_all=1).validate())

    # -- 19. New fields in summary + as_dict -----------------------------------
    print("\n  19. New fields in summary/as_dict...")
    tc19 = build_train_config(train_all=True, train_datasets=("x",))
    s19 = tc19.summary()
    check("summary has Val Split", "Val Split" in s19)
    check("summary has Train All", "Train All" in s19)
    d19 = tc19.as_dict()
    check("as_dict has validation_split", "validation_split" in d19)
    check("as_dict has train_all", "train_all" in d19)
    check("as_dict has train_datasets", "train_datasets" in d19)

    # -- Final -----------------------------------------------------------------
    total = passed + failed
    print(f"\n{'=' * 60}")
    if failed == 0:
        print(f"  [PASS]  {passed}/{total} checks passed")
    else:
        print(f"  [FAIL]  {passed}/{total} passed, {failed} failed")
    print("=" * 60)

    sys.exit(1 if failed > 0 else 0)
