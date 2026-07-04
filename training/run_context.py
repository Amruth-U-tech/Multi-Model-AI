# =============================================================================
# training/run_context.py
# Runtime Identity Authority -- Multimodal AI Pipeline
# =============================================================================
#
# Purpose:
#   THE SINGLE READ-ONLY AUTHORITY for runtime context discovery inside the
#   training subsystem. Answers exactly one question:
#       "What environment is this training run executing in?"
#
# Responsibilities (ONLY):
#   1. Accept a validated, frozen TrainConfig
#   2. Discover runtime device, CUDA state, mixed precision support
#   3. Expose resolved paths from configs.paths
#   4. Capture creation timestamp, Python/platform/torch versions
#   5. Become immutable immediately after construction
#   6. Provide summary() and to_dict() for interoperability
#
# What this file does NOT do:
#   - Train models, build optimizers/schedulers/losses/datasets
#   - Save/load checkpoints, write logs/reports/files
#   - Mutate or override TrainConfig
#   - Create directories or perform filesystem writes
#   - Perform experiment selection or CLI parsing
#   - Cache or memoize runtime information globally
#   - Initialize CUDA contexts or allocate tensors for discovery
#   - Perform network calls or create runtime side effects
#   - Act as a singleton -- each instance is one training run
#
# Ownership Map:
#   configs.paths            -> filesystem authority
#   data_pipeline/           -> data authority
#   models/                  -> representation authority
#   training/train_config.py -> configuration authority
#   training/run_context.py  -> runtime identity authority  (THIS FILE)
#   future trainer.py        -> orchestration authority
#   future checkpoint_mgr    -> persistence authority
#   future experiment_mgr    -> override authority
#
# Dependencies (minimal, one-directional):
#   Python stdlib, torch, training.train_config, configs.paths
#
# Usage:
#   from training.train_config import build_train_config
#   from training.run_context import build_run_context
#
#   cfg = build_train_config(...)
#   cfg.freeze()
#   ctx = build_run_context(cfg)
#   ctx.device          # "cuda" or "cpu"
#   ctx.summary()       # human-readable runtime snapshot
#   ctx.as_dict()       # serializable dict for checkpoint metadata
# =============================================================================


import sys
import copy
import platform
import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# -- Project root bootstrap ----------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _THIS_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

import torch
from training.train_config import TrainConfig, ConfigState
from configs.paths import (
    PROJECT_ROOT,
    TRAINING_DIR,
    CHECKPOINT_DIR,
    EXPERIMENT_DIR,
    LOG_DIR,
)


# =============================================================================
# Error
# =============================================================================

class RunContextError(Exception):
    """Structured runtime context error.

    Raised when runtime identity discovery fails due to invalid input,
    incompatible hardware, or malformed runtime state.
    """

    def __init__(self, stage: str, field_name: str, received: Any,
                 expected: str, resolution: str = ""):
        self.stage = stage
        self.field_name = field_name
        self.received = received
        self.expected = expected
        self.resolution = resolution
        lines = [
            "[RUN CONTEXT ERROR]",
            f"  Stage     : {stage}",
            f"  Field     : {field_name}",
            f"  Received  : {received!r}",
            f"  Expected  : {expected}",
        ]
        if resolution:
            lines.append(f"  Resolution: {resolution}")
        super().__init__("\n".join(lines))


# =============================================================================
# Immutability Sentinel
# =============================================================================

class _Frozen(Exception):
    """Raised internally when attempting to mutate a frozen RunContext."""
    pass


# =============================================================================
# Device Discovery (Pure Functions -- No Side Effects)
# =============================================================================

def _resolve_device(config_device: str) -> str:
    """Resolve the config device string to an actual runtime device string.

    Rules:
        "cpu"  -> always "cpu"
        "auto" -> "cuda" if available, else "cpu"
        "cuda" -> requires CUDA availability, else fatal error

    Uses torch only to query capability flags. Does NOT initialize CUDA
    contexts or allocate tensors solely for discovery.

    Args:
        config_device: One of "auto", "cpu", "cuda".

    Returns:
        Resolved device string: "cpu" or "cuda".

    Raises:
        RunContextError: If device="cuda" but CUDA is unavailable.
    """
    if config_device == "cpu":
        return "cpu"

    cuda_available = torch.cuda.is_available()

    if config_device == "auto":
        return "cuda" if cuda_available else "cpu"

    if config_device == "cuda":
        if not cuda_available:
            raise RunContextError(
                stage="device_detection",
                field_name="device",
                received="cuda",
                expected="CUDA-enabled runtime",
                resolution="Use device='auto' or install CUDA-enabled PyTorch.",
            )
        return "cuda"

    # Should never reach here if TrainConfig validated properly,
    # but defend against impossible state.
    raise RunContextError(
        stage="device_detection",
        field_name="device",
        received=config_device,
        expected="one of: 'auto', 'cpu', 'cuda'",
        resolution="Check TrainConfig validation. This value should not reach RunContext.",
    )


def _discover_gpu_info() -> Tuple[int, Optional[str], Tuple[str, ...]]:
    """Discover GPU count and names without initializing CUDA or allocating tensors.

    Returns:
        (gpu_count, primary_gpu_name, all_gpu_names)
        If CUDA is unavailable, returns (0, None, ()).
    """
    if not torch.cuda.is_available():
        return 0, None, ()

    count = torch.cuda.device_count()
    if count == 0:
        return 0, None, ()

    names = tuple(torch.cuda.get_device_name(i) for i in range(count))
    return count, names[0] if names else None, names


def _check_mixed_precision_supported(resolved_device: str) -> bool:
    """Check if AMP mixed precision is supported on the resolved device.

    Descriptive only -- does not decide policy. The trainer decides
    what to do when mixed_precision_requested != mixed_precision_supported.

    Returns:
        True if the runtime can support torch.cuda.amp autocast on this device.
    """
    if resolved_device != "cuda":
        return False

    if not torch.cuda.is_available():
        return False

    # CUDA is available. Check for >= sm_70 (Volta+) for robust FP16 support.
    # torch.cuda.get_device_capability() is a lightweight query.
    try:
        major, minor = torch.cuda.get_device_capability(0)
        return major >= 7
    except Exception:
        # If we cannot query capability, conservatively report unsupported.
        return False


# =============================================================================
# RunContext
# =============================================================================

class RunContext:
    """Immutable runtime identity snapshot for a single training run.

    A RunContext instance always represents exactly one training run.
    No refresh methods, no resetting, no swapping configs. If something
    changes, construct a new RunContext.

    All public attributes are read-only after construction. Mutable
    collections are exposed as immutable tuples.

    Properties:
        config                   : TrainConfig (frozen reference)
        device                   : str ("cpu" or "cuda")
        device_type              : str ("cpu" or "cuda")
        torch_device             : torch.device
        cuda_available           : bool
        gpu_count                : int
        gpu_name                 : Optional[str]
        gpu_names                : tuple[str, ...]
        mixed_precision_available: bool
        mixed_precision_requested: bool
        experiment_name          : str
        project_root             : Path
        training_dir             : Path
        checkpoint_dir           : Path
        experiment_dir           : Path
        log_dir                  : Path
        created_at               : str (ISO 8601)
        python_version           : str
        platform                 : str
        torch_version            : str
    """

    __slots__ = (
        "_config",
        "_device",
        "_torch_device",
        "_cuda_available",
        "_gpu_count",
        "_gpu_name",
        "_gpu_names",
        "_mixed_precision_available",
        "_mixed_precision_requested",
        "_experiment_name",
        "_project_root",
        "_training_dir",
        "_checkpoint_dir",
        "_experiment_dir",
        "_log_dir",
        "_created_at",
        "_python_version",
        "_platform",
        "_torch_version",
        "_frozen",
    )

    def __init__(self, config: TrainConfig):
        """Construct a RunContext from a validated, frozen TrainConfig.

        Args:
            config: A TrainConfig instance that has been validated and frozen.

        Raises:
            RunContextError: If config is not a TrainConfig, not validated,
                             not frozen, or if device resolution fails.
        """
        # -- Validate input contract -------------------------------------------
        if not isinstance(config, TrainConfig):
            raise RunContextError(
                stage="input_validation",
                field_name="config",
                received=type(config).__name__,
                expected="TrainConfig instance",
                resolution="Pass a TrainConfig object from build_train_config().",
            )

        if config.state == ConfigState.CREATED:
            raise RunContextError(
                stage="input_validation",
                field_name="config._state",
                received=config.state.value,
                expected="VALIDATED, OVERRIDDEN, or FROZEN",
                resolution="Call config.validate() before building RunContext.",
            )

        if not config.is_frozen:
            raise RunContextError(
                stage="input_validation",
                field_name="config._frozen",
                received=False,
                expected="frozen config (config.freeze())",
                resolution=(
                    "Call config.freeze() before building RunContext. "
                    "RunContext requires a finalized configuration."
                ),
            )

        # -- Store config reference (already frozen, cannot be mutated) --------
        object.__setattr__(self, "_config", config)

        # -- Device discovery --------------------------------------------------
        resolved_device = _resolve_device(config.device)
        cuda_available = torch.cuda.is_available()
        gpu_count, gpu_name, gpu_names = _discover_gpu_info()
        mp_available = _check_mixed_precision_supported(resolved_device)

        object.__setattr__(self, "_device", resolved_device)
        object.__setattr__(self, "_torch_device", torch.device(resolved_device))
        object.__setattr__(self, "_cuda_available", cuda_available)
        object.__setattr__(self, "_gpu_count", gpu_count)
        object.__setattr__(self, "_gpu_name", gpu_name)
        object.__setattr__(self, "_gpu_names", gpu_names)
        object.__setattr__(self, "_mixed_precision_available", mp_available)
        object.__setattr__(self, "_mixed_precision_requested", config.mixed_precision)

        # -- Experiment name ---------------------------------------------------
        object.__setattr__(self, "_experiment_name", config.experiment_name)

        # -- Path integration (read-only references from configs.paths) --------
        _validate_path_export("PROJECT_ROOT", PROJECT_ROOT)
        _validate_path_export("TRAINING_DIR", TRAINING_DIR)
        _validate_path_export("CHECKPOINT_DIR", CHECKPOINT_DIR)
        _validate_path_export("EXPERIMENT_DIR", EXPERIMENT_DIR)
        _validate_path_export("LOG_DIR", LOG_DIR)

        object.__setattr__(self, "_project_root", PROJECT_ROOT)
        object.__setattr__(self, "_training_dir", TRAINING_DIR)
        object.__setattr__(self, "_checkpoint_dir", CHECKPOINT_DIR)
        object.__setattr__(self, "_experiment_dir", EXPERIMENT_DIR)
        object.__setattr__(self, "_log_dir", LOG_DIR)

        # -- Environment snapshot ----------------------------------------------
        object.__setattr__(
            self, "_created_at",
            datetime.datetime.now(datetime.timezone.utc).isoformat()
        )
        object.__setattr__(self, "_python_version", platform.python_version())
        object.__setattr__(self, "_platform", platform.platform())
        object.__setattr__(self, "_torch_version", torch.__version__)

        # -- Freeze (immutable from this point) --------------------------------
        object.__setattr__(self, "_frozen", True)

    # -- Immutability guard ----------------------------------------------------

    def __setattr__(self, name: str, value: Any):
        raise AttributeError(
            f"RunContext is immutable. Cannot set '{name}' after construction."
        )

    def __delattr__(self, name: str):
        raise AttributeError(
            f"RunContext is immutable. Cannot delete '{name}'."
        )

    def __copy__(self):
        """Support shallow copy by creating a new object with same slot values."""
        cls = self.__class__
        new = cls.__new__(cls)
        for slot in self.__slots__:
            object.__setattr__(new, slot, getattr(self, slot))
        return new

    def __deepcopy__(self, memo):
        """Support deep copy by deep-copying each slot value."""
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for slot in self.__slots__:
            val = getattr(self, slot)
            object.__setattr__(new, slot, copy.deepcopy(val, memo))
        return new

    # -- Public properties (read-only) -----------------------------------------

    @property
    def config(self) -> TrainConfig:
        """The frozen TrainConfig this context was built from."""
        return self._config

    @property
    def device(self) -> str:
        """Resolved device string: 'cpu' or 'cuda'."""
        return self._device

    @property
    def device_type(self) -> str:
        """Device type string (alias for device)."""
        return self._device

    @property
    def torch_device(self) -> torch.device:
        """torch.device object for the resolved device."""
        return self._torch_device

    @property
    def cuda_available(self) -> bool:
        """Whether CUDA is available on this machine."""
        return self._cuda_available

    @property
    def gpu_count(self) -> int:
        """Number of CUDA GPUs available (0 if CPU-only)."""
        return self._gpu_count

    @property
    def gpu_name(self) -> Optional[str]:
        """Name of the primary GPU, or None if CPU-only."""
        return self._gpu_name

    @property
    def gpu_names(self) -> Tuple[str, ...]:
        """Names of all available GPUs as an immutable tuple."""
        return self._gpu_names

    @property
    def mixed_precision_available(self) -> bool:
        """Whether AMP mixed precision is supported on this device."""
        return self._mixed_precision_available

    @property
    def mixed_precision_requested(self) -> bool:
        """Whether the config requested mixed precision."""
        return self._mixed_precision_requested

    @property
    def experiment_name(self) -> str:
        """Experiment name from the config."""
        return self._experiment_name

    @property
    def project_root(self) -> Path:
        """Resolved project root directory."""
        return self._project_root

    @property
    def training_dir(self) -> Path:
        """Resolved training source directory."""
        return self._training_dir

    @property
    def checkpoint_dir(self) -> Path:
        """Resolved checkpoint output directory."""
        return self._checkpoint_dir

    @property
    def experiment_dir(self) -> Path:
        """Resolved experiment output directory."""
        return self._experiment_dir

    @property
    def log_dir(self) -> Path:
        """Resolved log output directory."""
        return self._log_dir

    @property
    def created_at(self) -> str:
        """ISO 8601 UTC timestamp of when this context was created."""
        return self._created_at

    @property
    def python_version(self) -> str:
        """Python version string (e.g., '3.11.5')."""
        return self._python_version

    @property
    def platform(self) -> str:
        """Platform string (e.g., 'Windows-10-10.0.19041-SP0')."""
        return self._platform

    @property
    def torch_version(self) -> str:
        """PyTorch version string (e.g., '2.1.0+cu121')."""
        return self._torch_version

    # -- Summary ---------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable runtime identity summary for logging.

        Returns:
            Multi-line string suitable for console output or log records.
        """
        mp_status = "supported" if self._mixed_precision_available else "NOT supported"
        if self._mixed_precision_requested:
            mp_status += " (requested)"

        gpu_info = self._gpu_name or "N/A"
        if self._gpu_count > 1:
            gpu_info += f" (+{self._gpu_count - 1} more)"

        lines = [
            "=" * 60,
            "  RUNTIME CONTEXT",
            "=" * 60,
            f"  Experiment       : {self._experiment_name}",
            f"  Device           : {self._device}",
            f"  CUDA Available   : {self._cuda_available}",
            f"  GPU              : {gpu_info}",
            f"  GPU Count        : {self._gpu_count}",
            f"  Mixed Precision  : {mp_status}",
            "-" * 60,
            f"  Project Root     : {self._project_root}",
            f"  Training Dir     : {self._training_dir}",
            f"  Checkpoint Dir   : {self._checkpoint_dir}",
            f"  Experiment Dir   : {self._experiment_dir}",
            f"  Log Dir          : {self._log_dir}",
            "-" * 60,
            f"  Python           : {self._python_version}",
            f"  Platform         : {self._platform}",
            f"  PyTorch          : {self._torch_version}",
            f"  Created At       : {self._created_at}",
            "=" * 60,
        ]
        return "\n".join(lines)

    # -- Serialization ---------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serializable dict suitable for checkpoint metadata and reproducibility records.

        Returns:
            Dictionary with stable schema. Path objects are converted to
            strings for serialization friendliness. Contains no raw tensors,
            model objects, or open handles.
        """
        return {
            "experiment_name": self._experiment_name,
            "device": self._device,
            "device_type": self._device,
            "cuda_available": self._cuda_available,
            "gpu_count": self._gpu_count,
            "gpu_name": self._gpu_name,
            "gpu_names": list(self._gpu_names),
            "mixed_precision_available": self._mixed_precision_available,
            "mixed_precision_requested": self._mixed_precision_requested,
            "project_root": str(self._project_root),
            "training_dir": str(self._training_dir),
            "checkpoint_dir": str(self._checkpoint_dir),
            "experiment_dir": str(self._experiment_dir),
            "log_dir": str(self._log_dir),
            "created_at": self._created_at,
            "python_version": self._python_version,
            "platform": self._platform,
            "torch_version": self._torch_version,
        }

    def as_dict(self) -> Dict[str, Any]:
        """Alias for to_dict(). Ergonomic compatibility."""
        return self.to_dict()

    # -- Repr ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"RunContext(experiment={self._experiment_name!r}, "
            f"device={self._device!r}, "
            f"cuda={self._cuda_available}, "
            f"gpu_count={self._gpu_count}, "
            f"created_at={self._created_at!r})"
        )


# =============================================================================
# Path Validation Helper
# =============================================================================

def _validate_path_export(name: str, value: Any) -> None:
    """Validate that a configs.paths export is a usable Path object.

    Does NOT check existence or write permissions -- only type safety.

    Raises:
        RunContextError: If the value is not a Path instance.
    """
    if not isinstance(value, Path):
        raise RunContextError(
            stage="path_integration",
            field_name=name,
            received=type(value).__name__,
            expected="pathlib.Path from configs.paths",
            resolution=f"Verify that configs.paths exports '{name}' as a Path object.",
        )


# =============================================================================
# Factory
# =============================================================================

def build_run_context(config: TrainConfig) -> RunContext:
    """Build a RunContext from a validated, frozen TrainConfig.

    This is the recommended entry point. Accepts a frozen TrainConfig
    and returns an immutable RunContext capturing the current runtime
    environment.

    Args:
        config: A TrainConfig instance that has been validated and frozen.

    Returns:
        An immutable RunContext instance.

    Raises:
        RunContextError: On invalid config, device detection failure,
                         or malformed runtime state.
    """
    return RunContext(config)


# =============================================================================
# Smoke Test
# =============================================================================

if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] [%(levelname)s] %(name)s -- %(message)s",
        datefmt="%H:%M:%S",
    )

    from training.train_config import build_train_config, TrainConfigError

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
    print("  training/run_context.py -- smoke test")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. Default construction from validated + frozen config
    # -------------------------------------------------------------------------
    print("\n  1. Default construction...")
    cfg = build_train_config()
    cfg.freeze()
    ctx = build_run_context(cfg)
    check("RunContext creates", ctx is not None)
    check("config reference preserved", ctx.config is cfg)
    check("config not mutated", cfg.state == ConfigState.FROZEN)
    check("experiment_name matches", ctx.experiment_name == cfg.experiment_name)

    # -------------------------------------------------------------------------
    # 2. CPU runtime path (always works)
    # -------------------------------------------------------------------------
    print("\n  2. CPU runtime path...")
    cfg_cpu = build_train_config(device="cpu")
    cfg_cpu.freeze()
    ctx_cpu = build_run_context(cfg_cpu)
    check("CPU device resolves", ctx_cpu.device == "cpu")
    check("CPU device_type", ctx_cpu.device_type == "cpu")
    check("CPU torch_device", str(ctx_cpu.torch_device) == "cpu")

    # -------------------------------------------------------------------------
    # 3. device='auto' behavior
    # -------------------------------------------------------------------------
    print("\n  3. device='auto' behavior...")
    cfg_auto = build_train_config(device="auto")
    cfg_auto.freeze()
    ctx_auto = build_run_context(cfg_auto)
    expected_auto = "cuda" if torch.cuda.is_available() else "cpu"
    check("auto resolves correctly", ctx_auto.device == expected_auto)
    check("auto cuda_available accurate", ctx_auto.cuda_available == torch.cuda.is_available())

    # -------------------------------------------------------------------------
    # 4. Explicit CPU request
    # -------------------------------------------------------------------------
    print("\n  4. Explicit CPU when CUDA exists...")
    cfg_explicit_cpu = build_train_config(device="cpu")
    cfg_explicit_cpu.freeze()
    ctx_ec = build_run_context(cfg_explicit_cpu)
    check("explicit CPU honored", ctx_ec.device == "cpu")

    # -------------------------------------------------------------------------
    # 5. Explicit CUDA request failure on CPU-only machine
    # -------------------------------------------------------------------------
    print("\n  5. Explicit CUDA request on CPU-only machine...")
    if not torch.cuda.is_available():
        cfg_cuda_fail = build_train_config(device="cuda")
        cfg_cuda_fail.freeze()
        expect_error("CUDA on CPU-only fails", RunContextError,
                     lambda: build_run_context(cfg_cuda_fail))
    else:
        cfg_cuda_ok = build_train_config(device="cuda")
        cfg_cuda_ok.freeze()
        ctx_cuda = build_run_context(cfg_cuda_ok)
        check("CUDA request succeeds on CUDA machine", ctx_cuda.device == "cuda")
        check("GPU name populated", ctx_cuda.gpu_name is not None)
        check("GPU count >= 1", ctx_cuda.gpu_count >= 1)
        check("GPU names is tuple", isinstance(ctx_cuda.gpu_names, tuple))
        check("GPU names non-empty", len(ctx_cuda.gpu_names) >= 1)

    # -------------------------------------------------------------------------
    # 6. summary() does not crash
    # -------------------------------------------------------------------------
    print("\n  6. summary()...")
    s = ctx.summary()
    check("summary returns string", isinstance(s, str))
    check("summary is non-trivial", len(s) > 100)
    check("summary contains experiment", ctx.experiment_name in s)
    check("summary contains device", ctx.device in s)

    # -------------------------------------------------------------------------
    # 7. to_dict() returns expected keys
    # -------------------------------------------------------------------------
    print("\n  7. to_dict()...")
    d = ctx.to_dict()
    check("to_dict returns dict", isinstance(d, dict))
    expected_keys = {
        "experiment_name", "device", "device_type", "cuda_available",
        "gpu_count", "gpu_name", "gpu_names",
        "mixed_precision_available", "mixed_precision_requested",
        "project_root", "training_dir", "checkpoint_dir",
        "experiment_dir", "log_dir",
        "created_at", "python_version", "platform", "torch_version",
    }
    check("to_dict has all keys", expected_keys.issubset(d.keys()),
          f"missing: {expected_keys - d.keys()}")
    check("paths are strings", isinstance(d["project_root"], str))
    check("gpu_names is list", isinstance(d["gpu_names"], list))

    # -------------------------------------------------------------------------
    # 8. as_dict() equals to_dict()
    # -------------------------------------------------------------------------
    print("\n  8. as_dict()...")
    check("as_dict matches to_dict", ctx.as_dict() == ctx.to_dict())

    # -------------------------------------------------------------------------
    # 9. Immutable after creation
    # -------------------------------------------------------------------------
    print("\n  9. Immutability...")
    expect_error("setattr blocked", AttributeError,
                 lambda: setattr(ctx, "device", "tpu"))
    expect_error("setattr _device blocked", AttributeError,
                 lambda: setattr(ctx, "_device", "tpu"))
    expect_error("delattr blocked", AttributeError,
                 lambda: delattr(ctx, "_device"))

    # -------------------------------------------------------------------------
    # 10. Bad config type rejected
    # -------------------------------------------------------------------------
    print("\n  10. Bad config type...")
    expect_error("dict rejected", RunContextError,
                 lambda: build_run_context({"device": "cpu"}))
    expect_error("None rejected", RunContextError,
                 lambda: build_run_context(None))
    expect_error("string rejected", RunContextError,
                 lambda: build_run_context("auto"))
    expect_error("int rejected", RunContextError,
                 lambda: build_run_context(42))

    # -------------------------------------------------------------------------
    # 11. Unvalidated config rejected
    # -------------------------------------------------------------------------
    print("\n  11. Unvalidated config...")
    cfg_raw = TrainConfig()
    expect_error("CREATED config rejected", RunContextError,
                 lambda: build_run_context(cfg_raw))

    # -------------------------------------------------------------------------
    # 12. Unfrozen config rejected
    # -------------------------------------------------------------------------
    print("\n  12. Unfrozen config...")
    cfg_validated_only = build_train_config()
    # cfg_validated_only is VALIDATED but NOT frozen
    expect_error("unfrozen config rejected", RunContextError,
                 lambda: build_run_context(cfg_validated_only))

    # -------------------------------------------------------------------------
    # 13. Path exposure works
    # -------------------------------------------------------------------------
    print("\n  13. Path exposure...")
    check("project_root is Path", isinstance(ctx.project_root, Path))
    check("training_dir is Path", isinstance(ctx.training_dir, Path))
    check("checkpoint_dir is Path", isinstance(ctx.checkpoint_dir, Path))
    check("experiment_dir is Path", isinstance(ctx.experiment_dir, Path))
    check("log_dir is Path", isinstance(ctx.log_dir, Path))
    check("project_root matches", ctx.project_root == PROJECT_ROOT)
    check("checkpoint_dir matches", ctx.checkpoint_dir == CHECKPOINT_DIR)

    # -------------------------------------------------------------------------
    # 14. repr works
    # -------------------------------------------------------------------------
    print("\n  14. repr()...")
    r = repr(ctx)
    check("repr returns string", isinstance(r, str))
    check("repr contains RunContext", "RunContext" in r)
    check("repr contains experiment", ctx.experiment_name in r)

    # -------------------------------------------------------------------------
    # 15. deepcopy safety
    # -------------------------------------------------------------------------
    print("\n  15. deepcopy...")
    ctx2 = copy.deepcopy(ctx)
    check("deepcopy creates new object", ctx2 is not ctx)
    check("deepcopy device matches", ctx2.device == ctx.device)
    check("deepcopy experiment matches", ctx2.experiment_name == ctx.experiment_name)
    check("deepcopy to_dict matches", ctx2.to_dict() == ctx.to_dict())
    # Deepcopy is also immutable
    expect_error("deepcopy immutable", AttributeError,
                 lambda: setattr(ctx2, "device", "tpu"))

    # -------------------------------------------------------------------------
    # 16. No mutation of input config
    # -------------------------------------------------------------------------
    print("\n  16. Config immutability preservation...")
    check("config still frozen after ctx build", cfg.is_frozen)
    check("config state still FROZEN", cfg.state == ConfigState.FROZEN)
    check("config epochs unchanged", cfg.epochs == 50)

    # -------------------------------------------------------------------------
    # 17. Repeated construction produces stable summaries
    # -------------------------------------------------------------------------
    print("\n  17. Repeated construction stability...")
    cfg_repeat = build_train_config(device="cpu")
    cfg_repeat.freeze()
    ctx_a = build_run_context(cfg_repeat)
    ctx_b = build_run_context(cfg_repeat)
    # Device, experiment, paths should all match (timestamps will differ)
    check("repeated device stable", ctx_a.device == ctx_b.device)
    check("repeated experiment stable", ctx_a.experiment_name == ctx_b.experiment_name)
    check("repeated project_root stable", ctx_a.project_root == ctx_b.project_root)

    # -------------------------------------------------------------------------
    # 18. Mixed precision on CPU runtime
    # -------------------------------------------------------------------------
    print("\n  18. Mixed precision on CPU...")
    cfg_mp_cpu = build_train_config(device="cpu", mixed_precision=True)
    cfg_mp_cpu.freeze()
    ctx_mp_cpu = build_run_context(cfg_mp_cpu)
    check("mixed_precision_requested is True", ctx_mp_cpu.mixed_precision_requested is True)
    check("mixed_precision_available is False on CPU", ctx_mp_cpu.mixed_precision_available is False)

    # -------------------------------------------------------------------------
    # 19. Environment metadata populated
    # -------------------------------------------------------------------------
    print("\n  19. Environment metadata...")
    check("python_version non-empty", len(ctx.python_version) > 0)
    check("platform non-empty", len(ctx.platform) > 0)
    check("torch_version non-empty", len(ctx.torch_version) > 0)
    check("created_at non-empty", len(ctx.created_at) > 0)
    check("created_at is ISO format", "T" in ctx.created_at)

    # -------------------------------------------------------------------------
    # Final results
    # -------------------------------------------------------------------------
    total = passed + failed
    print(f"\n{'=' * 60}")
    if failed == 0:
        print(f"  [PASS]  {passed}/{total} checks passed")
    else:
        print(f"  [FAIL]  {passed}/{total} passed, {failed} failed")
    print("=" * 60)

    sys.exit(1 if failed > 0 else 0)
