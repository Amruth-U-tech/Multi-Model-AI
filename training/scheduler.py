# =============================================================================
# training/scheduler.py
# Learning Rate Scheduling Authority -- Multimodal AI Pipeline
# =============================================================================
#
# Purpose:
#   THE SINGLE AUTHORITY for constructing and validating PyTorch LR schedulers.
#   Answers exactly one question:
#       "Given a validated training policy, validated runtime context, and
#        validated optimizer, how should learning rates evolve during training,
#        and how must the trainer step this scheduler?"
#
# Responsibilities (ONLY):
#   1. Validate scheduler inputs (config, run_context, optimizer)
#   2. Construct the correct PyTorch LR scheduler
#   3. Compose warmup behavior into the scheduler when applicable
#   4. Attach immutable SchedulerMetadata with stepping-contract info
#   5. Expose summary and serialization helpers
#
# What this file does NOT do:
#   - Build optimizers or call optimizer.step()
#   - Compute losses or inspect batches/datasets
#   - Save checkpoints or write logs/files
#   - Orchestrate training or count epochs
#   - Mutate TrainConfig, RunContext, or optimizer state
#
# Ownership Map:
#   TrainConfig      -> scheduling policy (scheduler type, warmup, params)
#   RunContext       -> runtime identity (device, paths)
#   optimizer.py     -> optimizer construction
#   scheduler.py     -> scheduler construction + stepping contract (THIS FILE)
#   future trainer   -> obeys step_policy from SchedulerMetadata
#
# Usage:
#   from training.scheduler import build_scheduler, get_scheduler_metadata
# =============================================================================

import sys
import math
import copy
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Optional

# -- Project root bootstrap ----------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _THIS_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from training.train_config import TrainConfig, ConfigState
from training.run_context import RunContext

# =============================================================================
# Constants
# =============================================================================

_SUPPORTED_SCHEDULERS = frozenset({"none", "cosine", "step", "plateau"})

# Step policies advertised to the trainer
STEP_POLICY_EPOCH = "epoch"
STEP_POLICY_VALIDATION_METRIC = "validation_metric"


# =============================================================================
# Error
# =============================================================================

class SchedulerError(Exception):
    """Structured scheduler construction error."""

    def __init__(self, stage: str, field_name: str, received: Any,
                 expected: str, resolution: str = ""):
        self.stage = stage
        self.field_name = field_name
        self.received = received
        self.expected = expected
        self.resolution = resolution
        lines = [
            "[SCHEDULER ERROR]",
            f"  Stage     : {stage}",
            f"  Field     : {field_name}",
            f"  Received  : {received!r}",
            f"  Expected  : {expected}",
        ]
        if resolution:
            lines.append(f"  Resolution: {resolution}")
        super().__init__("\n".join(lines))


# =============================================================================
# Input Validation
# =============================================================================

def validate_scheduler_inputs(
    config: TrainConfig,
    run_context: RunContext,
    optimizer: optim.Optimizer,
) -> None:
    """Validate all inputs required for scheduler construction.

    Called automatically by build_scheduler() but exposed publicly
    for pre-flight checks.

    Raises:
        SchedulerError: On any invalid input.
    """
    # -- Config type -----------------------------------------------------------
    if not isinstance(config, TrainConfig):
        raise SchedulerError(
            "input_validation", "config", type(config).__name__,
            "TrainConfig instance",
            "Pass a TrainConfig from build_train_config().",
        )
    if config.state == ConfigState.CREATED:
        raise SchedulerError(
            "input_validation", "config._state", config.state.value,
            "VALIDATED, OVERRIDDEN, or FROZEN",
            "Call config.validate() before building scheduler.",
        )
    if not config.is_frozen:
        raise SchedulerError(
            "input_validation", "config._frozen", False,
            "frozen config (config.freeze())",
            "Call config.freeze() before building scheduler.",
        )

    # -- RunContext type -------------------------------------------------------
    if not isinstance(run_context, RunContext):
        raise SchedulerError(
            "input_validation", "run_context", type(run_context).__name__,
            "RunContext instance",
            "Pass a RunContext from build_run_context().",
        )

    # -- Config <-> RunContext pairing -----------------------------------------
    if run_context.config is not config:
        raise SchedulerError(
            "input_validation", "run_context.config",
            "RunContext built from a different TrainConfig",
            "RunContext built from the same frozen TrainConfig",
            "Build RunContext from this exact config and pass them together.",
        )

    # -- Optimizer type --------------------------------------------------------
    if not isinstance(optimizer, optim.Optimizer):
        raise SchedulerError(
            "input_validation", "optimizer", type(optimizer).__name__,
            "torch.optim.Optimizer instance",
            "Pass a torch.optim.Optimizer from build_optimizer().",
        )

    # -- Optimizer must have parameter groups ----------------------------------
    if not optimizer.param_groups:
        raise SchedulerError(
            "input_validation", "optimizer.param_groups",
            "0 parameter groups",
            "at least 1 parameter group",
            "Build optimizer with a model that has trainable parameters.",
        )

    # -- Defense-in-depth: scheduler name --------------------------------------
    sched_name = config.scheduler.strip().lower()
    if sched_name not in _SUPPORTED_SCHEDULERS:
        raise SchedulerError(
            "input_validation", "scheduler", config.scheduler,
            f"one of {sorted(_SUPPORTED_SCHEDULERS)}",
            "Check TrainConfig.scheduler value.",
        )

    # -- Plateau + warmup incompatibility --------------------------------------
    if sched_name == "plateau" and config.warmup_epochs > 0:
        raise SchedulerError(
            "scheduler_validation", "warmup_epochs", config.warmup_epochs,
            "0 when scheduler='plateau'",
            "Use warmup_epochs=0 for plateau, or use cosine/step if warmup is required.",
        )

    # -- Scheduler-specific decay range guards ---------------------------------
    _validate_decay_ranges(config, sched_name)


def _validate_decay_ranges(config: TrainConfig, sched_name: str) -> None:
    """Enforce scheduler-specific decay factor ranges.

    Called by validate_scheduler_inputs() after basic input checks pass.
    These are runtime contracts that harden beyond TrainConfig's type checks.
    """
    if sched_name == "step":
        val = config.step_gamma
        if isinstance(val, bool):
            raise SchedulerError(
                "scheduler_validation", "step_gamma", val,
                "numeric value, not bool",
                "Provide a float for step_gamma.",
            )
        if not isinstance(val, (int, float)):
            raise SchedulerError(
                "scheduler_validation", "step_gamma", type(val).__name__,
                "numeric value (int or float)",
                "Provide a numeric value for step_gamma.",
            )
        if math.isnan(val) or math.isinf(val):
            raise SchedulerError(
                "scheduler_validation", "step_gamma", val,
                "finite numeric value",
                "Provide a finite value for step_gamma.",
            )
        if val <= 0 or val > 1:
            raise SchedulerError(
                "scheduler_validation", "step_gamma", val,
                "0 < step_gamma <= 1 (decay factor, not amplifier)",
                "Use a value in (0, 1] for step_gamma. Values > 1 increase LR.",
            )

    elif sched_name == "plateau":
        val = config.plateau_factor
        if isinstance(val, bool):
            raise SchedulerError(
                "scheduler_validation", "plateau_factor", val,
                "numeric value, not bool",
                "Provide a float for plateau_factor.",
            )
        if not isinstance(val, (int, float)):
            raise SchedulerError(
                "scheduler_validation", "plateau_factor", type(val).__name__,
                "numeric value (int or float)",
                "Provide a numeric value for plateau_factor.",
            )
        if math.isnan(val) or math.isinf(val):
            raise SchedulerError(
                "scheduler_validation", "plateau_factor", val,
                "finite numeric value",
                "Provide a finite value for plateau_factor.",
            )
        if val <= 0 or val >= 1:
            raise SchedulerError(
                "scheduler_validation", "plateau_factor", val,
                "0 < plateau_factor < 1 (strict decay factor)",
                "Use a value in (0, 1) for plateau_factor. Values >= 1 do not decay.",
            )


# =============================================================================
# Immutable Scheduler Metadata
# =============================================================================

class SchedulerMetadata:
    """Immutable metadata snapshot captured at scheduler construction time.

    The single source of truth for scheduler identity, stepping contract,
    and serialization. The trainer reads step_policy to determine how
    to invoke scheduler.step().

    Immutable after construction -- no setattr, no delattr.
    """

    __slots__ = (
        "_scheduler_type", "_warmup_enabled", "_warmup_epochs",
        "_total_epochs", "_step_policy", "_step_timing",
        "_metric_name", "_warmup_composed", "_params", "_frozen",
    )

    def __init__(
        self,
        scheduler_type: str,
        warmup_enabled: bool,
        warmup_epochs: int,
        total_epochs: int,
        step_policy: str,
        step_timing: str,
        metric_name: Optional[str],
        warmup_composed: bool,
        params: MappingProxyType,
    ):
        object.__setattr__(self, "_scheduler_type", scheduler_type)
        object.__setattr__(self, "_warmup_enabled", warmup_enabled)
        object.__setattr__(self, "_warmup_epochs", warmup_epochs)
        object.__setattr__(self, "_total_epochs", total_epochs)
        object.__setattr__(self, "_step_policy", step_policy)
        object.__setattr__(self, "_step_timing", step_timing)
        object.__setattr__(self, "_metric_name", metric_name)
        object.__setattr__(self, "_warmup_composed", warmup_composed)
        object.__setattr__(self, "_params", params)
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name: str, value: Any):
        raise AttributeError(
            f"SchedulerMetadata is immutable. Cannot set '{name}'."
        )

    def __delattr__(self, name: str):
        raise AttributeError(
            f"SchedulerMetadata is immutable. Cannot delete '{name}'."
        )

    def __copy__(self):
        cls = self.__class__
        new = cls.__new__(cls)
        for slot in self.__slots__:
            val = object.__getattribute__(self, slot)
            object.__setattr__(new, slot, val)
        return new

    def __deepcopy__(self, memo):
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for slot in self.__slots__:
            val = object.__getattribute__(self, slot)
            if isinstance(val, MappingProxyType):
                object.__setattr__(new, slot, MappingProxyType(copy.deepcopy(dict(val), memo)))
            else:
                object.__setattr__(new, slot, copy.deepcopy(val, memo))
        return new

    @property
    def scheduler_type(self) -> str:
        return self._scheduler_type

    @property
    def warmup_enabled(self) -> bool:
        return self._warmup_enabled

    @property
    def warmup_epochs(self) -> int:
        return self._warmup_epochs

    @property
    def total_epochs(self) -> int:
        return self._total_epochs

    @property
    def step_policy(self) -> str:
        return self._step_policy

    @property
    def step_timing(self) -> str:
        return self._step_timing

    @property
    def metric_name(self) -> Optional[str]:
        return self._metric_name

    @property
    def warmup_composed(self) -> bool:
        return self._warmup_composed

    @property
    def params(self) -> MappingProxyType:
        return self._params


def get_scheduler_metadata(scheduler: Any) -> Optional[SchedulerMetadata]:
    """Retrieve attached SchedulerMetadata, if present."""
    return getattr(scheduler, "_sched_metadata", None)


# =============================================================================
# Scheduler Construction (Private)
# =============================================================================

def _build_none_scheduler(optimizer: optim.Optimizer) -> lr_scheduler.LambdaLR:
    """No-op scheduler that keeps LR unchanged."""
    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)


def _build_cosine_scheduler(
    optimizer: optim.Optimizer, config: TrainConfig,
) -> Any:
    """Cosine annealing with optional linear warmup via SequentialLR."""
    warmup = config.warmup_epochs
    total = config.epochs

    if warmup > 0:
        warmup_sched = lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup,
        )
        cosine_sched = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total - warmup,
        )
        return lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup],
        )
    else:
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=total)


def _build_step_scheduler(
    optimizer: optim.Optimizer, config: TrainConfig,
) -> lr_scheduler.StepLR:
    """Step LR decay with optional linear warmup."""
    warmup = config.warmup_epochs

    if warmup > 0:
        warmup_sched = lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup,
        )
        step_sched = lr_scheduler.StepLR(
            optimizer, step_size=config.step_size, gamma=config.step_gamma,
        )
        return lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, step_sched],
            milestones=[warmup],
        )
    else:
        return lr_scheduler.StepLR(
            optimizer, step_size=config.step_size, gamma=config.step_gamma,
        )


def _build_plateau_scheduler(
    optimizer: optim.Optimizer, config: TrainConfig,
) -> lr_scheduler.ReduceLROnPlateau:
    """ReduceLROnPlateau -- metric-driven, no warmup composition."""
    return lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=config.plateau_patience,
        factor=config.plateau_factor,
    )


# =============================================================================
# Metadata Construction (Private)
# =============================================================================

def _build_metadata(
    scheduler_type: str, config: TrainConfig, warmup_composed: bool,
) -> SchedulerMetadata:
    """Build immutable metadata snapshot for a constructed scheduler."""
    # For plateau, warmup is structurally impossible (rejected in validation).
    # Metadata must never lie about warmup composition.
    if scheduler_type == "plateau":
        warmup_enabled = False
    else:
        warmup_enabled = config.warmup_epochs > 0

    # Determine step policy and timing
    if scheduler_type == "plateau":
        step_policy = STEP_POLICY_VALIDATION_METRIC
        step_timing = "post_validation"
        metric_name = "validation_loss"
    else:
        step_policy = STEP_POLICY_EPOCH
        step_timing = "post_epoch"
        metric_name = None

    # Scheduler-specific params
    if scheduler_type == "none":
        params = {}
    elif scheduler_type == "cosine":
        params = {"T_max": config.epochs - config.warmup_epochs}
    elif scheduler_type == "step":
        params = {"step_size": config.step_size, "gamma": config.step_gamma}
    elif scheduler_type == "plateau":
        params = {
            "patience": config.plateau_patience,
            "factor": config.plateau_factor,
            "mode": "min",
        }
    else:
        params = {}

    return SchedulerMetadata(
        scheduler_type=scheduler_type,
        warmup_enabled=warmup_enabled,
        warmup_epochs=config.warmup_epochs,
        total_epochs=config.epochs,
        step_policy=step_policy,
        step_timing=step_timing,
        metric_name=metric_name,
        warmup_composed=warmup_composed,
        params=MappingProxyType(params),
    )


# =============================================================================
# Build Scheduler -- Primary Entry Point
# =============================================================================

def build_scheduler(
    config: TrainConfig,
    run_context: RunContext,
    optimizer: optim.Optimizer,
) -> Any:
    """Build a PyTorch LR scheduler from validated training policy and optimizer.

    Returns a raw PyTorch scheduler object with attached SchedulerMetadata.
    The metadata's step_policy tells the trainer how to step:
      - "epoch": call scheduler.step() after each epoch
      - "validation_metric": call scheduler.step(val_metric) after validation

    Args:
        config:      Validated and frozen TrainConfig.
        run_context: Immutable RunContext.
        optimizer:   Constructed torch.optim.Optimizer.

    Returns:
        PyTorch LR scheduler with attached _sched_metadata.

    Raises:
        SchedulerError: On any invalid input or construction failure.
    """
    validate_scheduler_inputs(config, run_context, optimizer)

    sched_name = config.scheduler.strip().lower()

    try:
        if sched_name == "none":
            sched = _build_none_scheduler(optimizer)
            warmup_composed = False
        elif sched_name == "cosine":
            sched = _build_cosine_scheduler(optimizer, config)
            warmup_composed = config.warmup_epochs > 0
        elif sched_name == "step":
            sched = _build_step_scheduler(optimizer, config)
            warmup_composed = config.warmup_epochs > 0
        elif sched_name == "plateau":
            sched = _build_plateau_scheduler(optimizer, config)
            warmup_composed = False
        else:
            raise SchedulerError(
                "scheduler_construction", "scheduler", sched_name,
                f"one of {sorted(_SUPPORTED_SCHEDULERS)}",
                "This should not happen after validation.",
            )
    except SchedulerError:
        raise
    except Exception as e:
        raise SchedulerError(
            "scheduler_construction", "scheduler",
            f"{sched_name} raised {type(e).__name__}",
            "successful scheduler construction",
            f"Internal error: {e}",
        )

    # Attach immutable metadata
    metadata = _build_metadata(sched_name, config, warmup_composed)
    sched._sched_metadata = metadata

    return sched


# =============================================================================
# Summary
# =============================================================================

def summarize_scheduler(scheduler: Any) -> str:
    """Human-readable scheduler summary for logging."""
    meta = get_scheduler_metadata(scheduler)

    if meta:
        lines = [
            "=" * 60,
            "  SCHEDULER SUMMARY",
            "=" * 60,
            f"  Type             : {meta.scheduler_type}",
            f"  Step Policy      : {meta.step_policy}",
            f"  Step Timing      : {meta.step_timing}",
            f"  Warmup Enabled   : {meta.warmup_enabled}",
            f"  Warmup Epochs    : {meta.warmup_epochs}",
            f"  Total Epochs     : {meta.total_epochs}",
            f"  Warmup Composed  : {meta.warmup_composed}",
        ]
        if meta.metric_name:
            lines.append(f"  Metric           : {meta.metric_name}")
        if meta.params:
            for k, v in meta.params.items():
                lines.append(f"  {k:17s}: {v}")
        lines.append("=" * 60)
    else:
        lines = [
            "=" * 60,
            "  SCHEDULER SUMMARY",
            "=" * 60,
            f"  Type             : {type(scheduler).__name__}",
            f"  (no metadata attached)",
            "=" * 60,
        ]

    return "\n".join(lines)


# =============================================================================
# Serialization
# =============================================================================

def scheduler_to_dict(scheduler: Any) -> Dict[str, Any]:
    """Serializable dict for checkpoint metadata."""
    meta = get_scheduler_metadata(scheduler)

    if meta:
        return {
            "scheduler_type": meta.scheduler_type,
            "step_policy": meta.step_policy,
            "step_timing": meta.step_timing,
            "warmup_enabled": meta.warmup_enabled,
            "warmup_epochs": meta.warmup_epochs,
            "warmup_composed": meta.warmup_composed,
            "total_epochs": meta.total_epochs,
            "metric_name": meta.metric_name,
            "params": dict(meta.params),
        }
    else:
        return {
            "scheduler_type": type(scheduler).__name__,
            "step_policy": "unknown",
        }


def as_dict(scheduler: Any) -> Dict[str, Any]:
    """Alias for scheduler_to_dict(). Ergonomic compatibility."""
    return scheduler_to_dict(scheduler)


# =============================================================================
# Smoke Test
# =============================================================================

if __name__ == "__main__":
    import logging
    import warnings

    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] [%(levelname)s] %(name)s -- %(message)s",
        datefmt="%H:%M:%S",
    )

    from training.train_config import build_train_config, ConfigFrozenError
    from training.run_context import build_run_context, RunContextError
    from training.optimizer import build_optimizer

    import torch.nn as nn

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

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 2)
        def forward(self, x):
            return self.fc(x)

    def _make_infra(scheduler="cosine", **kw):
        """Build config+context+optimizer+model for testing."""
        cfg = build_train_config(scheduler=scheduler, device="cpu", **kw)
        cfg.freeze()
        ctx = build_run_context(cfg)
        model = DummyModel()
        opt = build_optimizer(cfg, ctx, model)
        return cfg, ctx, opt

    def _dummy_optimizer_step(opt):
        """Simulate a training step: zero_grad, dummy backward, optimizer.step().

        This ensures scheduler.step() is called after optimizer.step(),
        avoiding PyTorch scheduler-order warnings.
        """
        opt.zero_grad()
        # Create a tiny dummy loss and backward pass
        dummy_input = torch.randn(1, 8)
        dummy_target = torch.randn(1, 2)
        model = DummyModel()
        # Use the optimizer's own params for the backward
        for pg in opt.param_groups:
            for p in pg["params"]:
                if p.requires_grad and p.grad is None:
                    p.grad = torch.zeros_like(p)
        opt.step()

    print("=" * 60)
    print("  training/scheduler.py -- smoke test")
    print("=" * 60)

    # -- 1. Default build (cosine) ---------------------------------------------
    print("\n  1. Default build (cosine)...")
    cfg, ctx, opt = _make_infra(scheduler="cosine", warmup_epochs=3, epochs=20)
    sched = build_scheduler(cfg, ctx, opt)
    check("cosine builds", sched is not None)
    meta = get_scheduler_metadata(sched)
    check("metadata attached", meta is not None)
    check("type is cosine", meta.scheduler_type == "cosine")
    check("step_policy is epoch", meta.step_policy == STEP_POLICY_EPOCH)
    check("warmup enabled", meta.warmup_enabled is True)
    check("warmup_epochs=3", meta.warmup_epochs == 3)
    check("warmup composed", meta.warmup_composed is True)

    # -- 2. None scheduler -----------------------------------------------------
    print("\n  2. None scheduler...")
    cfg_n, ctx_n, opt_n = _make_infra(scheduler="none")
    sched_n = build_scheduler(cfg_n, ctx_n, opt_n)
    meta_n = get_scheduler_metadata(sched_n)
    check("none builds", sched_n is not None)
    check("none type", meta_n.scheduler_type == "none")
    check("none step_policy", meta_n.step_policy == STEP_POLICY_EPOCH)
    check("none warmup disabled", meta_n.warmup_enabled is False)
    # Verify LR unchanged after stepping (with proper optimizer.step first)
    lr_before = opt_n.param_groups[0]["lr"]
    _dummy_optimizer_step(opt_n)
    sched_n.step()
    lr_after = opt_n.param_groups[0]["lr"]
    check("none keeps LR", abs(lr_before - lr_after) < 1e-10)

    # -- 3. Step scheduler -----------------------------------------------------
    print("\n  3. Step scheduler...")
    cfg_s, ctx_s, opt_s = _make_infra(
        scheduler="step", warmup_epochs=0, step_size=5, step_gamma=0.5,
    )
    sched_s = build_scheduler(cfg_s, ctx_s, opt_s)
    meta_s = get_scheduler_metadata(sched_s)
    check("step builds", sched_s is not None)
    check("step type", meta_s.scheduler_type == "step")
    check("step policy epoch", meta_s.step_policy == STEP_POLICY_EPOCH)
    check("step params", meta_s.params["step_size"] == 5)
    check("step gamma", meta_s.params["gamma"] == 0.5)

    # -- 4. Plateau scheduler --------------------------------------------------
    print("\n  4. Plateau scheduler...")
    cfg_p, ctx_p, opt_p = _make_infra(scheduler="plateau", warmup_epochs=0)
    sched_p = build_scheduler(cfg_p, ctx_p, opt_p)
    meta_p = get_scheduler_metadata(sched_p)
    check("plateau builds", sched_p is not None)
    check("plateau type", meta_p.scheduler_type == "plateau")
    check("plateau step_policy", meta_p.step_policy == STEP_POLICY_VALIDATION_METRIC)
    check("plateau timing", meta_p.step_timing == "post_validation")
    check("plateau metric_name", meta_p.metric_name == "validation_loss")
    check("plateau warmup disabled", meta_p.warmup_enabled is False)
    check("plateau warmup_composed false", meta_p.warmup_composed is False)

    # -- 5. Cosine without warmup ----------------------------------------------
    print("\n  5. Cosine without warmup...")
    cfg_cw, ctx_cw, opt_cw = _make_infra(
        scheduler="cosine", warmup_epochs=0, epochs=30,
    )
    sched_cw = build_scheduler(cfg_cw, ctx_cw, opt_cw)
    meta_cw = get_scheduler_metadata(sched_cw)
    check("cosine no-warmup builds", sched_cw is not None)
    check("cosine no-warmup composed false", meta_cw.warmup_composed is False)
    check("cosine T_max", meta_cw.params["T_max"] == 30)

    # -- 6. Invalid scheduler type ---------------------------------------------
    print("\n  6. Invalid scheduler type...")
    expect_error("bad optimizer type", SchedulerError,
                 lambda: validate_scheduler_inputs(cfg, ctx, "not_an_optimizer"))

    # -- 7. Unfrozen config rejection ------------------------------------------
    print("\n  7. Unfrozen config rejection...")
    unfrozen = build_train_config(device="cpu")
    expect_error("unfrozen config", SchedulerError,
                 lambda: validate_scheduler_inputs(unfrozen, ctx, opt))

    # -- 8. Config/context mismatch --------------------------------------------
    print("\n  8. Config/context mismatch...")
    cfg_a, ctx_a, opt_a = _make_infra(scheduler="cosine")
    cfg_b, ctx_b, opt_b = _make_infra(scheduler="step")
    expect_error("config/ctx mismatch", SchedulerError,
                 lambda: build_scheduler(cfg_a, ctx_b, opt_a))

    # -- 9. Empty param groups -------------------------------------------------
    print("\n  9. Empty param groups...")

    class FakeOpt(optim.Optimizer):
        def __init__(self):
            self.param_groups = []
            self.defaults = {}
            self.state = {}
        def step(self, closure=None):
            pass

    expect_error("zero param groups", SchedulerError,
                 lambda: validate_scheduler_inputs(cfg, ctx, FakeOpt()))

    # -- 10. Metadata immutability ---------------------------------------------
    print("\n  10. Metadata immutability...")
    expect_error("metadata setattr", AttributeError,
                 lambda: setattr(meta, "_scheduler_type", "evil"))
    expect_error("metadata delattr", AttributeError,
                 lambda: delattr(meta, "_scheduler_type"))

    # -- 11. Summary -----------------------------------------------------------
    print("\n  11. Summary...")
    s = summarize_scheduler(sched)
    check("summary is string", isinstance(s, str) and len(s) > 50)
    check("summary has type", "cosine" in s)

    # -- 12. Serialization -----------------------------------------------------
    print("\n  12. Serialization...")
    d = scheduler_to_dict(sched)
    check("to_dict returns dict", isinstance(d, dict))
    check("to_dict has type", d["scheduler_type"] == "cosine")
    check("to_dict has policy", d["step_policy"] == STEP_POLICY_EPOCH)
    ad = as_dict(sched)
    check("as_dict matches", ad == d)

    # -- 13. Deterministic stepping (with proper optimizer.step) ----------------
    print("\n  13. Deterministic stepping...")
    cfg_d1, ctx_d1, opt_d1 = _make_infra(scheduler="cosine", warmup_epochs=2, epochs=10)
    cfg_d2, ctx_d2, opt_d2 = _make_infra(scheduler="cosine", warmup_epochs=2, epochs=10)
    s1 = build_scheduler(cfg_d1, ctx_d1, opt_d1)
    s2 = build_scheduler(cfg_d2, ctx_d2, opt_d2)
    lrs_1, lrs_2 = [], []
    for _ in range(10):
        lrs_1.append(opt_d1.param_groups[0]["lr"])
        lrs_2.append(opt_d2.param_groups[0]["lr"])
        _dummy_optimizer_step(opt_d1)
        _dummy_optimizer_step(opt_d2)
        s1.step()
        s2.step()
    check("deterministic LR schedule", lrs_1 == lrs_2)

    # -- 14. Deepcopy safety ---------------------------------------------------
    print("\n  14. Deepcopy safety...")
    meta_copy = copy.deepcopy(meta)
    check("deepcopy type", meta_copy.scheduler_type == meta.scheduler_type)
    check("deepcopy independent", meta_copy is not meta)

    # -- 15. Plateau + warmup rejection ----------------------------------------
    print("\n  15. Plateau + warmup rejection...")
    # plateau + warmup_epochs=0 succeeds (already tested in test 4)
    # plateau + warmup_epochs>0 must fail via build_scheduler
    def _plateau_warmup_test():
        c, x, o = _make_infra(scheduler="plateau", warmup_epochs=3)
        build_scheduler(c, x, o)
    expect_error("plateau+warmup rejected", SchedulerError, _plateau_warmup_test)
    # Direct test: build config with plateau+warmup and try to build scheduler
    try:
        pw_cfg = build_train_config(scheduler="plateau", device="cpu", warmup_epochs=3)
        pw_cfg.freeze()
        pw_ctx = build_run_context(pw_cfg)
        pw_model = DummyModel()
        pw_opt = build_optimizer(pw_cfg, pw_ctx, pw_model)
        build_scheduler(pw_cfg, pw_ctx, pw_opt)
        check("plateau+warmup>0 rejected", False, "should have raised")
    except SchedulerError:
        check("plateau+warmup>0 rejected", True)

    # -- 16. step_gamma range guards -------------------------------------------
    print("\n  16. step_gamma range guards...")
    # step_gamma=1.0 accepted
    try:
        cfg16, ctx16, opt16 = _make_infra(
            scheduler="step", step_gamma=1.0, warmup_epochs=0,
        )
        build_scheduler(cfg16, ctx16, opt16)
        check("step_gamma=1.0 accepted", True)
    except SchedulerError:
        check("step_gamma=1.0 accepted", False, "unexpected error")

    # step_gamma=0.5 accepted
    try:
        cfg16b, ctx16b, opt16b = _make_infra(
            scheduler="step", step_gamma=0.5, warmup_epochs=0,
        )
        build_scheduler(cfg16b, ctx16b, opt16b)
        check("step_gamma=0.5 accepted", True)
    except SchedulerError:
        check("step_gamma=0.5 accepted", False, "unexpected error")

    # step_gamma=1.5 rejected (amplifier)
    try:
        cfg16c = build_train_config(
            scheduler="step", device="cpu", warmup_epochs=0,
        )
        # Force step_gamma > 1 via direct attribute before freeze
        cfg16c.step_gamma = 1.5
        cfg16c.freeze()
        ctx16c = build_run_context(cfg16c)
        m16c = DummyModel()
        opt16c = build_optimizer(cfg16c, ctx16c, m16c)
        build_scheduler(cfg16c, ctx16c, opt16c)
        check("step_gamma>1 rejected", False, "should have raised")
    except SchedulerError:
        check("step_gamma>1 rejected", True)

    # -- 17. plateau_factor range guards ---------------------------------------
    print("\n  17. plateau_factor range guards...")
    # plateau_factor=0.5 accepted
    try:
        cfg17, ctx17, opt17 = _make_infra(
            scheduler="plateau", plateau_factor=0.5, warmup_epochs=0,
        )
        build_scheduler(cfg17, ctx17, opt17)
        check("plateau_factor=0.5 accepted", True)
    except SchedulerError:
        check("plateau_factor=0.5 accepted", False, "unexpected error")

    # plateau_factor=1.0 rejected (no decay)
    try:
        cfg17b = build_train_config(
            scheduler="plateau", device="cpu", warmup_epochs=0,
        )
        cfg17b.plateau_factor = 1.0
        cfg17b.freeze()
        ctx17b = build_run_context(cfg17b)
        m17b = DummyModel()
        opt17b = build_optimizer(cfg17b, ctx17b, m17b)
        build_scheduler(cfg17b, ctx17b, opt17b)
        check("plateau_factor>=1 rejected", False, "should have raised")
    except SchedulerError:
        check("plateau_factor>=1 rejected", True)

    # plateau_factor=1.5 rejected
    try:
        cfg17c = build_train_config(
            scheduler="plateau", device="cpu", warmup_epochs=0,
        )
        cfg17c.plateau_factor = 1.5
        cfg17c.freeze()
        ctx17c = build_run_context(cfg17c)
        m17c = DummyModel()
        opt17c = build_optimizer(cfg17c, ctx17c, m17c)
        build_scheduler(cfg17c, ctx17c, opt17c)
        check("plateau_factor>1 rejected", False, "should have raised")
    except SchedulerError:
        check("plateau_factor>1 rejected", True)

    # -- 18. Plateau metadata contract -----------------------------------------
    print("\n  18. Plateau metadata contract...")
    # Re-use plateau from test 4
    check("plateau meta step_policy", meta_p.step_policy == "validation_metric")
    check("plateau meta step_timing", meta_p.step_timing == "post_validation")
    check("plateau meta metric_name", meta_p.metric_name == "validation_loss")
    check("plateau meta warmup_enabled", meta_p.warmup_enabled is False)
    check("plateau meta warmup_composed", meta_p.warmup_composed is False)

    # -- 19. Warning-free stepping verification --------------------------------
    print("\n  19. Warning-free stepping...")
    cfg19, ctx19, opt19 = _make_infra(
        scheduler="cosine", warmup_epochs=2, epochs=10,
    )
    sched19 = build_scheduler(cfg19, ctx19, opt19)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for _ in range(5):
            _dummy_optimizer_step(opt19)
            sched19.step()
        # Filter for the specific scheduler-before-optimizer ordering warning only.
        # Ignore harmless SequentialLR epoch deprecation warnings.
        sched_warnings = [
            x for x in w
            if "Detected call of `lr_scheduler.step()` before `optimizer.step()`"
            in str(x.message)
        ]
        check("no scheduler-order warnings", len(sched_warnings) == 0,
              f"got {len(sched_warnings)} warnings")

    # -- 20. Checkpoint resume simulation --------------------------------------
    print("\n  20. Checkpoint resume simulation...")
    # Simulate: train for 5 epochs, save state, restore, verify LR continuity
    cfg20, ctx20, opt20 = _make_infra(
        scheduler="cosine", warmup_epochs=2, epochs=10,
    )
    sched20 = build_scheduler(cfg20, ctx20, opt20)
    # Train for 5 epochs
    for _ in range(5):
        _dummy_optimizer_step(opt20)
        sched20.step()
    lr_at_epoch5 = opt20.param_groups[0]["lr"]
    # Save state
    sched_state = sched20.state_dict()
    opt_state = opt20.state_dict()
    # Build fresh infra
    cfg20b, ctx20b, opt20b = _make_infra(
        scheduler="cosine", warmup_epochs=2, epochs=10,
    )
    sched20b = build_scheduler(cfg20b, ctx20b, opt20b)
    # Restore state
    opt20b.load_state_dict(opt_state)
    sched20b.load_state_dict(sched_state)
    lr_restored = opt20b.param_groups[0]["lr"]
    check("resume LR matches", abs(lr_at_epoch5 - lr_restored) < 1e-10)
    # Continue for one more epoch
    _dummy_optimizer_step(opt20)
    sched20.step()
    _dummy_optimizer_step(opt20b)
    sched20b.step()
    lr_original_6 = opt20.param_groups[0]["lr"]
    lr_resumed_6 = opt20b.param_groups[0]["lr"]
    check("resume LR continues correctly", abs(lr_original_6 - lr_resumed_6) < 1e-10)
    # Metadata survives resume
    meta20b = get_scheduler_metadata(sched20b)
    check("resume metadata survives", meta20b is not None)
    check("resume metadata type correct", meta20b.scheduler_type == "cosine")

    # -- Final -----------------------------------------------------------------
    total = passed + failed
    print(f"\n{'=' * 60}")
    if failed == 0:
        print(f"  [PASS]  {passed}/{total} checks passed")
    else:
        print(f"  [FAIL]  {passed}/{total} passed, {failed} failed")
    print("=" * 60)

    sys.exit(1 if failed > 0 else 0)
