# =============================================================================
# training/optimizer.py
# Optimization Authority -- Multimodal AI Pipeline
# =============================================================================
#
# Purpose:
#   THE SINGLE AUTHORITY for constructing and validating PyTorch optimizers
#   inside the training subsystem. Answers exactly one question:
#       "Given a validated training policy, a validated runtime context,
#        and a model, how should this model be optimized?"
#
# Responsibilities (ONLY):
#   1. Accept validated+frozen TrainConfig, immutable RunContext, nn.Module
#   2. Discover trainable parameters (model-provided groups or fallback)
#   3. Validate parameter groups for correctness
#   4. Construct the specified torch.optim.Optimizer
#   5. Provide summary and serialization helpers for interoperability
#
# What this file does NOT do:
#   - Train models (no .backward(), .step(), .zero_grad())
#   - Build or schedule learning rate policies
#   - Save or load optimizer state (checkpoint_manager owns that)
#   - Write logs, reports, or files
#   - Mutate TrainConfig or RunContext
#   - Inspect model architecture by name (architecture-independent)
#   - Discover runtime environment (RunContext owns that)
#   - Perform dataset or dataloader operations
#   - Own SHAP or explainability concerns
#   - Cache or memoize optimizer instances globally
#
# Ownership Map:
#   TrainConfig          -> optimization policy (lr, wd, optimizer name)
#   RunContext           -> runtime identity
#   torch.nn.Module      -> trainable parameters
#   optimizer.py         -> parameter-group validation + optimizer construction
#   future trainer.py    -> zero_grad(), backward(), step()
#   future scheduler.py  -> LR scheduling
#   future checkpoint_mgr -> optimizer state persistence
#
# Optimizer Defaults Policy:
#   TrainConfig owns high-level policy (lr, wd, optimizer name).
#   This file owns optimizer-specific defaults (betas, eps, momentum, nesterov).
#   These defaults are NOT user-configurable in v1 to keep the surface small.
#
# Dependencies (minimal, one-directional):
#   Python stdlib, torch, training.train_config, training.run_context
#
# Usage:
#   from training.train_config import build_train_config
#   from training.run_context import build_run_context
#   from training.optimizer import build_optimizer
#
#   cfg = build_train_config(...)
#   cfg.freeze()
#   ctx = build_run_context(cfg)
#   optimizer = build_optimizer(cfg, ctx, model)
# =============================================================================


import sys
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Optional, Tuple

# -- Project root bootstrap ----------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _THIS_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

import torch
import torch.nn as nn
import torch.optim as optim

from training.train_config import TrainConfig, ConfigState
from training.run_context import RunContext


# =============================================================================
# Constants -- Optimizer-Specific Defaults
# =============================================================================
# TrainConfig owns high-level policy.  This file owns internal optimizer knobs.

ADAMW_DEFAULTS: Dict[str, Any] = {
    "betas": (0.9, 0.999),
    "eps": 1e-8,
}

ADAM_DEFAULTS: Dict[str, Any] = {
    "betas": (0.9, 0.999),
    "eps": 1e-8,
}

SGD_DEFAULTS: Dict[str, Any] = {
    "momentum": 0.9,
    "nesterov": False,
}

_SUPPORTED_OPTIMIZERS = frozenset({"adamw", "adam", "sgd"})


# =============================================================================
# Error
# =============================================================================

class OptimizerError(Exception):
    """Structured optimizer construction error.

    Raised when optimizer building fails due to invalid inputs,
    malformed parameter groups, or impossible configuration.
    """

    def __init__(self, stage: str, field_name: str, received: Any,
                 expected: str, resolution: str = ""):
        self.stage = stage
        self.field_name = field_name
        self.received = received
        self.expected = expected
        self.resolution = resolution
        lines = [
            "[OPTIMIZER ERROR]",
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

def validate_optimizer_inputs(
    config: TrainConfig,
    run_context: RunContext,
    model: nn.Module,
) -> None:
    """Validate all inputs required for optimizer construction.

    This is called automatically by build_optimizer() but is exposed
    publicly for pre-flight checks.

    Args:
        config:      Validated and frozen TrainConfig.
        run_context: Immutable RunContext.
        model:       A torch.nn.Module with parameters.

    Raises:
        OptimizerError: On any invalid input.
    """
    # -- Config validation -----------------------------------------------------
    if not isinstance(config, TrainConfig):
        raise OptimizerError(
            stage="input_validation",
            field_name="config",
            received=type(config).__name__,
            expected="TrainConfig instance",
            resolution="Pass a TrainConfig from build_train_config().",
        )
    if config.state == ConfigState.CREATED:
        raise OptimizerError(
            stage="input_validation",
            field_name="config._state",
            received=config.state.value,
            expected="VALIDATED, OVERRIDDEN, or FROZEN",
            resolution="Call config.validate() before building optimizer.",
        )
    if not config.is_frozen:
        raise OptimizerError(
            stage="input_validation",
            field_name="config._frozen",
            received=False,
            expected="frozen config (config.freeze())",
            resolution="Call config.freeze() before building optimizer.",
        )

    # -- RunContext validation --------------------------------------------------
    if not isinstance(run_context, RunContext):
        raise OptimizerError(
            stage="input_validation",
            field_name="run_context",
            received=type(run_context).__name__,
            expected="RunContext instance",
            resolution="Pass a RunContext from build_run_context().",
        )

    # -- Config <-> RunContext pairing -----------------------------------------
    if run_context.config is not config:
        raise OptimizerError(
            stage="input_validation",
            field_name="run_context.config",
            received="RunContext built from a different TrainConfig",
            expected="RunContext built from the same frozen TrainConfig",
            resolution="Build RunContext from this exact config and pass them together.",
        )

    # -- Model validation ------------------------------------------------------
    if not isinstance(model, nn.Module):
        raise OptimizerError(
            stage="input_validation",
            field_name="model",
            received=type(model).__name__,
            expected="torch.nn.Module instance",
            resolution="Pass a torch.nn.Module (e.g., FusionModel).",
        )

    # -- Optimizer name validation (defense-in-depth) --------------------------
    opt_name = config.optimizer.strip().lower()
    if opt_name not in _SUPPORTED_OPTIMIZERS:
        raise OptimizerError(
            stage="input_validation",
            field_name="optimizer",
            received=config.optimizer,
            expected=f"one of {sorted(_SUPPORTED_OPTIMIZERS)}",
            resolution="Check TrainConfig.optimizer value.",
        )


# =============================================================================
# Parameter Group Discovery
# =============================================================================

def _discover_parameter_groups(
    config: TrainConfig,
    model: nn.Module,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Discover parameter groups from the model.

    Preferred path: model.get_optimizer_parameter_groups()
    Fallback path:  single group from model.named_parameters()

    Args:
        config: Frozen TrainConfig (for LR, WD, backbone_lr_multiplier).
        model:  The nn.Module to extract parameters from.

    Returns:
        (groups, used_model_api): list of validated group dicts and whether
        the model's own API was used.

    Raises:
        OptimizerError: On malformed groups or zero trainable parameters.
    """
    # -- Preferred: model-provided groups --------------------------------------
    if hasattr(model, "get_optimizer_parameter_groups") and callable(
        model.get_optimizer_parameter_groups
    ):
        try:
            raw_groups = model.get_optimizer_parameter_groups()
        except Exception as e:
            raise OptimizerError(
                stage="parameter_group_discovery",
                field_name="model.get_optimizer_parameter_groups()",
                received=str(e),
                expected="valid list of parameter group dicts",
                resolution="Fix model.get_optimizer_parameter_groups() implementation.",
            )

        if not isinstance(raw_groups, (list, tuple)):
            raise OptimizerError(
                stage="parameter_group_discovery",
                field_name="model.get_optimizer_parameter_groups()",
                received=type(raw_groups).__name__,
                expected="list of parameter group dicts",
                resolution="Return a list of dicts from get_optimizer_parameter_groups().",
            )

        groups = _validate_model_groups(config, raw_groups)
        return groups, True

    # -- Fallback: single group from all trainable parameters ------------------
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise OptimizerError(
            stage="parameter_group_discovery",
            field_name="model.parameters()",
            received="0 trainable parameters",
            expected="at least 1 trainable parameter",
            resolution="Unfreeze some model parameters before building optimizer.",
        )

    group = {
        "name": "all_trainable",
        "params": trainable,
        "lr": config.learning_rate,
        "weight_decay": config.weight_decay,
    }
    return [group], False


def _validate_model_groups(
    config: TrainConfig,
    raw_groups: List[Any],
) -> List[Dict[str, Any]]:
    """Validate and normalize model-provided parameter groups.

    Expected group schema:
        {
            "name": str,                    # required
            "params": iterable of Tensors,  # required
            "lr_scale": float,              # optional, multiplied by config.learning_rate
            "weight_decay": float,          # optional override
            "is_backbone": bool,            # optional, for backbone_lr_multiplier
        }

    Args:
        config:     Frozen TrainConfig.
        raw_groups: List of group dicts from the model.

    Returns:
        List of validated, optimizer-ready group dicts.

    Raises:
        OptimizerError: On any invalid group.
    """
    if not raw_groups:
        raise OptimizerError(
            stage="parameter_group_validation",
            field_name="parameter_groups",
            received="empty list",
            expected="at least 1 parameter group",
            resolution="Model must provide non-empty parameter groups.",
        )

    validated = []
    seen_param_ids: set = set()
    total_trainable = 0

    for idx, group in enumerate(raw_groups):
        label = f"group[{idx}]"

        # -- Must be dict ------------------------------------------------------
        if not isinstance(group, dict):
            raise OptimizerError(
                stage="parameter_group_validation",
                field_name=label,
                received=type(group).__name__,
                expected="dict with 'name' and 'params' keys",
                resolution="Each parameter group must be a dict.",
            )

        # -- Must have name ----------------------------------------------------
        name = group.get("name")
        if not isinstance(name, str) or not name.strip():
            raise OptimizerError(
                stage="parameter_group_validation",
                field_name=f"{label}.name",
                received=name,
                expected="non-empty string",
                resolution="Provide a human-readable 'name' for each group.",
            )
        name = name.strip()

        # -- Must have params --------------------------------------------------
        if "params" not in group:
            raise OptimizerError(
                stage="parameter_group_validation",
                field_name=f"{label}.params",
                received="missing",
                expected="iterable of torch parameters",
                resolution=f"Add 'params' to group '{name}'.",
            )

        try:
            raw_params = list(group["params"])
        except TypeError:
            raise OptimizerError(
                stage="parameter_group_validation",
                field_name=f"{label}.params",
                received=type(group["params"]).__name__,
                expected="iterable of torch parameters",
                resolution=f"Ensure 'params' in group '{name}' is iterable.",
            )

        # -- Filter to trainable only ------------------------------------------
        trainable_params = [p for p in raw_params if isinstance(p, torch.Tensor)
                           and p.requires_grad]

        # -- Duplicate detection -----------------------------------------------
        for p in trainable_params:
            pid = id(p)
            if pid in seen_param_ids:
                raise OptimizerError(
                    stage="parameter_group_validation",
                    field_name=f"{label}.params",
                    received=f"duplicate parameter in group '{name}'",
                    expected="each trainable parameter in exactly one group",
                    resolution="Remove overlapping parameters between groups.",
                )
            seen_param_ids.add(pid)

        # -- Skip empty groups (but track) -------------------------------------
        if not trainable_params:
            continue

        total_trainable += len(trainable_params)

        # -- Resolve effective LR ----------------------------------------------
        effective_lr = config.learning_rate

        # lr_scale takes priority
        lr_scale = group.get("lr_scale")
        if lr_scale is not None:
            _validate_numeric(
                f"{label}.lr_scale", lr_scale,
                allow_zero=False, stage="parameter_group_validation",
            )
            effective_lr = config.learning_rate * lr_scale

        # backbone_lr_multiplier if marked as backbone
        is_backbone = group.get("is_backbone", False)
        if is_backbone and config.backbone_lr_multiplier is not None:
            if lr_scale is None:
                # Only apply backbone multiplier if no explicit lr_scale
                effective_lr = config.learning_rate * config.backbone_lr_multiplier

        # -- Validate effective LR is positive ---------------------------------
        if effective_lr <= 0 or math.isnan(effective_lr) or math.isinf(effective_lr):
            raise OptimizerError(
                stage="parameter_group_validation",
                field_name=f"{label}.effective_lr",
                received=effective_lr,
                expected="finite positive float",
                resolution=(
                    f"Check lr_scale or backbone_lr_multiplier for group '{name}'. "
                    f"base_lr={config.learning_rate}, lr_scale={lr_scale}, "
                    f"is_backbone={is_backbone}, multiplier={config.backbone_lr_multiplier}"
                ),
            )

        # -- Resolve weight decay ----------------------------------------------
        wd = group.get("weight_decay", config.weight_decay)
        _validate_numeric(
            f"{label}.weight_decay", wd,
            allow_zero=True, stage="parameter_group_validation",
        )

        # -- Build validated group ---------------------------------------------
        validated.append({
            "name": name,
            "params": trainable_params,
            "lr": effective_lr,
            "weight_decay": wd,
        })

    # -- Must have at least one trainable parameter ----------------------------
    if total_trainable == 0:
        raise OptimizerError(
            stage="parameter_group_validation",
            field_name="parameter_groups",
            received="0 trainable parameters across all groups",
            expected="at least 1 trainable parameter",
            resolution="Unfreeze some model parameters before building optimizer.",
        )

    return validated


def _validate_numeric(
    field_name: str, value: Any, allow_zero: bool, stage: str,
) -> None:
    """Validate a numeric field is finite and non-negative (or positive).

    Raises:
        OptimizerError: If validation fails.
    """
    if isinstance(value, bool):
        raise OptimizerError(
            stage=stage, field_name=field_name, received=value,
            expected="numeric value, not bool",
            resolution=f"Provide a float for {field_name}.",
        )
    if not isinstance(value, (int, float)):
        raise OptimizerError(
            stage=stage, field_name=field_name, received=type(value).__name__,
            expected="numeric value (int or float)",
            resolution=f"Provide a numeric value for {field_name}.",
        )
    if math.isnan(value) or math.isinf(value):
        raise OptimizerError(
            stage=stage, field_name=field_name, received=value,
            expected="finite numeric value",
            resolution=f"Provide a finite value for {field_name}.",
        )
    if allow_zero:
        if value < 0:
            raise OptimizerError(
                stage=stage, field_name=field_name, received=value,
                expected="non-negative numeric value",
                resolution=f"Provide >= 0 for {field_name}.",
            )
    else:
        if value <= 0:
            raise OptimizerError(
                stage=stage, field_name=field_name, received=value,
                expected="positive numeric value",
                resolution=f"Provide > 0 for {field_name}.",
            )


# =============================================================================
# Optimizer Construction
# =============================================================================

def _build_torch_optimizer(
    opt_name: str,
    param_groups: List[Dict[str, Any]],
) -> optim.Optimizer:
    """Construct the actual torch optimizer from validated parameter groups.

    Optimizer-specific defaults (betas, eps, momentum) are owned here.
    Parameter groups already contain resolved lr and weight_decay.

    Args:
        opt_name:     Normalized optimizer name ("adamw", "adam", "sgd").
        param_groups: Validated list of parameter group dicts.

    Returns:
        torch.optim.Optimizer instance.

    Raises:
        OptimizerError: On construction failure.
    """
    # Strip non-torch keys before passing to optimizer constructor
    torch_groups = []
    for g in param_groups:
        torch_group = {
            "params": g["params"],
            "lr": g["lr"],
            "weight_decay": g["weight_decay"],
        }
        torch_groups.append(torch_group)

    try:
        if opt_name == "adamw":
            return optim.AdamW(torch_groups, **ADAMW_DEFAULTS)
        elif opt_name == "adam":
            return optim.Adam(torch_groups, **ADAM_DEFAULTS)
        elif opt_name == "sgd":
            return optim.SGD(torch_groups, **SGD_DEFAULTS)
        else:
            raise OptimizerError(
                stage="optimizer_construction",
                field_name="optimizer",
                received=opt_name,
                expected=f"one of {sorted(_SUPPORTED_OPTIMIZERS)}",
                resolution="This should not happen after validation.",
            )
    except OptimizerError:
        raise
    except Exception as e:
        raise OptimizerError(
            stage="optimizer_construction",
            field_name="optimizer",
            received=f"{opt_name} raised {type(e).__name__}",
            expected="successful optimizer construction",
            resolution=f"Internal error: {e}",
        )


# =============================================================================
# Immutable Optimizer Metadata
# =============================================================================

class OptimizerMetadata:
    """Immutable metadata snapshot captured at optimizer construction time.

    Preserves semantic group provenance (names, discovery method, backbone
    flags) that torch.optim.Optimizer does not retain. This is the single
    source of truth for summarize_optimizer() and optimizer_to_dict().

    Immutable after construction -- no setattr, no delattr.
    """

    __slots__ = (
        "_optimizer_type",
        "_groups",
        "_used_model_api",
        "_trainable_params",
        "_frozen_params",
        "_total_optimizer_params",
        "_frozen",
    )

    def __init__(
        self,
        optimizer_type: str,
        groups: Tuple[MappingProxyType, ...],
        used_model_api: bool,
        trainable_params: int,
        frozen_params: int,
        total_optimizer_params: int,
    ):
        object.__setattr__(self, "_optimizer_type", optimizer_type)
        object.__setattr__(self, "_groups", groups)
        object.__setattr__(self, "_used_model_api", used_model_api)
        object.__setattr__(self, "_trainable_params", trainable_params)
        object.__setattr__(self, "_frozen_params", frozen_params)
        object.__setattr__(self, "_total_optimizer_params", total_optimizer_params)
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name: str, value: Any):
        raise AttributeError(
            f"OptimizerMetadata is immutable. Cannot set '{name}'."
        )

    def __delattr__(self, name: str):
        raise AttributeError(
            f"OptimizerMetadata is immutable. Cannot delete '{name}'."
        )

    @property
    def optimizer_type(self) -> str:
        return self._optimizer_type

    @property
    def groups(self) -> Tuple[MappingProxyType, ...]:
        return self._groups

    @property
    def used_model_api(self) -> bool:
        return self._used_model_api

    @property
    def trainable_params(self) -> int:
        return self._trainable_params

    @property
    def frozen_params(self) -> int:
        return self._frozen_params

    @property
    def total_optimizer_params(self) -> int:
        return self._total_optimizer_params


def _build_metadata(
    optimizer: optim.Optimizer,
    param_groups: List[Dict[str, Any]],
    used_model_api: bool,
    model: nn.Module,
) -> OptimizerMetadata:
    """Build an immutable metadata snapshot from validated parameter groups.

    Called once during build_optimizer(). Captures semantic group names,
    provenance, and parameter counts before torch strips non-native keys.

    Args:
        optimizer:    Constructed torch optimizer (for class name).
        param_groups: Validated groups (still contain 'name' key).
        used_model_api: Whether model.get_optimizer_parameter_groups() was used.
        model:        The model (for trainable/frozen counts).

    Returns:
        Frozen OptimizerMetadata instance.
    """
    trainable, frozen = _count_params(model)

    group_snapshots = []
    total_opt_params = 0
    for idx, g in enumerate(param_groups):
        param_count = sum(p.numel() for p in g["params"])
        total_opt_params += param_count
        snapshot = {
            "index": idx,
            "name": g.get("name", f"group_{idx}"),
            "lr": g["lr"],
            "weight_decay": g["weight_decay"],
            "param_count": param_count,
        }
        group_snapshots.append(MappingProxyType(snapshot))

    return OptimizerMetadata(
        optimizer_type=type(optimizer).__name__,
        groups=tuple(group_snapshots),
        used_model_api=used_model_api,
        trainable_params=trainable,
        frozen_params=frozen,
        total_optimizer_params=total_opt_params,
    )


def get_optimizer_metadata(
    optimizer: optim.Optimizer,
) -> Optional[OptimizerMetadata]:
    """Retrieve attached OptimizerMetadata from an optimizer, if present.

    Returns None if the optimizer was not built by build_optimizer().
    """
    return getattr(optimizer, "_optim_metadata", None)


# =============================================================================
# Build Optimizer -- Primary Entry Point
# =============================================================================

def build_optimizer(
    config: TrainConfig,
    run_context: RunContext,
    model: nn.Module,
) -> optim.Optimizer:
    """Build a PyTorch optimizer from validated training policy and model.

    This is the recommended entry point. Accepts a frozen TrainConfig,
    an immutable RunContext, and an nn.Module. Returns a configured
    torch.optim.Optimizer ready for trainer.py consumption.

    Parameter group strategy:
      - Preferred: model.get_optimizer_parameter_groups()
      - Fallback:  single group from all trainable parameters

    Attaches immutable OptimizerMetadata to the returned optimizer as
    a private attribute (_optim_metadata). This metadata preserves
    semantic group names and provenance for summary and serialization.

    Args:
        config:      Validated and frozen TrainConfig.
        run_context: Immutable RunContext.
        model:       A torch.nn.Module with trainable parameters.

    Returns:
        torch.optim.Optimizer: Configured optimizer instance with
        attached OptimizerMetadata.

    Raises:
        OptimizerError: On any invalid input, parameter group issue,
                        or construction failure.
    """
    # -- Validate inputs -------------------------------------------------------
    validate_optimizer_inputs(config, run_context, model)

    # -- Discover and validate parameter groups --------------------------------
    param_groups, used_model_api = _discover_parameter_groups(config, model)

    # -- Construct optimizer ---------------------------------------------------
    opt_name = config.optimizer.strip().lower()
    optimizer = _build_torch_optimizer(opt_name, param_groups)

    # -- Attach immutable metadata (single source of truth) --------------------
    metadata = _build_metadata(optimizer, param_groups, used_model_api, model)
    optimizer._optim_metadata = metadata

    return optimizer


# =============================================================================
# Metadata Helpers
# =============================================================================

def _count_params(model: nn.Module) -> Tuple[int, int]:
    """Count trainable and frozen parameters in a model.

    Returns:
        (trainable_count, frozen_count)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen


# =============================================================================
# Summary (metadata-driven, single source of truth)
# =============================================================================

def summarize_optimizer(
    optimizer: optim.Optimizer,
    model: Optional[nn.Module] = None,
    config: Optional[TrainConfig] = None,
) -> str:
    """Human-readable optimizer summary for logging and debugging.

    If the optimizer was built by build_optimizer(), uses the attached
    immutable OptimizerMetadata for semantic group names and provenance.
    Falls back to torch param_groups if metadata is absent.

    Descriptive only -- no side effects.

    Args:
        optimizer: Constructed torch.optim.Optimizer.
        model:     Optional model for trainable/frozen param counts
                   (used only if metadata is absent).
        config:    Optional config for policy context.

    Returns:
        Multi-line summary string.
    """
    meta = get_optimizer_metadata(optimizer)

    # -- Header info -----------------------------------------------------------
    opt_class = meta.optimizer_type if meta else type(optimizer).__name__
    num_groups = len(meta.groups) if meta else len(optimizer.param_groups)
    total_opt_params = meta.total_optimizer_params if meta else sum(
        sum(p.numel() for p in pg["params"]) for pg in optimizer.param_groups
    )

    # -- First group LR/WD for header ------------------------------------------
    if meta and meta.groups:
        default_lr = meta.groups[0]["lr"]
        default_wd = meta.groups[0]["weight_decay"]
    elif optimizer.param_groups:
        default_lr = optimizer.param_groups[0].get("lr", "N/A")
        default_wd = optimizer.param_groups[0].get("weight_decay", "N/A")
    else:
        default_lr = "N/A"
        default_wd = "N/A"

    # -- Trainable/frozen counts -----------------------------------------------
    if meta:
        trainable_count = f"{meta.trainable_params:,}"
        frozen_count = f"{meta.frozen_params:,}"
    elif model is not None:
        t, f = _count_params(model)
        trainable_count = f"{t:,}"
        frozen_count = f"{f:,}"
    else:
        trainable_count = "N/A"
        frozen_count = "N/A"

    # -- Discovery method ------------------------------------------------------
    if meta:
        discovery = "model API" if meta.used_model_api else "fallback (all trainable)"
    else:
        discovery = "unknown"

    lines = [
        "=" * 60,
        "  OPTIMIZER SUMMARY",
        "=" * 60,
        f"  Optimizer        : {opt_class}",
    ]

    if isinstance(default_lr, float):
        lines.append(f"  Learning Rate    : {default_lr:.3e}")
    else:
        lines.append(f"  Learning Rate    : {default_lr}")

    if isinstance(default_wd, float):
        lines.append(f"  Weight Decay     : {default_wd:.3e}")
    else:
        lines.append(f"  Weight Decay     : {default_wd}")

    lines.extend([
        f"  Optimizer Params : {total_opt_params:,}",
        f"  Trainable Params : {trainable_count}",
        f"  Frozen Params    : {frozen_count}",
        f"  Parameter Groups : {num_groups}",
        f"  Discovery        : {discovery}",
    ])

    # -- Per-group detail (metadata-driven when available) ----------------------
    if meta:
        for gm in meta.groups:
            lr = gm["lr"]
            wd = gm["weight_decay"]
            lines.append("-" * 60)
            lines.append(f"  Group {gm['index'] + 1:<3}  {gm['name']}")
            if isinstance(lr, float):
                lines.append(f"    LR             : {lr:.3e}")
            else:
                lines.append(f"    LR             : {lr}")
            if isinstance(wd, float):
                lines.append(f"    Weight Decay   : {wd:.3e}")
            else:
                lines.append(f"    Weight Decay   : {wd}")
            lines.append(f"    Params         : {gm['param_count']:,}")
    else:
        for idx, pg in enumerate(optimizer.param_groups):
            param_count = sum(p.numel() for p in pg["params"])
            lr = pg.get("lr")
            wd = pg.get("weight_decay")
            lines.append("-" * 60)
            lines.append(f"  Group {idx + 1}")
            if isinstance(lr, float):
                lines.append(f"    LR             : {lr:.3e}")
            else:
                lines.append(f"    LR             : {lr}")
            if isinstance(wd, float):
                lines.append(f"    Weight Decay   : {wd:.3e}")
            else:
                lines.append(f"    Weight Decay   : {wd}")
            lines.append(f"    Params         : {param_count:,}")

    lines.append("=" * 60)
    return "\n".join(lines)


# =============================================================================
# Serialization (metadata-driven, single source of truth)
# =============================================================================

def optimizer_to_dict(
    optimizer: optim.Optimizer,
    model: Optional[nn.Module] = None,
    config: Optional[TrainConfig] = None,
) -> Dict[str, Any]:
    """Lightweight dict representation of optimizer configuration.

    If the optimizer was built by build_optimizer(), uses the attached
    immutable OptimizerMetadata for semantic group names and provenance.
    Falls back to torch param_groups if metadata is absent.

    Suitable for checkpoint metadata and reproducibility records.
    Contains no raw tensors, model objects, or open handles.
    Does NOT include optimizer.state_dict() -- that belongs to
    checkpoint_manager.

    Args:
        optimizer: Constructed torch.optim.Optimizer.
        model:     Optional model for param counts
                   (used only if metadata is absent).
        config:    Optional config for policy context.

    Returns:
        Serializable dict with stable schema.
    """
    meta = get_optimizer_metadata(optimizer)

    if meta:
        # -- Metadata-driven (preferred) ---------------------------------------
        groups = []
        for gm in meta.groups:
            group_dict = {
                "index": gm["index"],
                "name": gm["name"],
                "lr": gm["lr"],
                "weight_decay": gm["weight_decay"],
                "param_count": gm["param_count"],
            }
            # Enrich with optimizer-specific fields from torch param_groups
            if gm["index"] < len(optimizer.param_groups):
                pg = optimizer.param_groups[gm["index"]]
                for key in ("betas", "eps", "momentum", "nesterov"):
                    if key in pg:
                        group_dict[key] = pg[key]
            groups.append(group_dict)

        result: Dict[str, Any] = {
            "optimizer_type": meta.optimizer_type,
            "num_groups": len(groups),
            "total_optimizer_params": meta.total_optimizer_params,
            "used_model_api": meta.used_model_api,
            "trainable_params": meta.trainable_params,
            "frozen_params": meta.frozen_params,
            "groups": groups,
        }
    else:
        # -- Fallback (optimizer not built by build_optimizer) -----------------
        opt_class = type(optimizer).__name__
        groups = []
        total_opt_params = 0
        for idx, pg in enumerate(optimizer.param_groups):
            param_count = sum(p.numel() for p in pg["params"])
            total_opt_params += param_count
            group_meta = {
                "index": idx,
                "lr": pg.get("lr"),
                "weight_decay": pg.get("weight_decay"),
                "param_count": param_count,
            }
            for key in ("betas", "eps", "momentum", "nesterov"):
                if key in pg:
                    group_meta[key] = pg[key]
            groups.append(group_meta)

        result = {
            "optimizer_type": opt_class,
            "num_groups": len(groups),
            "total_optimizer_params": total_opt_params,
            "groups": groups,
        }

        if model is not None:
            t, f = _count_params(model)
            result["trainable_params"] = t
            result["frozen_params"] = f

    if config is not None:
        result["config_optimizer"] = config.optimizer
        result["config_learning_rate"] = config.learning_rate
        result["config_weight_decay"] = config.weight_decay
        result["config_backbone_lr_multiplier"] = config.backbone_lr_multiplier

    return result


def as_dict(
    optimizer: optim.Optimizer,
    model: Optional[nn.Module] = None,
    config: Optional[TrainConfig] = None,
) -> Dict[str, Any]:
    """Alias for optimizer_to_dict(). Ergonomic compatibility."""
    return optimizer_to_dict(optimizer, model, config)


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

    from training.train_config import build_train_config
    from training.run_context import build_run_context

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

    # -- Helper: build frozen config + context ---------------------------------
    def _build_ctx(**kwargs):
        cfg = build_train_config(**kwargs)
        cfg.freeze()
        ctx = build_run_context(cfg)
        return cfg, ctx

    # -- Helper: simple trainable model ----------------------------------------
    class DummyModel(nn.Module):
        def __init__(self, in_dim=8, hidden=32, out_dim=1):
            super().__init__()
            self.layer1 = nn.Linear(in_dim, hidden)
            self.layer2 = nn.Linear(hidden, out_dim)

        def forward(self, x):
            return self.layer2(torch.relu(self.layer1(x)))

    # -- Helper: model with custom parameter groups ----------------------------
    class GroupedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(16, 32)
            self.head = nn.Linear(32, 1)

        def forward(self, x):
            return self.head(torch.relu(self.backbone(x)))

        def get_optimizer_parameter_groups(self):
            return [
                {
                    "name": "Backbone",
                    "params": self.backbone.parameters(),
                    "lr_scale": 0.1,
                    "is_backbone": True,
                },
                {
                    "name": "Head",
                    "params": self.head.parameters(),
                },
            ]

    # -- Helper: model with all frozen params ----------------------------------
    class FrozenModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)
            for p in self.parameters():
                p.requires_grad = False

        def forward(self, x):
            return self.fc(x)

    # -- Helper: empty model (no params) ---------------------------------------
    class EmptyModel(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x

    # -- Helper: model with duplicate params in groups -------------------------
    class DuplicateGroupModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, x):
            return self.fc(x)

        def get_optimizer_parameter_groups(self):
            params = list(self.fc.parameters())
            return [
                {"name": "Group A", "params": params},
                {"name": "Group B", "params": params},  # duplicate!
            ]

    # -- Helper: model with backbone multiplier group --------------------------
    class BackboneMultModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(8, 16)
            self.head = nn.Linear(16, 1)

        def forward(self, x):
            return self.head(torch.relu(self.backbone(x)))

        def get_optimizer_parameter_groups(self):
            return [
                {
                    "name": "Backbone",
                    "params": self.backbone.parameters(),
                    "is_backbone": True,
                },
                {
                    "name": "Head",
                    "params": self.head.parameters(),
                },
            ]

    print("=" * 60)
    print("  training/optimizer.py -- smoke test")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. AdamW construction
    # -------------------------------------------------------------------------
    print("\n  1. AdamW construction...")
    cfg, ctx = _build_ctx(optimizer="adamw", device="cpu")
    model = DummyModel()
    opt = build_optimizer(cfg, ctx, model)
    check("AdamW builds", isinstance(opt, optim.AdamW))
    check("AdamW has param groups", len(opt.param_groups) > 0)

    # -------------------------------------------------------------------------
    # 2. Adam construction
    # -------------------------------------------------------------------------
    print("\n  2. Adam construction...")
    cfg2, ctx2 = _build_ctx(optimizer="adam", device="cpu")
    opt2 = build_optimizer(cfg2, ctx2, DummyModel())
    check("Adam builds", isinstance(opt2, optim.Adam) and not isinstance(opt2, optim.AdamW))

    # -------------------------------------------------------------------------
    # 3. SGD construction
    # -------------------------------------------------------------------------
    print("\n  3. SGD construction...")
    cfg3, ctx3 = _build_ctx(optimizer="sgd", device="cpu")
    opt3 = build_optimizer(cfg3, ctx3, DummyModel())
    check("SGD builds", isinstance(opt3, optim.SGD))

    # -------------------------------------------------------------------------
    # 4. Optimizer name normalization
    # -------------------------------------------------------------------------
    print("\n  4. Name normalization...")
    for name_variant in ["AdamW", " adamw ", "ADAMW", "adamw"]:
        cfg_n, ctx_n = _build_ctx(optimizer=name_variant, device="cpu")
        opt_n = build_optimizer(cfg_n, ctx_n, DummyModel())
        check(f"'{name_variant}' -> AdamW", isinstance(opt_n, optim.AdamW))

    # -------------------------------------------------------------------------
    # 5. Fallback single-group path
    # -------------------------------------------------------------------------
    print("\n  5. Fallback single-group path...")
    cfg5, ctx5 = _build_ctx(device="cpu")
    model5 = DummyModel()
    opt5 = build_optimizer(cfg5, ctx5, model5)
    check("fallback has 1 group", len(opt5.param_groups) == 1)
    check("fallback LR matches config", opt5.param_groups[0]["lr"] == cfg5.learning_rate)
    check("fallback WD matches config", opt5.param_groups[0]["weight_decay"] == cfg5.weight_decay)

    # -------------------------------------------------------------------------
    # 6. Custom get_optimizer_parameter_groups() path
    # -------------------------------------------------------------------------
    print("\n  6. Custom parameter groups...")
    cfg6, ctx6 = _build_ctx(device="cpu", learning_rate=1e-3)
    gm = GroupedModel()
    opt6 = build_optimizer(cfg6, ctx6, gm)
    check("grouped has 2 groups", len(opt6.param_groups) == 2)
    check("backbone LR scaled", abs(opt6.param_groups[0]["lr"] - 1e-4) < 1e-10)
    check("head LR is base", abs(opt6.param_groups[1]["lr"] - 1e-3) < 1e-10)

    # -------------------------------------------------------------------------
    # 7. Duplicate parameter detection
    # -------------------------------------------------------------------------
    print("\n  7. Duplicate parameter detection...")
    cfg7, ctx7 = _build_ctx(device="cpu")
    expect_error("duplicate params rejected", OptimizerError,
                 lambda: build_optimizer(cfg7, ctx7, DuplicateGroupModel()))

    # -------------------------------------------------------------------------
    # 8. All parameters frozen
    # -------------------------------------------------------------------------
    print("\n  8. All parameters frozen...")
    cfg8, ctx8 = _build_ctx(device="cpu")
    expect_error("frozen model rejected", OptimizerError,
                 lambda: build_optimizer(cfg8, ctx8, FrozenModel()))

    # -------------------------------------------------------------------------
    # 9. Empty model (no parameters)
    # -------------------------------------------------------------------------
    print("\n  9. Empty model...")
    cfg9, ctx9 = _build_ctx(device="cpu")
    expect_error("empty model rejected", OptimizerError,
                 lambda: build_optimizer(cfg9, ctx9, EmptyModel()))

    # -------------------------------------------------------------------------
    # 10. Invalid optimizer name
    # -------------------------------------------------------------------------
    print("\n  10. Invalid optimizer name...")
    # TrainConfig rejects invalid optimizer names during validation,
    # so test defense-in-depth via direct construction check
    expect_error("bad optimizer name via validation", Exception,
                 lambda: build_train_config(optimizer="magic"))

    # -------------------------------------------------------------------------
    # 11. Invalid config type
    # -------------------------------------------------------------------------
    print("\n  11. Invalid config type...")
    _, ctx11 = _build_ctx(device="cpu")
    expect_error("dict config rejected", OptimizerError,
                 lambda: build_optimizer({"optimizer": "adamw"}, ctx11, DummyModel()))
    expect_error("None config rejected", OptimizerError,
                 lambda: build_optimizer(None, ctx11, DummyModel()))

    # -------------------------------------------------------------------------
    # 12. Invalid run context type
    # -------------------------------------------------------------------------
    print("\n  12. Invalid run context type...")
    cfg12, _ = _build_ctx(device="cpu")
    expect_error("dict context rejected", OptimizerError,
                 lambda: build_optimizer(cfg12, {"device": "cpu"}, DummyModel()))
    expect_error("None context rejected", OptimizerError,
                 lambda: build_optimizer(cfg12, None, DummyModel()))

    # -------------------------------------------------------------------------
    # 13. Invalid model type
    # -------------------------------------------------------------------------
    print("\n  13. Invalid model type...")
    cfg13, ctx13 = _build_ctx(device="cpu")
    expect_error("string model rejected", OptimizerError,
                 lambda: build_optimizer(cfg13, ctx13, "not_a_model"))
    expect_error("None model rejected", OptimizerError,
                 lambda: build_optimizer(cfg13, ctx13, None))

    # -------------------------------------------------------------------------
    # 14. Summary helper
    # -------------------------------------------------------------------------
    print("\n  14. Summary helper...")
    cfg14, ctx14 = _build_ctx(device="cpu")
    model14 = DummyModel()
    opt14 = build_optimizer(cfg14, ctx14, model14)
    s = summarize_optimizer(opt14, model=model14, config=cfg14)
    check("summary returns string", isinstance(s, str))
    check("summary non-trivial", len(s) > 100)
    check("summary contains optimizer type", "AdamW" in s)
    check("summary contains LR", "Learning Rate" in s or "LR" in s)

    # -------------------------------------------------------------------------
    # 15. Serialization helper
    # -------------------------------------------------------------------------
    print("\n  15. Serialization helper...")
    d = optimizer_to_dict(opt14, model=model14, config=cfg14)
    check("to_dict returns dict", isinstance(d, dict))
    check("to_dict has optimizer_type", "optimizer_type" in d)
    check("to_dict has groups", "groups" in d and isinstance(d["groups"], list))
    check("to_dict has trainable_params", "trainable_params" in d)
    check("to_dict has config_learning_rate", "config_learning_rate" in d)
    # No raw tensors in serialized output
    check("to_dict groups have no params",
          all("params" not in g for g in d["groups"]))

    # as_dict alias
    ad = as_dict(opt14, model=model14, config=cfg14)
    check("as_dict matches to_dict", ad == d)

    # -------------------------------------------------------------------------
    # 16. Backbone multiplier via group metadata
    # -------------------------------------------------------------------------
    print("\n  16. Backbone multiplier...")
    cfg16, ctx16 = _build_ctx(
        device="cpu", learning_rate=1e-3, backbone_lr_multiplier=0.01,
    )
    bm = BackboneMultModel()
    opt16 = build_optimizer(cfg16, ctx16, bm)
    check("backbone group has reduced LR",
          abs(opt16.param_groups[0]["lr"] - 1e-5) < 1e-12)
    check("head group has full LR",
          abs(opt16.param_groups[1]["lr"] - 1e-3) < 1e-12)

    # lr_scale takes precedence over backbone_lr_multiplier
    cfg16b, ctx16b = _build_ctx(
        device="cpu", learning_rate=1e-3, backbone_lr_multiplier=0.01,
    )
    gm16 = GroupedModel()  # has lr_scale=0.1 on backbone
    opt16b = build_optimizer(cfg16b, ctx16b, gm16)
    check("lr_scale overrides backbone_multiplier",
          abs(opt16b.param_groups[0]["lr"] - 1e-4) < 1e-12)

    # -------------------------------------------------------------------------
    # 17. Unfrozen config rejected
    # -------------------------------------------------------------------------
    print("\n  17. Unfrozen config rejected...")
    cfg_unfrozen = build_train_config(device="cpu")
    # cfg_unfrozen is VALIDATED but NOT frozen
    _, ctx17 = _build_ctx(device="cpu")
    expect_error("unfrozen config rejected", OptimizerError,
                 lambda: build_optimizer(cfg_unfrozen, ctx17, DummyModel()))

    # -------------------------------------------------------------------------
    # 18. Deterministic group ordering
    # -------------------------------------------------------------------------
    print("\n  18. Deterministic group ordering...")
    cfg18, ctx18 = _build_ctx(device="cpu", learning_rate=1e-3)
    gm18 = GroupedModel()
    opt18a = build_optimizer(cfg18, ctx18, gm18)
    opt18b = build_optimizer(cfg18, ctx18, gm18)
    check("group count stable", len(opt18a.param_groups) == len(opt18b.param_groups))
    check("group LRs stable",
          [g["lr"] for g in opt18a.param_groups] == [g["lr"] for g in opt18b.param_groups])

    # -------------------------------------------------------------------------
    # 19. Optimizer defaults correctness
    # -------------------------------------------------------------------------
    print("\n  19. Optimizer defaults...")
    # AdamW betas
    cfg19, ctx19 = _build_ctx(optimizer="adamw", device="cpu")
    opt19 = build_optimizer(cfg19, ctx19, DummyModel())
    check("AdamW betas=(0.9, 0.999)",
          opt19.param_groups[0].get("betas") == (0.9, 0.999))
    check("AdamW eps=1e-8",
          opt19.param_groups[0].get("eps") == 1e-8)

    # SGD momentum
    cfg19s, ctx19s = _build_ctx(optimizer="sgd", device="cpu")
    opt19s = build_optimizer(cfg19s, ctx19s, DummyModel())
    check("SGD momentum=0.9",
          opt19s.param_groups[0].get("momentum") == 0.9)
    check("SGD nesterov=False",
          opt19s.param_groups[0].get("nesterov") is False)

    # -------------------------------------------------------------------------
    # 20. validate_optimizer_inputs public API
    # -------------------------------------------------------------------------
    print("\n  20. validate_optimizer_inputs()...")
    cfg20, ctx20 = _build_ctx(device="cpu")
    try:
        validate_optimizer_inputs(cfg20, ctx20, DummyModel())
        check("valid inputs pass validation", True)
    except OptimizerError:
        check("valid inputs pass validation", False, "unexpected error")

    # -------------------------------------------------------------------------
    # 21. Config/context mismatch rejection
    # -------------------------------------------------------------------------
    print("\n  21. Config/context mismatch rejection...")
    cfg_a, ctx_a = _build_ctx(device="cpu")
    cfg_b, ctx_b = _build_ctx(device="cpu")
    # cfg_a + ctx_b = mismatch
    expect_error("mismatched config/context rejected", OptimizerError,
                 lambda: build_optimizer(cfg_a, ctx_b, DummyModel()))
    # correct pair still works
    try:
        build_optimizer(cfg_a, ctx_a, DummyModel())
        check("matched config/context accepted", True)
    except OptimizerError:
        check("matched config/context accepted", False, "unexpected error")

    # -------------------------------------------------------------------------
    # 22. Semantic group names in summary
    # -------------------------------------------------------------------------
    print("\n  22. Semantic group names in summary...")
    cfg22, ctx22 = _build_ctx(device="cpu", learning_rate=1e-3)
    gm22 = GroupedModel()
    opt22 = build_optimizer(cfg22, ctx22, gm22)
    s22 = summarize_optimizer(opt22)
    check("summary contains 'Backbone'", "Backbone" in s22)
    check("summary contains 'Head'", "Head" in s22)
    check("summary contains 'model API'", "model API" in s22)

    # -------------------------------------------------------------------------
    # 23. Semantic group names in optimizer_to_dict
    # -------------------------------------------------------------------------
    print("\n  23. Semantic group names in to_dict...")
    d23 = optimizer_to_dict(opt22)
    check("to_dict has used_model_api", d23.get("used_model_api") is True)
    check("to_dict group 0 name is Backbone",
          d23["groups"][0].get("name") == "Backbone")
    check("to_dict group 1 name is Head",
          d23["groups"][1].get("name") == "Head")

    # -------------------------------------------------------------------------
    # 24. Fallback path metadata
    # -------------------------------------------------------------------------
    print("\n  24. Fallback path metadata...")
    cfg24, ctx24 = _build_ctx(device="cpu")
    opt24 = build_optimizer(cfg24, ctx24, DummyModel())
    d24 = optimizer_to_dict(opt24)
    check("fallback used_model_api is False", d24.get("used_model_api") is False)
    check("fallback group name is all_trainable",
          d24["groups"][0].get("name") == "all_trainable")
    s24 = summarize_optimizer(opt24)
    check("fallback summary says fallback", "fallback" in s24)

    # -------------------------------------------------------------------------
    # 25. Metadata survives optimizer construction (immutable)
    # -------------------------------------------------------------------------
    print("\n  25. Metadata immutability...")
    meta25 = get_optimizer_metadata(opt22)
    check("metadata attached", meta25 is not None)
    check("metadata optimizer_type", meta25.optimizer_type == "AdamW")
    check("metadata groups count", len(meta25.groups) == 2)
    check("metadata used_model_api", meta25.used_model_api is True)
    check("metadata trainable_params > 0", meta25.trainable_params > 0)

    # immutability
    try:
        meta25.optimizer_type = "SGD"
        check("metadata setattr blocked", False, "should have raised")
    except AttributeError:
        check("metadata setattr blocked", True)
    try:
        del meta25._optimizer_type
        check("metadata delattr blocked", False, "should have raised")
    except AttributeError:
        check("metadata delattr blocked", True)

    # group proxy is read-only
    try:
        meta25.groups[0]["name"] = "hacked"
        check("group proxy is read-only", False, "should have raised")
    except TypeError:
        check("group proxy is read-only", True)

    # -------------------------------------------------------------------------
    # 26. Deterministic ordering in metadata
    # -------------------------------------------------------------------------
    print("\n  26. Deterministic ordering in metadata...")
    cfg26, ctx26 = _build_ctx(device="cpu", learning_rate=1e-3)
    gm26 = GroupedModel()
    opt26a = build_optimizer(cfg26, ctx26, gm26)
    opt26b = build_optimizer(cfg26, ctx26, gm26)
    ma = get_optimizer_metadata(opt26a)
    mb = get_optimizer_metadata(opt26b)
    check("metadata group names stable",
          [g["name"] for g in ma.groups] == [g["name"] for g in mb.groups])
    check("metadata group LRs stable",
          [g["lr"] for g in ma.groups] == [g["lr"] for g in mb.groups])
    check("metadata group order stable",
          [g["index"] for g in ma.groups] == [g["index"] for g in mb.groups])

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
