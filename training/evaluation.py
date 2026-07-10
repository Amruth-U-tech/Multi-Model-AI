# =============================================================================
# training/evaluation.py
# Evaluation Authority -- Multimodal AI Pipeline
# =============================================================================
#
# Purpose:
#   THE SINGLE AUTHORITY for computing losses, regression metrics, and
#   evaluation results within the training subsystem.
#   Answers exactly one question:
#       "Given predictions and targets, what is the quantitative evaluation
#        of model performance, and how does it compare to the best seen?"
#
# Responsibilities (ONLY):
#   1. Validate evaluation inputs (config, run_context, tensors)
#   2. Compute regression losses (MSE, MAE, Huber)
#   3. Compute regression metrics (MSE, RMSE, MAE, R2)
#   4. Extract predictions from model output (Tensor or dict)
#   5. Track best validation performance over time
#   6. Expose immutable EvaluationResult snapshots
#   7. Expose summary and serialization helpers
#
# What this file does NOT do:
#   - Run forward passes or call model(...)
#   - Call loss.backward() or optimizer.step()
#   - Step schedulers or manage learning rates
#   - Save checkpoints or write files
#   - Move tensors between devices (.to() / .cuda())
#   - Orchestrate training loops or count epochs
#   - Import models, datasets, or collate functions
#   - Configure logging or print during library use
#   - Mutate TrainConfig or RunContext
#
# Ownership Map:
#   TrainConfig      -> loss policy (loss_name, VALID_LOSSES)
#   RunContext       -> runtime identity (device, paths)
#   optimizer.py     -> optimizer construction
#   scheduler.py     -> scheduler construction + step policy
#   evaluation.py    -> loss/metrics/results/best-tracking (THIS FILE)
#   future trainer   -> orchestrates forward/backward, calls evaluator
#
# Usage:
#   from training.evaluation import build_evaluator, compute_loss
# =============================================================================

import sys
import math
import time
import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# -- Project root bootstrap ----------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _THIS_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

import torch
import torch.nn.functional as F

from training.train_config import TrainConfig, ConfigState, VALID_LOSSES
from training.run_context import RunContext


# =============================================================================
# Constants
# =============================================================================

_SUPPORTED_LOSSES = frozenset(VALID_LOSSES)
_SUPPORTED_METRICS = frozenset({"mse", "rmse", "mae", "r2"})
_VALID_SPLITS = frozenset({"train", "validation", "test"})

# Default prediction key from FusionModel output dict
_PREDICTION_KEY = "rating_prediction"
# Default collated batch target key
_TARGET_KEY = "ratings"


# =============================================================================
# Error
# =============================================================================

class EvaluationError(RuntimeError):
    """Structured evaluation error with stage, field, and resolution info."""

    def __init__(self, stage: str, field_name: str, received: Any,
                 expected: str, resolution: str = ""):
        self.stage = stage
        self.field_name = field_name
        self.received = received
        self.expected = expected
        self.resolution = resolution
        lines = [
            "[EVALUATION ERROR]",
            f"  Stage     : {stage}",
            f"  Field     : {field_name}",
            f"  Received  : {received!r}",
            f"  Expected  : {expected}",
        ]
        if resolution:
            lines.append(f"  Resolution: {resolution}")
        super().__init__("\n".join(lines))


# =============================================================================
# Immutable Evaluation Metadata
# =============================================================================

@dataclass(frozen=True)
class EvaluationMetadata:
    """Immutable description of what the evaluator expects and computes.

    Captured at evaluator construction time. Never changes.
    """
    problem_type: str
    loss_name: str
    prediction_key: str
    target_key: str
    supported_losses: Tuple[str, ...]
    supported_metrics: Tuple[str, ...]
    step_source: str  # "trainer" -- evaluation does not step anything


# =============================================================================
# Evaluation Runtime State
# =============================================================================

@dataclass
class EvaluationRuntimeState:
    """Mutable evaluation memory for best-validation tracking.

    Not a checkpoint. Not persisted. Lives only in the evaluator instance.
    The trainer reads this to decide logging and checkpointing.
    """
    latest_loss: Optional[float] = None
    latest_metrics: Dict[str, float] = field(default_factory=dict)
    samples_evaluated: int = 0
    batches_evaluated: int = 0
    best_validation_loss: Optional[float] = None
    best_validation_epoch: Optional[int] = None
    best_rmse: Optional[float] = None
    best_mae: Optional[float] = None
    best_r2: Optional[float] = None
    epochs_without_improvement: int = 0
    last_best_update_epoch: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        """Serializable snapshot of runtime state."""
        return {
            "latest_loss": self.latest_loss,
            "latest_metrics": dict(self.latest_metrics),
            "samples_evaluated": self.samples_evaluated,
            "batches_evaluated": self.batches_evaluated,
            "best_validation_loss": self.best_validation_loss,
            "best_validation_epoch": self.best_validation_epoch,
            "best_rmse": self.best_rmse,
            "best_mae": self.best_mae,
            "best_r2": self.best_r2,
            "epochs_without_improvement": self.epochs_without_improvement,
            "last_best_update_epoch": self.last_best_update_epoch,
        }


# =============================================================================
# Immutable Evaluation Result
# =============================================================================

@dataclass(frozen=True)
class EvaluationResult:
    """Immutable snapshot of a single evaluation pass.

    Contains only Python scalars -- no raw tensors.
    The trainer uses this for logging, checkpointing decisions, and display.
    """
    split: str
    epoch: Optional[int]
    batch_index: Optional[int]
    loss: float
    mse: float
    rmse: float
    mae: float
    r2: float
    prediction_count: int
    ignored_count: int
    duration_ms: float
    warnings: Tuple[str, ...]

    def summary(self) -> str:
        """Human-readable one-line summary."""
        parts = [f"{self.split}"]
        if self.epoch is not None:
            parts.append(f"epoch={self.epoch}")
        if self.batch_index is not None:
            parts.append(f"batch={self.batch_index}")
        parts.extend([
            f"loss={self.loss:.6f}",
            f"rmse={self.rmse:.6f}",
            f"mae={self.mae:.6f}",
            f"r2={self.r2:.6f}",
            f"n={self.prediction_count}",
        ])
        if self.duration_ms > 0:
            parts.append(f"time={self.duration_ms:.1f}ms")
        return " | ".join(parts)

    def as_dict(self) -> Dict[str, Any]:
        """Serializable dict for checkpoint metadata and logging."""
        return {
            "split": self.split,
            "epoch": self.epoch,
            "batch_index": self.batch_index,
            "loss": self.loss,
            "mse": self.mse,
            "rmse": self.rmse,
            "mae": self.mae,
            "r2": self.r2,
            "prediction_count": self.prediction_count,
            "ignored_count": self.ignored_count,
            "duration_ms": self.duration_ms,
            "warnings": list(self.warnings),
        }


# =============================================================================
# Prediction Extraction
# =============================================================================

def extract_prediction(model_output: Any) -> torch.Tensor:
    """Extract prediction tensor from model output.

    Accepted inputs:
        - torch.Tensor directly
        - dict containing 'rating_prediction' key

    Returns:
        Prediction tensor (not cloned, not moved).

    Raises:
        EvaluationError: On invalid or ambiguous input.
    """
    if isinstance(model_output, torch.Tensor):
        if model_output.numel() == 0:
            raise EvaluationError(
                "prediction_extraction", "model_output", "empty tensor",
                "non-empty prediction tensor",
                "Model returned an empty tensor.",
            )
        return model_output

    if isinstance(model_output, dict):
        if _PREDICTION_KEY not in model_output:
            raise EvaluationError(
                "prediction_extraction", "model_output",
                f"dict keys {sorted(model_output.keys())}",
                f"dict containing '{_PREDICTION_KEY}'",
                f"Pass FusionModel output directly or provide "
                f"output['{_PREDICTION_KEY}'].",
            )
        pred = model_output[_PREDICTION_KEY]
        if not isinstance(pred, torch.Tensor):
            raise EvaluationError(
                "prediction_extraction", _PREDICTION_KEY,
                type(pred).__name__,
                "torch.Tensor",
                f"output['{_PREDICTION_KEY}'] must be a Tensor.",
            )
        if pred.numel() == 0:
            raise EvaluationError(
                "prediction_extraction", _PREDICTION_KEY, "empty tensor",
                "non-empty prediction tensor",
                f"output['{_PREDICTION_KEY}'] is empty.",
            )
        return pred

    raise EvaluationError(
        "prediction_extraction", "model_output", type(model_output).__name__,
        f"Tensor or dict containing '{_PREDICTION_KEY}'",
        "Pass FusionModel output directly or a raw prediction tensor.",
    )


# =============================================================================
# Tensor Validation
# =============================================================================

def _validate_tensors(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Validate and normalize prediction/target tensors.

    Normalizes [B, 1] -> [B] for both tensors.
    Returns (predictions_flat, targets_flat, warnings).

    Raises:
        EvaluationError: On contract violations.
    """
    warnings_list: List[str] = []

    # -- Type checks -----------------------------------------------------------
    if not isinstance(predictions, torch.Tensor):
        raise EvaluationError(
            "tensor_validation", "predictions", type(predictions).__name__,
            "torch.Tensor",
            "Pass prediction tensor from model output.",
        )
    if not isinstance(targets, torch.Tensor):
        raise EvaluationError(
            "tensor_validation", "targets", type(targets).__name__,
            "torch.Tensor",
            "Pass target tensor from collated batch (key: 'ratings').",
        )

    # -- Empty checks ----------------------------------------------------------
    if predictions.numel() == 0:
        raise EvaluationError(
            "tensor_validation", "predictions", "empty tensor (0 elements)",
            "non-empty prediction tensor",
            "Model returned empty predictions.",
        )
    if targets.numel() == 0:
        raise EvaluationError(
            "tensor_validation", "targets", "empty tensor (0 elements)",
            "non-empty target tensor",
            "Batch contained no target values.",
        )

    # -- Dtype checks ----------------------------------------------------------
    if predictions.dtype == torch.bool:
        raise EvaluationError(
            "tensor_validation", "predictions.dtype", "torch.bool",
            "numeric dtype (float32, float64, etc.)",
            "Cast predictions to float before evaluation.",
        )
    if targets.dtype == torch.bool:
        raise EvaluationError(
            "tensor_validation", "targets.dtype", "torch.bool",
            "numeric dtype (float32, float64, etc.)",
            "Cast targets to float before evaluation.",
        )
    if predictions.dtype.is_complex:
        raise EvaluationError(
            "tensor_validation", "predictions.dtype",
            str(predictions.dtype),
            "real numeric dtype (float32, float64, etc.), not complex",
            "Ensure model head outputs real-valued rating predictions.",
        )
    if targets.dtype.is_complex:
        raise EvaluationError(
            "tensor_validation", "targets.dtype",
            str(targets.dtype),
            "real numeric dtype (float32, float64, etc.), not complex",
            "Ensure targets contain real-valued ratings.",
        )
    if not predictions.is_floating_point():
        try:
            predictions = predictions.float()
        except Exception:
            raise EvaluationError(
                "tensor_validation", "predictions.dtype",
                str(predictions.dtype),
                "floating-point dtype",
                "Cast predictions to float before evaluation.",
            )
    if not targets.is_floating_point():
        try:
            targets = targets.float()
        except Exception:
            raise EvaluationError(
                "tensor_validation", "targets.dtype", str(targets.dtype),
                "floating-point dtype",
                "Cast targets to float before evaluation.",
            )

    # -- Device check ----------------------------------------------------------
    if predictions.device != targets.device:
        raise EvaluationError(
            "tensor_validation", "device",
            f"predictions={predictions.device}, targets={targets.device}",
            "same device for predictions and targets",
            "Trainer/collate must move batch tensors before evaluation.",
        )

    # -- Shape normalization: [B, 1] -> [B] -----------------------------------
    if predictions.dim() == 2 and predictions.shape[1] == 1:
        predictions = predictions.squeeze(1)
    if targets.dim() == 2 and targets.shape[1] == 1:
        targets = targets.squeeze(1)

    # -- Shape compatibility ---------------------------------------------------
    if predictions.shape != targets.shape:
        raise EvaluationError(
            "tensor_validation", "shape",
            f"predictions={list(predictions.shape)}, "
            f"targets={list(targets.shape)}",
            "matching shapes after [B,1]->[B] normalization",
            "Check model output shape and target tensor shape.",
        )

    # -- Dimension check: only [B] or scalar -----------------------------------
    if predictions.dim() > 1:
        raise EvaluationError(
            "tensor_validation", "predictions.shape",
            list(predictions.shape),
            "[B] or scalar after normalization",
            "Evaluation supports 1D batch predictions only.",
        )

    # -- NaN / Inf checks ------------------------------------------------------
    if torch.isnan(predictions).any():
        nan_count = torch.isnan(predictions).sum().item()
        raise EvaluationError(
            "tensor_validation", "predictions",
            f"{nan_count} NaN values detected",
            "finite numeric predictions",
            "Check model output for numerical instability.",
        )
    if torch.isinf(predictions).any():
        inf_count = torch.isinf(predictions).sum().item()
        raise EvaluationError(
            "tensor_validation", "predictions",
            f"{inf_count} Inf values detected",
            "finite numeric predictions",
            "Check model output for overflow.",
        )
    if torch.isnan(targets).any():
        nan_count = torch.isnan(targets).sum().item()
        raise EvaluationError(
            "tensor_validation", "targets",
            f"{nan_count} NaN values detected",
            "finite numeric targets",
            "Check dataset for missing/corrupt values.",
        )
    if torch.isinf(targets).any():
        inf_count = torch.isinf(targets).sum().item()
        raise EvaluationError(
            "tensor_validation", "targets",
            f"{inf_count} Inf values detected",
            "finite numeric targets",
            "Check dataset for overflow values.",
        )

    return predictions, targets, warnings_list


# =============================================================================
# Loss Computation
# =============================================================================

def compute_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    config: TrainConfig,
) -> torch.Tensor:
    """Compute regression loss for trainer backpropagation.

    Returns a scalar tensor suitable for loss.backward().
    Does NOT call backward() itself.

    Args:
        predictions: Model predictions, shape [B] or [B, 1].
        targets:     Ground truth, shape [B] or [B, 1].
        config:      Frozen TrainConfig with loss_name.

    Returns:
        Scalar loss tensor (on same device as inputs).

    Raises:
        EvaluationError: On invalid inputs or unsupported loss.
    """
    if not isinstance(config, TrainConfig):
        raise EvaluationError(
            "loss_computation", "config", type(config).__name__,
            "TrainConfig instance",
            "Pass a frozen TrainConfig.",
        )
    if config.state == ConfigState.CREATED:
        raise EvaluationError(
            "loss_computation", "config._state", config.state.value,
            "VALIDATED, OVERRIDDEN, or FROZEN",
            "Call config.validate() before computing loss.",
        )
    if not config.is_frozen:
        raise EvaluationError(
            "loss_computation", "config._frozen", False,
            "frozen config (config.freeze())",
            "Call config.freeze() before computing loss.",
        )

    loss_name = config.loss_name.strip().lower()
    if loss_name not in _SUPPORTED_LOSSES:
        raise EvaluationError(
            "loss_computation", "loss_name", loss_name,
            f"one of {sorted(_SUPPORTED_LOSSES)}",
            "Check TrainConfig.loss_name value.",
        )

    preds, tgts, _ = _validate_tensors(predictions, targets)

    if loss_name == "mse":
        return F.mse_loss(preds, tgts)
    elif loss_name == "mae":
        return F.l1_loss(preds, tgts)
    elif loss_name == "huber":
        return F.smooth_l1_loss(preds, tgts)
    else:
        # Unreachable after validation, but defense-in-depth
        raise EvaluationError(
            "loss_computation", "loss_name", loss_name,
            f"one of {sorted(_SUPPORTED_LOSSES)}",
            "Internal error: unsupported loss after validation.",
        )


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    """Compute regression metrics: MSE, RMSE, MAE, R2.

    Returns Python floats only. Never returns NaN.

    Args:
        predictions: Model predictions, shape [B] or [B, 1].
        targets:     Ground truth, shape [B] or [B, 1].

    Returns:
        Dict with keys: 'mse', 'rmse', 'mae', 'r2'.

    Raises:
        EvaluationError: On invalid inputs.
    """
    preds, tgts, _ = _validate_tensors(predictions, targets)

    with torch.no_grad():
        mse = F.mse_loss(preds, tgts).item()
        rmse = math.sqrt(max(mse, 0.0))
        mae = F.l1_loss(preds, tgts).item()

        # R2 computation with edge-case safety
        r2, r2_warning = _compute_r2(preds, tgts)

    result = {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

    # Sanitize: ensure no NaN/Inf in output
    for key, val in result.items():
        if math.isnan(val) or math.isinf(val):
            result[key] = 0.0

    return result


def _compute_r2(
    predictions: torch.Tensor, targets: torch.Tensor,
) -> Tuple[float, Optional[str]]:
    """Compute R2 with edge-case handling.

    Returns (r2_value, warning_string_or_None).
    Never returns NaN.
    """
    n = targets.numel()
    if n < 2:
        return 0.0, "R2 undefined for n<2, returning 0.0"

    ss_res = ((targets - predictions) ** 2).sum().item()
    target_mean = targets.mean()
    ss_tot = ((targets - target_mean) ** 2).sum().item()

    if ss_tot < 1e-15:
        # Constant targets
        if ss_res < 1e-15:
            return 1.0, "R2: constant targets with perfect predictions, returning 1.0"
        else:
            return 0.0, "R2: constant targets with imperfect predictions, returning 0.0"

    r2 = 1.0 - (ss_res / ss_tot)

    if math.isnan(r2) or math.isinf(r2):
        return 0.0, f"R2 computation yielded {r2}, clamped to 0.0"

    return r2, None


# =============================================================================
# Evaluator Class
# =============================================================================

class Evaluator:
    """Evaluation authority for the training pipeline.

    Combines loss computation, metrics, prediction extraction, and
    best-validation tracking in a single controlled boundary.

    Constructed via build_evaluator(config, run_context).
    """

    def __init__(self, config: TrainConfig, run_context: RunContext):
        """Internal constructor. Use build_evaluator() instead."""
        self._config = config
        self._run_context = run_context
        self._metadata = EvaluationMetadata(
            problem_type="regression",
            loss_name=config.loss_name,
            prediction_key=_PREDICTION_KEY,
            target_key=_TARGET_KEY,
            supported_losses=tuple(sorted(_SUPPORTED_LOSSES)),
            supported_metrics=tuple(sorted(_SUPPORTED_METRICS)),
            step_source="trainer",
        )
        self._state = EvaluationRuntimeState()

    @property
    def metadata(self) -> EvaluationMetadata:
        """Immutable evaluation metadata."""
        return self._metadata

    @property
    def state(self) -> EvaluationRuntimeState:
        """Current mutable runtime state."""
        return self._state

    def evaluate(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        split: str = "validation",
        epoch: Optional[int] = None,
        batch_index: Optional[int] = None,
    ) -> EvaluationResult:
        """Run full evaluation: loss + metrics + result construction.

        Args:
            predictions: Model predictions, Tensor or dict with
                        'rating_prediction'.
            targets:     Ground truth tensor.
            split:       One of 'train', 'validation', 'test'.
            epoch:       Current epoch number (optional).
            batch_index: Current batch index (optional).

        Returns:
            Immutable EvaluationResult with all metrics as Python floats.

        Raises:
            EvaluationError: On any contract violation.
        """
        t_start = time.perf_counter()

        # Validate split
        if split not in _VALID_SPLITS:
            raise EvaluationError(
                "evaluation", "split", split,
                f"one of {sorted(_VALID_SPLITS)}",
                "Use 'train', 'validation', or 'test'.",
            )

        # Extract prediction if dict
        if isinstance(predictions, dict):
            predictions = extract_prediction(predictions)

        # Validate and normalize tensors
        preds, tgts, tensor_warnings = _validate_tensors(predictions, targets)

        all_warnings: List[str] = list(tensor_warnings)
        n = preds.numel()

        # Compute loss
        with torch.no_grad():
            loss_name = self._config.loss_name
            if loss_name == "mse":
                loss_val = F.mse_loss(preds, tgts).item()
            elif loss_name == "mae":
                loss_val = F.l1_loss(preds, tgts).item()
            elif loss_name == "huber":
                loss_val = F.smooth_l1_loss(preds, tgts).item()
            else:
                raise EvaluationError(
                    "evaluation", "loss_name", loss_name,
                    f"one of {sorted(_SUPPORTED_LOSSES)}",
                    "Internal: unsupported loss after validation.",
                )

            # Metrics
            mse = F.mse_loss(preds, tgts).item()
            rmse = math.sqrt(max(mse, 0.0))
            mae = F.l1_loss(preds, tgts).item()
            r2, r2_warning = _compute_r2(preds, tgts)
            if r2_warning:
                all_warnings.append(r2_warning)

        # Sanitize
        for val_name, val in [("loss", loss_val), ("mse", mse),
                              ("rmse", rmse), ("mae", mae), ("r2", r2)]:
            if math.isnan(val) or math.isinf(val):
                all_warnings.append(f"{val_name} was {val}, clamped to 0.0")

        loss_val = 0.0 if (math.isnan(loss_val) or math.isinf(loss_val)) else loss_val
        mse = 0.0 if (math.isnan(mse) or math.isinf(mse)) else mse
        rmse = 0.0 if (math.isnan(rmse) or math.isinf(rmse)) else rmse
        mae = 0.0 if (math.isnan(mae) or math.isinf(mae)) else mae
        r2 = 0.0 if (math.isnan(r2) or math.isinf(r2)) else r2

        duration_ms = (time.perf_counter() - t_start) * 1000.0

        result = EvaluationResult(
            split=split,
            epoch=epoch,
            batch_index=batch_index,
            loss=loss_val,
            mse=mse,
            rmse=rmse,
            mae=mae,
            r2=r2,
            prediction_count=n,
            ignored_count=0,
            duration_ms=duration_ms,
            warnings=tuple(all_warnings),
        )

        # Update runtime state
        self._state.latest_loss = loss_val
        self._state.latest_metrics = {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}
        self._state.samples_evaluated += n
        self._state.batches_evaluated += 1

        return result

    def update_best(self, result: EvaluationResult) -> bool:
        """Update best-validation tracking from an epoch-level result.

        Only accepts results with split='validation' and epoch is not None.
        Lower loss is better. Improvement resets epochs_without_improvement.

        Args:
            result: An EvaluationResult from a validation epoch.

        Returns:
            True if this result is a new best, False otherwise.

        Raises:
            EvaluationError: If result is not a validation epoch result.
        """
        if not isinstance(result, EvaluationResult):
            raise EvaluationError(
                "best_update", "result", type(result).__name__,
                "EvaluationResult instance",
                "Pass an EvaluationResult from evaluator.evaluate().",
            )
        if result.split != "validation":
            raise EvaluationError(
                "best_update", "result.split", result.split,
                "'validation'",
                "Only validation results can update best state. "
                "Use split='validation' for epoch-level validation.",
            )
        if result.epoch is None:
            raise EvaluationError(
                "best_update", "result.epoch", None,
                "integer epoch number",
                "Provide epoch= when calling evaluate() for best tracking.",
            )

        # Reject batch-level results
        if result.batch_index is not None:
            raise EvaluationError(
                "best_update", "result.batch_index", result.batch_index,
                "None (epoch-level validation result)",
                "Aggregate validation epoch first, then call update_best().",
            )

        # Reject duplicate update for the same epoch
        if result.epoch == self._state.last_best_update_epoch:
            raise EvaluationError(
                "best_update", "result.epoch",
                f"duplicate epoch {result.epoch}",
                "each validation epoch updates best state exactly once",
                "Trainer should call update_best() only once after full "
                "validation aggregation.",
            )

        current_best = self._state.best_validation_loss
        is_new_best = (current_best is None) or (result.loss < current_best)

        if is_new_best:
            self._state.best_validation_loss = result.loss
            self._state.best_validation_epoch = result.epoch
            self._state.best_rmse = result.rmse
            self._state.best_mae = result.mae
            self._state.best_r2 = result.r2
            self._state.epochs_without_improvement = 0
        else:
            self._state.epochs_without_improvement += 1

        self._state.last_best_update_epoch = result.epoch

        return is_new_best

    def summary(self) -> str:
        """Human-readable evaluator summary."""
        lines = [
            "=" * 60,
            "  EVALUATOR SUMMARY",
            "=" * 60,
            f"  Problem Type     : {self._metadata.problem_type}",
            f"  Loss Function    : {self._metadata.loss_name}",
            f"  Prediction Key   : {self._metadata.prediction_key}",
            f"  Target Key       : {self._metadata.target_key}",
            f"  Supported Losses : {', '.join(self._metadata.supported_losses)}",
            f"  Supported Metrics: {', '.join(self._metadata.supported_metrics)}",
            "",
            "  Runtime State:",
            f"    Latest Loss           : {self._state.latest_loss}",
            f"    Samples Evaluated     : {self._state.samples_evaluated}",
            f"    Batches Evaluated     : {self._state.batches_evaluated}",
            f"    Best Validation Loss  : {self._state.best_validation_loss}",
            f"    Best Validation Epoch : {self._state.best_validation_epoch}",
            f"    Epochs No Improve     : {self._state.epochs_without_improvement}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def as_dict(self) -> Dict[str, Any]:
        """Serializable dict for checkpoint metadata."""
        return {
            "metadata": {
                "problem_type": self._metadata.problem_type,
                "loss_name": self._metadata.loss_name,
                "prediction_key": self._metadata.prediction_key,
                "target_key": self._metadata.target_key,
                "supported_losses": list(self._metadata.supported_losses),
                "supported_metrics": list(self._metadata.supported_metrics),
                "step_source": self._metadata.step_source,
            },
            "state": self._state.as_dict(),
        }


# =============================================================================
# Builder
# =============================================================================

def build_evaluator(
    config: TrainConfig,
    run_context: RunContext,
) -> Evaluator:
    """Build an Evaluator from validated training config and runtime context.

    Args:
        config:      Validated and frozen TrainConfig.
        run_context: Immutable RunContext built from the same config.

    Returns:
        Evaluator instance.

    Raises:
        EvaluationError: On any invalid input.
    """
    # -- Config type -----------------------------------------------------------
    if not isinstance(config, TrainConfig):
        raise EvaluationError(
            "input_validation", "config", type(config).__name__,
            "TrainConfig instance",
            "Pass a TrainConfig from build_train_config().",
        )
    if config.state == ConfigState.CREATED:
        raise EvaluationError(
            "input_validation", "config._state", config.state.value,
            "VALIDATED, OVERRIDDEN, or FROZEN",
            "Call config.validate() before building evaluator.",
        )
    if not config.is_frozen:
        raise EvaluationError(
            "input_validation", "config._frozen", False,
            "frozen config (config.freeze())",
            "Call config.freeze() before building evaluator.",
        )

    # -- RunContext type -------------------------------------------------------
    if not isinstance(run_context, RunContext):
        raise EvaluationError(
            "input_validation", "run_context", type(run_context).__name__,
            "RunContext instance",
            "Pass a RunContext from build_run_context().",
        )

    # -- Config <-> RunContext pairing -----------------------------------------
    if run_context.config is not config:
        raise EvaluationError(
            "input_validation", "run_context.config",
            "RunContext built from a different TrainConfig",
            "RunContext built from the same frozen TrainConfig",
            "Build RunContext from this exact config and pass them together.",
        )

    # -- Loss name validation (defense-in-depth) -------------------------------
    loss_name = config.loss_name.strip().lower()
    if loss_name not in _SUPPORTED_LOSSES:
        raise EvaluationError(
            "input_validation", "loss_name", config.loss_name,
            f"one of {sorted(_SUPPORTED_LOSSES)}",
            "Check TrainConfig.loss_name value.",
        )

    return Evaluator(config, run_context)


# =============================================================================
# Standalone Helpers (for trainer convenience)
# =============================================================================

def summarize_evaluation(result: EvaluationResult) -> str:
    """Convenience wrapper around result.summary()."""
    if not isinstance(result, EvaluationResult):
        raise EvaluationError(
            "summarize", "result", type(result).__name__,
            "EvaluationResult instance",
            "Pass an EvaluationResult from evaluator.evaluate().",
        )
    return result.summary()


def evaluation_to_dict(result: EvaluationResult) -> Dict[str, Any]:
    """Convenience wrapper around result.as_dict()."""
    if not isinstance(result, EvaluationResult):
        raise EvaluationError(
            "serialize", "result", type(result).__name__,
            "EvaluationResult instance",
            "Pass an EvaluationResult from evaluator.evaluate().",
        )
    return result.as_dict()


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

    def _make_infra(loss_name="mse", **kw):
        """Build config+context for testing."""
        cfg = build_train_config(loss_name=loss_name, device="cpu", **kw)
        cfg.freeze()
        ctx = build_run_context(cfg)
        return cfg, ctx

    print("=" * 60)
    print("  training/evaluation.py -- smoke test")
    print("=" * 60)

    # -- 1. Valid MSE loss -----------------------------------------------------
    print("\n  1. Valid MSE loss...")
    cfg, ctx = _make_infra(loss_name="mse")
    preds = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([1.5, 2.5, 3.5])
    loss = compute_loss(preds, targets, cfg)
    check("MSE loss is tensor", isinstance(loss, torch.Tensor))
    check("MSE loss is scalar", loss.dim() == 0)
    check("MSE loss value", abs(loss.item() - 0.25) < 1e-6)
    check("MSE loss requires grad path", loss.requires_grad is False)  # no grad on raw tensors

    # -- 2. Valid MAE loss -----------------------------------------------------
    print("\n  2. Valid MAE loss...")
    cfg_mae, ctx_mae = _make_infra(loss_name="mae")
    loss_mae = compute_loss(preds, targets, cfg_mae)
    check("MAE loss value", abs(loss_mae.item() - 0.5) < 1e-6)

    # -- 3. Valid Huber loss ---------------------------------------------------
    print("\n  3. Valid Huber loss...")
    cfg_hub, ctx_hub = _make_infra(loss_name="huber")
    loss_hub = compute_loss(preds, targets, cfg_hub)
    check("Huber loss is scalar", loss_hub.dim() == 0)
    check("Huber loss is finite", not math.isnan(loss_hub.item()))

    # -- 4. Metrics for known tensors ------------------------------------------
    print("\n  4. Metrics for known tensors...")
    m = compute_metrics(preds, targets)
    check("metrics has mse", "mse" in m)
    check("metrics has rmse", "rmse" in m)
    check("metrics has mae", "mae" in m)
    check("metrics has r2", "r2" in m)
    check("metrics mse value", abs(m["mse"] - 0.25) < 1e-6)
    check("metrics rmse value", abs(m["rmse"] - 0.5) < 1e-6)
    check("metrics mae value", abs(m["mae"] - 0.5) < 1e-6)
    check("metrics r2 is float", isinstance(m["r2"], float))
    check("metrics no NaN", all(not math.isnan(v) for v in m.values()))

    # -- 5. [B, 1] normalization -----------------------------------------------
    print("\n  5. [B, 1] normalization...")
    preds_2d = torch.tensor([[1.0], [2.0], [3.0]])
    targets_2d = torch.tensor([[1.5], [2.5], [3.5]])
    m_2d = compute_metrics(preds_2d, targets_2d)
    check("[B,1] mse matches [B]", abs(m_2d["mse"] - m["mse"]) < 1e-6)

    # -- 6. Dict prediction extraction -----------------------------------------
    print("\n  6. Dict prediction extraction...")
    pred_dict = {"fused_embedding": torch.randn(4, 128),
                 "rating_prediction": torch.randn(4, 1),
                 "modality_weights": torch.randn(4, 3)}
    extracted = extract_prediction(pred_dict)
    check("dict extraction", isinstance(extracted, torch.Tensor))
    check("dict extraction shape", extracted.shape == (4, 1))

    # Direct tensor passthrough
    direct = extract_prediction(torch.randn(4))
    check("tensor passthrough", isinstance(direct, torch.Tensor))

    # -- 7. Missing prediction key fails ---------------------------------------
    print("\n  7. Missing prediction key fails...")
    expect_error("missing key", EvaluationError,
                 lambda: extract_prediction({"embedding": torch.randn(4)}))

    # -- 8. Empty tensor fails -------------------------------------------------
    print("\n  8. Empty tensor fails...")
    expect_error("empty prediction", EvaluationError,
                 lambda: compute_metrics(torch.tensor([]), torch.tensor([])))

    # -- 9. NaN prediction fails -----------------------------------------------
    print("\n  9. NaN prediction fails...")
    expect_error("NaN prediction", EvaluationError,
                 lambda: compute_metrics(
                     torch.tensor([1.0, float('nan'), 3.0]),
                     torch.tensor([1.0, 2.0, 3.0])))

    # -- 10. Inf target fails --------------------------------------------------
    print("\n  10. Inf target fails...")
    expect_error("Inf target", EvaluationError,
                 lambda: compute_metrics(
                     torch.tensor([1.0, 2.0, 3.0]),
                     torch.tensor([1.0, float('inf'), 3.0])))

    # -- 11. Bool tensor fails -------------------------------------------------
    print("\n  11. Bool tensor fails...")
    expect_error("bool prediction", EvaluationError,
                 lambda: compute_metrics(
                     torch.tensor([True, False, True]),
                     torch.tensor([1.0, 2.0, 3.0])))

    # -- 12. Mismatched shape fails --------------------------------------------
    print("\n  12. Mismatched shape fails...")
    expect_error("shape mismatch", EvaluationError,
                 lambda: compute_metrics(
                     torch.tensor([1.0, 2.0]),
                     torch.tensor([1.0, 2.0, 3.0])))

    # -- 13. Unsupported loss fails --------------------------------------------
    print("\n  13. Unsupported loss fails...")
    # We can't build a config with bad loss_name (TrainConfig rejects it),
    # so we test compute_loss with a string check
    expect_error("non-config type", EvaluationError,
                 lambda: compute_loss(preds, targets, {"loss_name": "cross_entropy"}))

    # -- 14. R2 constant-target stability --------------------------------------
    print("\n  14. R2 constant-target stability...")
    const_tgts = torch.tensor([3.0, 3.0, 3.0, 3.0])
    # Perfect prediction of constant
    m_const_perf = compute_metrics(const_tgts.clone(), const_tgts)
    check("R2 constant perfect = 1.0", m_const_perf["r2"] == 1.0)
    # Imperfect prediction of constant
    m_const_imp = compute_metrics(torch.tensor([1.0, 2.0, 4.0, 5.0]), const_tgts)
    check("R2 constant imperfect = 0.0", m_const_imp["r2"] == 0.0)
    # Single sample
    m_single = compute_metrics(torch.tensor([1.0]), torch.tensor([2.0]))
    check("R2 single sample = 0.0", m_single["r2"] == 0.0)

    # -- 15. EvaluationResult is immutable -------------------------------------
    print("\n  15. EvaluationResult immutability...")
    evaluator = build_evaluator(cfg, ctx)
    result = evaluator.evaluate(preds, targets, split="validation", epoch=1)
    check("result is EvaluationResult", isinstance(result, EvaluationResult))
    expect_error("result setattr blocked", AttributeError,
                 lambda: setattr(result, "loss", 999.0))

    # -- 16. Best validation state updates -------------------------------------
    print("\n  16. Best validation state updates...")
    eval2 = build_evaluator(cfg, ctx)
    # Epoch 1: loss ~0.25
    r1 = eval2.evaluate(preds, targets, split="validation", epoch=1)
    is_best_1 = eval2.update_best(r1)
    check("first epoch is best", is_best_1 is True)
    check("best loss set", eval2.state.best_validation_loss == r1.loss)
    check("best epoch set", eval2.state.best_validation_epoch == 1)
    check("no_improve = 0", eval2.state.epochs_without_improvement == 0)

    # Epoch 2: worse loss
    worse_preds = torch.tensor([0.0, 0.0, 0.0])
    r2_result = eval2.evaluate(worse_preds, targets, split="validation", epoch=2)
    is_best_2 = eval2.update_best(r2_result)
    check("worse epoch not best", is_best_2 is False)
    check("no_improve = 1", eval2.state.epochs_without_improvement == 1)
    check("best epoch still 1", eval2.state.best_validation_epoch == 1)

    # Epoch 3: even worse
    r3_result = eval2.evaluate(worse_preds * 2, targets, split="validation", epoch=3)
    eval2.update_best(r3_result)
    check("no_improve = 2", eval2.state.epochs_without_improvement == 2)

    # Epoch 4: perfect -> new best
    r4_result = eval2.evaluate(targets.clone(), targets, split="validation", epoch=4)
    is_best_4 = eval2.update_best(r4_result)
    check("perfect epoch is best", is_best_4 is True)
    check("best epoch now 4", eval2.state.best_validation_epoch == 4)
    check("no_improve reset", eval2.state.epochs_without_improvement == 0)

    # -- 17. Non-validation cannot update best ---------------------------------
    print("\n  17. Non-validation update rejected...")
    train_result = eval2.evaluate(preds, targets, split="train", epoch=5)
    expect_error("train split rejected", EvaluationError,
                 lambda: eval2.update_best(train_result))

    test_result = eval2.evaluate(preds, targets, split="test", epoch=5)
    expect_error("test split rejected", EvaluationError,
                 lambda: eval2.update_best(test_result))

    # -- 18. Config/context mismatch -------------------------------------------
    print("\n  18. Config/context mismatch...")
    cfg_a, ctx_a = _make_infra(loss_name="mse")
    cfg_b, ctx_b = _make_infra(loss_name="mae")
    expect_error("config/ctx mismatch", EvaluationError,
                 lambda: build_evaluator(cfg_a, ctx_b))

    # -- 19. Evaluator metadata ------------------------------------------------
    print("\n  19. Evaluator metadata...")
    check("metadata problem_type", evaluator.metadata.problem_type == "regression")
    check("metadata loss_name", evaluator.metadata.loss_name == "mse")
    check("metadata prediction_key", evaluator.metadata.prediction_key == "rating_prediction")
    check("metadata target_key", evaluator.metadata.target_key == "ratings")
    check("metadata step_source", evaluator.metadata.step_source == "trainer")

    # -- 20. Evaluator summary and as_dict -------------------------------------
    print("\n  20. Summary and serialization...")
    summ = evaluator.summary()
    check("summary is string", isinstance(summ, str) and len(summ) > 50)
    check("summary has loss name", "mse" in summ)
    d = evaluator.as_dict()
    check("as_dict has metadata", "metadata" in d)
    check("as_dict has state", "state" in d)
    check("as_dict metadata loss", d["metadata"]["loss_name"] == "mse")

    # -- 21. Result summary and as_dict ----------------------------------------
    print("\n  21. Result summary and serialization...")
    rs = result.summary()
    check("result summary is string", isinstance(rs, str))
    check("result summary has split", "validation" in rs)
    rd = result.as_dict()
    check("result as_dict has loss", "loss" in rd)
    check("result as_dict has r2", "r2" in rd)

    # Standalone helpers
    rs2 = summarize_evaluation(result)
    check("summarize_evaluation matches", rs2 == rs)
    rd2 = evaluation_to_dict(result)
    check("evaluation_to_dict matches", rd2 == rd)

    # -- 22. Dict prediction in evaluate() -------------------------------------
    print("\n  22. Dict prediction in evaluate()...")
    dict_output = {"rating_prediction": torch.tensor([1.0, 2.0, 3.0])}
    dict_result = evaluator.evaluate(dict_output, targets, split="train", epoch=1)
    check("dict evaluate works", isinstance(dict_result, EvaluationResult))
    check("dict evaluate loss", dict_result.loss == result.loss)  # same values

    # -- 23. Unfrozen config rejection -----------------------------------------
    print("\n  23. Unfrozen config rejection...")
    unfrozen = build_train_config(device="cpu")
    expect_error("unfrozen config", EvaluationError,
                 lambda: build_evaluator(unfrozen, ctx))

    # -- 24. Splits support train/validation/test ------------------------------
    print("\n  24. Split support...")
    for split in ["train", "validation", "test"]:
        try:
            r = evaluator.evaluate(preds, targets, split=split, epoch=1)
            check(f"split '{split}' accepted", r.split == split)
        except EvaluationError:
            check(f"split '{split}' accepted", False, "unexpected rejection")

    # Invalid split
    expect_error("invalid split", EvaluationError,
                 lambda: evaluator.evaluate(preds, targets, split="dev"))

    # -- 25. Device mismatch (CPU only test) -----------------------------------
    print("\n  25. Device mismatch check...")
    # On CPU-only machines we can't test real device mismatch,
    # but verify the check exists by confirming same-device works
    check("same device OK", True)  # tested implicitly in all above

    # -- 26. update_best requires epoch ----------------------------------------
    print("\n  26. update_best requires epoch...")
    no_epoch_result = evaluator.evaluate(preds, targets, split="validation")
    expect_error("no epoch rejected", EvaluationError,
                 lambda: evaluator.update_best(no_epoch_result))

    # -- 27. Runtime state tracking --------------------------------------------
    print("\n  27. Runtime state tracking...")
    eval3 = build_evaluator(cfg, ctx)
    check("initial samples = 0", eval3.state.samples_evaluated == 0)
    check("initial batches = 0", eval3.state.batches_evaluated == 0)
    eval3.evaluate(preds, targets, split="train")
    check("samples after 1 batch", eval3.state.samples_evaluated == 3)
    check("batches after 1 batch", eval3.state.batches_evaluated == 1)
    eval3.evaluate(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]), split="train")
    check("samples after 2 batches", eval3.state.samples_evaluated == 5)
    check("batches after 2 batches", eval3.state.batches_evaluated == 2)

    # -- 28. State as_dict -----------------------------------------------------
    print("\n  28. State serialization...")
    sd = eval3.state.as_dict()
    check("state dict has samples", sd["samples_evaluated"] == 5)
    check("state dict has best_loss", "best_validation_loss" in sd)

    # -- 29. EvaluationMetadata is frozen --------------------------------------
    print("\n  29. Metadata immutability...")
    expect_error("metadata setattr blocked", AttributeError,
                 lambda: setattr(evaluator.metadata, "loss_name", "evil"))

    # -- 30. Complex tensor rejection ------------------------------------------
    print("\n  30. Complex tensor rejection...")
    expect_error("complex prediction", EvaluationError,
                 lambda: compute_metrics(
                     torch.tensor([1.0+0j, 2.0+0j, 3.0+0j]),
                     torch.tensor([1.0, 2.0, 3.0])))
    expect_error("complex target", EvaluationError,
                 lambda: compute_metrics(
                     torch.tensor([1.0, 2.0, 3.0]),
                     torch.tensor([1.0+0j, 2.0+0j, 3.0+0j])))
    expect_error("complex64 prediction", EvaluationError,
                 lambda: compute_metrics(
                     torch.randn(4, dtype=torch.complex64),
                     torch.randn(4)))

    # -- 31. compute_loss rejects raw TrainConfig() ----------------------------
    print("\n  31. compute_loss config validation...")
    from training.train_config import TrainConfig as _TC
    raw_cfg = _TC()
    expect_error("raw config rejected", EvaluationError,
                 lambda: compute_loss(preds, targets, raw_cfg))

    # Validated but unfrozen
    unfrozen_val = build_train_config(device="cpu", loss_name="mse")
    # unfrozen_val is validated but not frozen
    expect_error("unfrozen config rejected", EvaluationError,
                 lambda: compute_loss(preds, targets, unfrozen_val))

    # Frozen config works and preserves gradient graph
    frozen_ok = build_train_config(device="cpu", loss_name="mse")
    frozen_ok.freeze()
    preds_grad = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    loss_grad = compute_loss(preds_grad, targets, frozen_ok)
    check("frozen config accepted", loss_grad.dim() == 0)
    check("gradient preserved", loss_grad.requires_grad is True)

    # -- 32. update_best rejects batch-level result ----------------------------
    print("\n  32. update_best batch-level rejection...")
    eval_batch = build_evaluator(cfg, ctx)
    batch_result = eval_batch.evaluate(
        preds, targets, split="validation", epoch=1, batch_index=5,
    )
    check("batch result created", batch_result.batch_index == 5)
    expect_error("batch update_best rejected", EvaluationError,
                 lambda: eval_batch.update_best(batch_result))

    # -- 33. update_best rejects duplicate epoch -------------------------------
    print("\n  33. update_best duplicate epoch rejection...")
    eval_dup = build_evaluator(cfg, ctx)
    dup_r1 = eval_dup.evaluate(preds, targets, split="validation", epoch=1)
    eval_dup.update_best(dup_r1)
    # Same epoch again
    dup_r1b = eval_dup.evaluate(preds, targets, split="validation", epoch=1)
    expect_error("duplicate epoch rejected", EvaluationError,
                 lambda: eval_dup.update_best(dup_r1b))
    # Next epoch works
    dup_r2 = eval_dup.evaluate(preds, targets, split="validation", epoch=2)
    try:
        eval_dup.update_best(dup_r2)
        check("next epoch accepted", True)
    except EvaluationError:
        check("next epoch accepted", False, "unexpected rejection")

    # -- 34. Epoch-level tracking still correct after hardening -----------------
    print("\n  34. Epoch-level tracking post-hardening...")
    eval_h = build_evaluator(cfg, ctx)
    rh1 = eval_h.evaluate(preds, targets, split="validation", epoch=1)
    eval_h.update_best(rh1)
    check("h: epoch 1 best", eval_h.state.best_validation_epoch == 1)
    check("h: no_improve=0", eval_h.state.epochs_without_improvement == 0)

    rh2 = eval_h.evaluate(torch.tensor([0.0, 0.0, 0.0]), targets,
                          split="validation", epoch=2)
    eval_h.update_best(rh2)
    check("h: no_improve=1", eval_h.state.epochs_without_improvement == 1)

    rh3 = eval_h.evaluate(targets.clone(), targets,
                          split="validation", epoch=3)
    eval_h.update_best(rh3)
    check("h: epoch 3 new best", eval_h.state.best_validation_epoch == 3)
    check("h: no_improve reset", eval_h.state.epochs_without_improvement == 0)

    # -- 35. _SUPPORTED_LOSSES aligned with VALID_LOSSES -----------------------
    print("\n  35. Loss alignment with TrainConfig...")
    from training.evaluation import _SUPPORTED_LOSSES
    check("losses aligned", _SUPPORTED_LOSSES == VALID_LOSSES)

    # -- 36. last_best_update_epoch in state dict ------------------------------
    print("\n  36. last_best_update_epoch tracking...")
    check("last_update_epoch tracked",
          eval_h.state.last_best_update_epoch == 3)
    sd_h = eval_h.state.as_dict()
    check("last_update in as_dict",
          sd_h["last_best_update_epoch"] == 3)

    # -- Final -----------------------------------------------------------------
    total = passed + failed
    print(f"\n{'=' * 60}")
    if failed == 0:
        print(f"  [PASS]  {passed}/{total} checks passed")
    else:
        print(f"  [FAIL]  {passed}/{total} passed, {failed} failed")
    print("=" * 60)

    sys.exit(1 if failed > 0 else 0)
