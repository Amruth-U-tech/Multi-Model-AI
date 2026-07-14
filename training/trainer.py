# =============================================================================
# training/trainer.py
# Training Orchestrator -- Multimodal AI Pipeline
# =============================================================================
#
# Single orchestration authority for the training subsystem.
#
# Trainer owns:
#   - Time (epoch, batch, global_step)
#   - Events (chronological, monotonic, terminal-safe)
#   - History (epoch-level, append-only, synchronized)
#   - Checkpointing (save/load/resume/interrupted)
#   - Dashboard (assembled from subsystem summaries, never raw internals)
#   - State machine (CREATED -> READY -> RUNNING -> COMPLETED / FAILED / INTERRUPTED)
#
# Trainer does NOT own:
#   - Loss logic              -> training/evaluation.py
#   - Metric logic            -> training/evaluation.py
#   - Optimizer construction  -> training/optimizer.py
#   - Scheduler mathematics   -> training/scheduler.py
#   - Path root detection     -> configs/paths.py
#   - Dataset loading         -> data_pipeline/
#   - Tokenization            -> data_pipeline/tokenization.py
#   - Transforms              -> data_pipeline/transforms.py
#   - Collate logic           -> data_pipeline/collate.py
#   - Model architecture      -> models/
#
# Compatible with:
#   - CPU / CUDA execution
#   - FP16 mixed precision (AMP)
#   - Gradient accumulation
#   - Gradient clipping
#   - Tesla T4 / Colab execution
#   - Windows / Linux
# =============================================================================

from __future__ import annotations

import copy
import datetime
import logging
import os
import platform
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Project Import Routing (local + Colab compatible)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn

from training.train_config import TrainConfig, build_train_config
from training.run_context import RunContext, build_run_context
from training.optimizer import (
    build_optimizer,
    get_optimizer_metadata,
    summarize_optimizer,
    optimizer_to_dict,
)
from training.scheduler import (
    STEP_POLICY_EPOCH,
    STEP_POLICY_VALIDATION_METRIC,
    build_scheduler,
    get_scheduler_metadata,
    summarize_scheduler,
    scheduler_to_dict,
)
from training.evaluation import (
    Evaluator,
    EvaluationResult,
    build_evaluator,
    compute_loss,
    extract_prediction,
    summarize_evaluation,
    evaluation_to_dict,
)


# =============================================================================
# Logging
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Constants
# =============================================================================

_REQUIRED_MODEL_KEYS = frozenset({
    "image_encoder",
    "text_encoder",
    "tabular_encoder",
    "fusion_model",
})

_REQUIRED_BATCH_KEYS = frozenset({
    "images",
    "input_ids",
    "attention_mask",
    "tabular",
    "ratings",
})

_TENSOR_BATCH_KEYS = frozenset({
    "images",
    "input_ids",
    "attention_mask",
    "tabular",
    "ratings",
})

_CHECKPOINT_VERSION = 1
_TRAINER_SCHEMA_VERSION = 1

_CHECKPOINT_REQUIRED_KEYS = frozenset({
    "checkpoint_version",
    "trainer_schema_version",
    "trainer_class",
    "kind",
    "epoch",
    "global_step",
    "model_state_dict",
    "optimizer_state_dict",
    "config",
    "run_context",
    "trainer_runtime",
    "training_history",
    "checkpoint_state",
    "evaluation",
    "reproducibility",
})


# =============================================================================
# 2. TrainerError
# =============================================================================

class TrainerError(RuntimeError):
    """Structured training error with full diagnostic context.

    Every TrainerError carries stage, event, epoch, batch, subsystem,
    received value, expected value, and resolution guidance.
    """

    def __init__(
        self,
        stage: str,
        event: str = "",
        epoch: Optional[int] = None,
        batch: Optional[int] = None,
        subsystem: str = "",
        received: Any = "",
        expected: Any = "",
        resolution: str = "",
    ):
        self.stage = stage
        self.event = event
        self.epoch = epoch
        self.batch = batch
        self.subsystem = subsystem
        self.received = received
        self.expected = expected
        self.resolution = resolution

        parts = [
            "[TRAINER ERROR]",
            f"  Stage      : {stage}",
        ]
        if event:
            parts.append(f"  Event      : {event}")
        if epoch is not None:
            parts.append(f"  Epoch      : {epoch}")
        if batch is not None:
            parts.append(f"  Batch      : {batch}")
        if subsystem:
            parts.append(f"  Subsystem  : {subsystem}")
        if received != "":
            parts.append(f"  Received   : {received}")
        if expected != "":
            parts.append(f"  Expected   : {expected}")
        if resolution:
            parts.append(f"  Resolution : {resolution}")

        super().__init__("\n".join(parts))


# =============================================================================
# 3. Internal Enums
# =============================================================================

class _TrainingEvent(Enum):
    """Chronological training events emitted by the Trainer."""
    INITIALIZED              = "initialized"
    TRAINING_STARTED         = "training_started"
    EPOCH_STARTED            = "epoch_started"
    BATCH_STARTED            = "batch_started"
    BATCH_TRANSFERRED        = "batch_transferred"
    FORWARD_STARTED          = "forward_started"
    FORWARD_COMPLETED        = "forward_completed"
    LOSS_COMPUTED            = "loss_computed"
    BACKWARD_COMPLETED       = "backward_completed"
    GRADIENT_CLIPPED         = "gradient_clipped"
    OPTIMIZER_UPDATED        = "optimizer_updated"
    SCHEDULER_UPDATED        = "scheduler_updated"
    VALIDATION_STARTED       = "validation_started"
    VALIDATION_BATCH_COMPLETED = "validation_batch_completed"
    VALIDATION_COMPLETED     = "validation_completed"
    CHECKPOINT_SAVED         = "checkpoint_saved"
    EPOCH_COMPLETED          = "epoch_completed"
    TRAINING_COMPLETED       = "training_completed"
    TRAINING_INTERRUPTED     = "training_interrupted"
    TRAINING_FAILED          = "training_failed"
    RESUME_STARTED           = "resume_started"
    RESUME_COMPLETED         = "resume_completed"


class _TrainerStatus(Enum):
    """Deterministic lifecycle states."""
    CREATED        = "created"
    READY          = "ready"
    RUNNING        = "running"
    VALIDATING     = "validating"
    CHECKPOINTING  = "checkpointing"
    INTERRUPTED    = "interrupted"
    FAILED         = "failed"
    COMPLETED      = "completed"


# Status transition table
_VALID_TRANSITIONS = {
    _TrainerStatus.CREATED:       {_TrainerStatus.READY, _TrainerStatus.FAILED},
    _TrainerStatus.READY:         {_TrainerStatus.RUNNING, _TrainerStatus.FAILED},
    _TrainerStatus.RUNNING:       {_TrainerStatus.VALIDATING, _TrainerStatus.CHECKPOINTING,
                                   _TrainerStatus.COMPLETED, _TrainerStatus.INTERRUPTED,
                                   _TrainerStatus.FAILED},
    _TrainerStatus.VALIDATING:    {_TrainerStatus.RUNNING, _TrainerStatus.CHECKPOINTING,
                                   _TrainerStatus.INTERRUPTED, _TrainerStatus.FAILED},
    _TrainerStatus.CHECKPOINTING: {_TrainerStatus.RUNNING, _TrainerStatus.COMPLETED,
                                   _TrainerStatus.INTERRUPTED, _TrainerStatus.FAILED},
    _TrainerStatus.INTERRUPTED:   set(),  # terminal
    _TrainerStatus.FAILED:        set(),  # terminal
    _TrainerStatus.COMPLETED:     set(),  # terminal
}

_TERMINAL_STATUSES = frozenset({
    _TrainerStatus.INTERRUPTED,
    _TrainerStatus.FAILED,
    _TrainerStatus.COMPLETED,
})


# =============================================================================
# 4. Internal Dataclasses
# =============================================================================

@dataclass
class _TrainerRuntimeState:
    """Mutable internal runtime state. Exposed as dict copy only."""
    status: str = _TrainerStatus.CREATED.value
    current_epoch: int = 0
    current_batch: int = 0
    total_epochs: int = 0
    global_step: int = 0
    optimizer_steps: int = 0
    last_event: str = ""
    healthy: bool = True
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    elapsed_seconds: float = 0.0
    estimated_remaining_seconds: Optional[float] = None
    last_warning: Optional[str] = None
    last_error: Optional[str] = None
    interrupted: bool = False
    failed: bool = False
    # AMP state
    amp_status: str = "disabled"
    amp_fallback_reason: Optional[str] = None
    # Per-epoch timing breakdown (seconds)
    train_time_seconds: float = 0.0
    val_time_seconds: float = 0.0
    checkpoint_time_seconds: float = 0.0
    # Warning counter
    warning_count: int = 0
    # GPU peak memory (MB, None if unavailable)
    peak_gpu_memory_mb: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "current_epoch": self.current_epoch,
            "current_batch": self.current_batch,
            "total_epochs": self.total_epochs,
            "global_step": self.global_step,
            "optimizer_steps": self.optimizer_steps,
            "last_event": self.last_event,
            "healthy": self.healthy,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "elapsed_seconds": self.elapsed_seconds,
            "estimated_remaining_seconds": self.estimated_remaining_seconds,
            "last_warning": self.last_warning,
            "last_error": self.last_error,
            "interrupted": self.interrupted,
            "failed": self.failed,
            "amp_status": self.amp_status,
            "amp_fallback_reason": self.amp_fallback_reason,
            "train_time_seconds": self.train_time_seconds,
            "val_time_seconds": self.val_time_seconds,
            "checkpoint_time_seconds": self.checkpoint_time_seconds,
            "warning_count": self.warning_count,
            "peak_gpu_memory_mb": self.peak_gpu_memory_mb,
        }


@dataclass
class _TrainingHistory:
    """Mutable epoch-level training history. Append-only, synchronized."""
    epoch: List[int] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    validation_loss: List[float] = field(default_factory=list)
    validation_rmse: List[float] = field(default_factory=list)
    validation_mae: List[float] = field(default_factory=list)
    validation_r2: List[float] = field(default_factory=list)
    learning_rate: List[float] = field(default_factory=list)
    epoch_duration_seconds: List[float] = field(default_factory=list)
    checkpoint_saved: List[bool] = field(default_factory=list)
    best_epoch: Optional[int] = None
    best_validation_loss: Optional[float] = None

    def _check_lengths(self) -> bool:
        """Verify all list fields have identical length."""
        lists = [
            self.epoch, self.train_loss, self.validation_loss,
            self.validation_rmse, self.validation_mae, self.validation_r2,
            self.learning_rate, self.epoch_duration_seconds,
            self.checkpoint_saved,
        ]
        lengths = [len(lst) for lst in lists]
        return len(set(lengths)) <= 1

    def as_dict(self) -> Dict[str, Any]:
        return {
            "epoch": list(self.epoch),
            "train_loss": list(self.train_loss),
            "validation_loss": list(self.validation_loss),
            "validation_rmse": list(self.validation_rmse),
            "validation_mae": list(self.validation_mae),
            "validation_r2": list(self.validation_r2),
            "learning_rate": list(self.learning_rate),
            "epoch_duration_seconds": list(self.epoch_duration_seconds),
            "checkpoint_saved": list(self.checkpoint_saved),
            "best_epoch": self.best_epoch,
            "best_validation_loss": self.best_validation_loss,
        }


@dataclass
class _CheckpointState:
    """Mutable internal checkpoint tracking."""
    latest_checkpoint: Optional[str] = None
    best_checkpoint: Optional[str] = None
    last_saved_epoch: Optional[int] = None
    best_epoch: Optional[int] = None
    checkpoint_count: int = 0
    resume_checkpoint: Optional[str] = None
    resume_epoch: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "latest_checkpoint": self.latest_checkpoint,
            "best_checkpoint": self.best_checkpoint,
            "last_saved_epoch": self.last_saved_epoch,
            "best_epoch": self.best_epoch,
            "checkpoint_count": self.checkpoint_count,
            "resume_checkpoint": self.resume_checkpoint,
            "resume_epoch": self.resume_epoch,
        }


@dataclass(frozen=True)
class _ReproducibilitySnapshot:
    """Immutable reproducibility context captured at training start."""
    experiment_name: str
    seed: int
    python_version: str
    platform_info: str
    torch_version: str
    cuda_available: bool
    device: str
    mixed_precision_requested: bool
    mixed_precision_available: bool
    config_snapshot: Dict[str, Any]
    run_context_snapshot: Dict[str, Any]
    created_at: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "python_version": self.python_version,
            "platform_info": self.platform_info,
            "torch_version": self.torch_version,
            "cuda_available": self.cuda_available,
            "device": self.device,
            "mixed_precision_requested": self.mixed_precision_requested,
            "mixed_precision_available": self.mixed_precision_available,
            "config_snapshot": dict(self.config_snapshot),
            "run_context_snapshot": dict(self.run_context_snapshot),
            "created_at": self.created_at,
        }


# =============================================================================
# 5. Validation Helpers (Module-Level)
# =============================================================================

def _sanitize_experiment_name(name: str) -> str:
    """Sanitize experiment name for use as a filesystem directory name."""
    clean = re.sub(r'[^\w\-.]', '_', name.strip())
    clean = re.sub(r'_+', '_', clean).strip('_')
    return clean[:128] if clean else "unnamed_experiment"


def _assert_child_path(base: Path, candidate: Path, stage: str) -> Path:
    """Verify candidate path is inside base directory using resolved paths.

    Returns:
        Resolved candidate path.

    Raises:
        TrainerError: If candidate escapes base directory.
    """
    try:
        resolved_base = base.resolve()
        resolved_candidate = candidate.resolve()
        resolved_candidate.relative_to(resolved_base)
        return resolved_candidate
    except ValueError:
        raise TrainerError(
            stage, "path_safety",
            received=str(candidate),
            expected=f"path inside {base}",
            resolution="Path traversal detected. Use a filename without directory separators.",
        )
    except OSError as exc:
        raise TrainerError(
            stage, "path_safety",
            received=str(exc)[:200],
            expected="valid filesystem path",
            resolution="Check directory accessibility.",
        ) from exc


def _gpu_memory_snapshot() -> Dict[str, Any]:
    """Lightweight GPU memory snapshot. Safe on CPU-only systems.

    Returns:
        Dict with cuda_available, device_name, allocated_mb, reserved_mb,
        max_allocated_mb. All values are None when CUDA is unavailable.
    """
    snapshot = {
        "cuda_available": torch.cuda.is_available(),
        "device_name": None,
        "allocated_mb": None,
        "reserved_mb": None,
        "max_allocated_mb": None,
    }
    if torch.cuda.is_available():
        try:
            device = torch.cuda.current_device()
            snapshot["device_name"] = torch.cuda.get_device_name(device)
            snapshot["allocated_mb"] = round(
                torch.cuda.memory_allocated(device) / (1024 * 1024), 2,
            )
            snapshot["reserved_mb"] = round(
                torch.cuda.memory_reserved(device) / (1024 * 1024), 2,
            )
            snapshot["max_allocated_mb"] = round(
                torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2,
            )
        except Exception:
            pass  # GPU query failed -- return partial snapshot
    return snapshot


def _validate_loader_like(loader: Any, name: str) -> None:
    """Validate that a loader is an iterable, DataLoader-like object.

    Rejects None, scalars, strings, bytes, and plain dicts.
    Calls iter() once without consuming data to verify iterability.
    """
    if loader is None:
        raise TrainerError(
            "constructor", "validation", subsystem=name,
            received="None",
            expected="iterable DataLoader-like object",
            resolution="Pass a DataLoader, list of batch dicts, or iterable batch source.",
        )

    # Reject scalar types and string/bytes/dict
    _REJECTED_TYPES = (int, float, bool, str, bytes, dict)
    if isinstance(loader, _REJECTED_TYPES):
        raise TrainerError(
            "constructor", "validation", subsystem=name,
            received=type(loader).__name__,
            expected="iterable DataLoader-like object",
            resolution="Pass a DataLoader, list of batch dicts, or iterable batch source.",
        )

    # Verify iterability without consuming data or triggering __iter__ side effects
    if not hasattr(loader, '__iter__'):
        raise TrainerError(
            "constructor", "validation", subsystem=name,
            received=type(loader).__name__,
            expected="iterable DataLoader-like object",
            resolution="Pass a DataLoader, list of batch dicts, or iterable batch source.",
        )


def _validate_constructor_inputs(
    config: Any,
    run_context: Any,
    model_bundle: Any,
    optimizer: Any,
    scheduler: Any,
    evaluator: Any,
    train_loader: Any,
    val_loader: Any,
    render_dashboard: Any,
) -> None:
    """Validate all constructor inputs. Raises TrainerError on failure."""

    # 1. Config type and frozen state
    if not isinstance(config, TrainConfig):
        raise TrainerError(
            "constructor", "validation", subsystem="config",
            received=type(config).__name__, expected="TrainConfig",
            resolution="Pass a frozen TrainConfig instance.",
        )
    if not config.is_frozen:
        raise TrainerError(
            "constructor", "validation", subsystem="config",
            received="unfrozen config", expected="frozen config",
            resolution="Call config.freeze() before building trainer.",
        )

    # 2. RunContext type and config identity
    if not isinstance(run_context, RunContext):
        raise TrainerError(
            "constructor", "validation", subsystem="run_context",
            received=type(run_context).__name__, expected="RunContext",
            resolution="Pass a RunContext built from the same config.",
        )
    if run_context.config is not config:
        raise TrainerError(
            "constructor", "validation", subsystem="run_context",
            received="run_context.config is not config",
            expected="run_context.config is config (object identity)",
            resolution="Build RunContext from the same TrainConfig instance.",
        )

    # 3. Model bundle type and keys
    if not isinstance(model_bundle, nn.ModuleDict):
        raise TrainerError(
            "constructor", "validation", subsystem="model_bundle",
            received=type(model_bundle).__name__,
            expected="torch.nn.ModuleDict",
            resolution="Wrap model components in nn.ModuleDict.",
        )
    missing_keys = _REQUIRED_MODEL_KEYS - set(model_bundle.keys())
    if missing_keys:
        raise TrainerError(
            "constructor", "validation", subsystem="model_bundle",
            received=f"keys {sorted(model_bundle.keys())}",
            expected=f"keys {sorted(_REQUIRED_MODEL_KEYS)}",
            resolution=f"Add missing keys: {sorted(missing_keys)}",
        )
    for key in _REQUIRED_MODEL_KEYS:
        component = model_bundle[key]
        if not isinstance(component, nn.Module):
            raise TrainerError(
                "constructor", "validation", subsystem="model_bundle",
                received=f"model_bundle['{key}'] is {type(component).__name__}",
                expected="torch.nn.Module",
                resolution=f"Ensure model_bundle['{key}'] is an nn.Module.",
            )

    # 4. Optimizer
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TrainerError(
            "constructor", "validation", subsystem="optimizer",
            received=type(optimizer).__name__,
            expected="torch.optim.Optimizer",
            resolution="Pass an optimizer from build_optimizer().",
        )

    # 5. Scheduler metadata
    sched_meta = get_scheduler_metadata(scheduler)
    if sched_meta is None:
        raise TrainerError(
            "constructor", "validation", subsystem="scheduler",
            received="no scheduler metadata",
            expected="scheduler with attached SchedulerMetadata",
            resolution="Pass a scheduler from build_scheduler().",
        )

    # 6. Evaluator
    if not isinstance(evaluator, Evaluator):
        raise TrainerError(
            "constructor", "validation", subsystem="evaluator",
            received=type(evaluator).__name__,
            expected="Evaluator",
            resolution="Pass an evaluator from build_evaluator().",
        )

    # 7. Train loader
    _validate_loader_like(train_loader, "train_loader")

    # 8. Val loader requirements
    needs_val = False
    val_reason = ""
    if sched_meta.step_policy == STEP_POLICY_VALIDATION_METRIC:
        needs_val = True
        val_reason = "scheduler step_policy is 'validation_metric'"
    if config.save_best:
        needs_val = True
        val_reason = "config.save_best is True"

    if needs_val and val_loader is None:
        raise TrainerError(
            "constructor", "validation", subsystem="val_loader",
            received="None",
            expected=f"validation loader (required because {val_reason})",
            resolution="Provide a val_loader or disable save_best/plateau.",
        )

    # 9. Val loader validation (when provided)
    if val_loader is not None:
        _validate_loader_like(val_loader, "val_loader")

    # 10. Loader identity leakage
    if val_loader is not None and val_loader is train_loader:
        raise TrainerError(
            "constructor", "validation", subsystem="loaders",
            received="val_loader is train_loader (same object)",
            expected="distinct loader objects",
            resolution="Use separate DataLoader instances for train and val.",
        )

    # 10. render_dashboard
    if not isinstance(render_dashboard, bool):
        raise TrainerError(
            "constructor", "validation", subsystem="render_dashboard",
            received=type(render_dashboard).__name__,
            expected="bool",
            resolution="Pass render_dashboard=True or False.",
        )


# =============================================================================
# 6. Trainer Class
# =============================================================================

class Trainer:
    """Training orchestrator. Coordinates time, events, and subsystems.

    Trainer does not redefine subsystem logic. It schedules, routes,
    observes, checks, saves, resumes, and reports.

    Public API:
        train()          -> Run the complete training loop.
        resume()         -> Resume from a checkpoint.
        summary()        -> Human-readable summary string.
        runtime_state()  -> Dict copy of current runtime state.
        history()        -> Dict copy of training history.
        to_dict()        -> Full serializable snapshot.
        as_dict()        -> Alias for to_dict().
    """

    def __init__(
        self,
        *,
        config: TrainConfig,
        run_context: RunContext,
        model_bundle: nn.ModuleDict,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        evaluator: Evaluator,
        train_loader: Iterable,
        val_loader: Optional[Iterable] = None,
        render_dashboard: bool = True,
    ):
        # Validate all inputs before storing
        _validate_constructor_inputs(
            config, run_context, model_bundle, optimizer,
            scheduler, evaluator, train_loader, val_loader,
            render_dashboard,
        )

        # Store dependencies (never re-create them)
        self._config = config
        self._run_context = run_context
        self._model_bundle = model_bundle
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._evaluator = evaluator
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._render_dashboard_flag = render_dashboard

        # Scheduler metadata (validated above)
        self._scheduler_meta = get_scheduler_metadata(scheduler)

        # Internal state
        self._runtime = _TrainerRuntimeState(
            total_epochs=config.epochs,
        )
        self._history = _TrainingHistory()
        self._checkpoint_state = _CheckpointState()
        self._repro: Optional[_ReproducibilitySnapshot] = None

        # AMP state
        self._amp_enabled = False
        self._scaler: Optional[torch.cuda.amp.GradScaler] = None

        # Training clock
        self._train_start_time: Optional[float] = None

        # Start epoch (may be overridden by resume)
        self._start_epoch = 1

        # Event log (lightweight list of event names for diagnostics)
        self._event_log: List[str] = []

        # Checkpoint directory
        exp_name_clean = _sanitize_experiment_name(config.experiment_name)
        self._checkpoint_dir = Path(run_context.checkpoint_dir) / exp_name_clean

        # Initialize
        self._emit_event(_TrainingEvent.INITIALIZED)
        self._set_status(_TrainerStatus.READY)

    # =========================================================================
    # Public API
    # =========================================================================

    def train(self) -> Dict[str, Any]:
        """Run the complete training loop.

        Returns:
            Dict containing final history and runtime state.

        Raises:
            TrainerError: On any orchestration or subsystem failure.
        """
        try:
            self._validate_ready_state()

            # Resume if requested
            if self._config.resume:
                self._do_resume()

            # Capture reproducibility snapshot
            self._capture_reproducibility()

            # Prepare model and AMP
            self._prepare_model()
            self._prepare_amp()

            # Begin training
            self._train_start_time = time.perf_counter()
            self._runtime.started_at = datetime.datetime.now().isoformat()
            self._set_status(_TrainerStatus.RUNNING)
            self._emit_event(_TrainingEvent.TRAINING_STARTED)

            if self._render_dashboard_flag:
                self._render_dashboard()

            # Main training loop
            for epoch in range(self._start_epoch, self._config.epochs + 1):
                self._emit_event(_TrainingEvent.EPOCH_STARTED, epoch=epoch)
                self._runtime.current_epoch = epoch

                # Pre-epoch runtime contract verification
                self._check_epoch_invariants(epoch)

                epoch_start = time.perf_counter()

                # -- Train one epoch --
                train_start = time.perf_counter()
                avg_train_loss = self._train_one_epoch(epoch)
                self._runtime.train_time_seconds = time.perf_counter() - train_start

                # -- Validation if due --
                val_result: Optional[EvaluationResult] = None
                is_best = False
                val_start = time.perf_counter()
                if self._val_loader is not None and epoch % self._config.validation_frequency == 0:
                    val_result = self._validate_one_epoch(epoch)
                    is_best = self._evaluator.update_best(val_result)
                self._runtime.val_time_seconds = time.perf_counter() - val_start

                # -- Step scheduler --
                self._step_scheduler_after_epoch(val_result)

                # -- Epoch duration --
                epoch_duration = time.perf_counter() - epoch_start

                # -- Checkpoint intent (decisions only, no I/O yet) --
                will_save_latest = self._should_save_latest(epoch)
                will_save_best = self._should_save_best(is_best)
                checkpoint_saved = will_save_latest or will_save_best

                # -- Update history BEFORE checkpoint save --
                self._update_history(
                    epoch, avg_train_loss, val_result,
                    epoch_duration, checkpoint_saved,
                )

                # -- Save checkpoints (history is now current in payload) --
                ckpt_start = time.perf_counter()
                if will_save_latest:
                    self._save_checkpoint("latest", epoch)
                if will_save_best:
                    self._save_checkpoint("best", epoch)
                self._runtime.checkpoint_time_seconds = time.perf_counter() - ckpt_start

                # -- GPU memory snapshot at epoch end --
                gpu_snap = _gpu_memory_snapshot()
                if gpu_snap["max_allocated_mb"] is not None:
                    self._runtime.peak_gpu_memory_mb = gpu_snap["max_allocated_mb"]

                # -- Update timing --
                self._update_elapsed_and_eta(epoch)

                # -- Dashboard --
                if self._render_dashboard_flag:
                    self._render_dashboard()

                self._emit_event(_TrainingEvent.EPOCH_COMPLETED, epoch=epoch)

            # Training complete
            self._runtime.completed_at = datetime.datetime.now().isoformat()
            self._set_status(_TrainerStatus.COMPLETED)
            self._emit_event(_TrainingEvent.TRAINING_COMPLETED)

            if self._render_dashboard_flag:
                self._render_dashboard()

            return {
                "history": self.history(),
                "runtime_state": self.runtime_state(),
            }

        except KeyboardInterrupt:
            self._handle_keyboard_interrupt()
            return {
                "history": self.history(),
                "runtime_state": self.runtime_state(),
            }
        except TrainerError as exc:
            self._record_failure(exc)
            raise
        except Exception as exc:
            self._handle_failure(exc)
            raise

    def resume(self) -> None:
        """Resume training from a checkpoint.

        Delegates to train() which handles resume internally
        when config.resume is True.
        """
        if not self._config.resume:
            raise TrainerError(
                "resume", "validation",
                received="config.resume=False",
                expected="config.resume=True",
                resolution="Set config.resume=True before building trainer.",
            )
        self.train()

    def summary(self) -> str:
        """Human-readable trainer summary assembled from subsystem summaries."""
        return self._build_dashboard_string()

    def runtime_state(self) -> Dict[str, Any]:
        """Dict copy of current runtime state."""
        return self._runtime.as_dict()

    def history(self) -> Dict[str, Any]:
        """Dict copy of training history."""
        return self._history.as_dict()

    def to_dict(self) -> Dict[str, Any]:
        """Full serializable snapshot of the trainer. All collections are copies."""
        return {
            "runtime_state": self.runtime_state(),
            "history": self.history(),
            "checkpoint_state": self._checkpoint_state.as_dict(),
            "evaluator": self._evaluator.as_dict(),
            "reproducibility": self._repro.as_dict() if self._repro else None,
            "event_count": len(self._event_log),
            "last_events": list(self._event_log[-10:]),  # copy, not view
        }

    def as_dict(self) -> Dict[str, Any]:
        """Alias for to_dict()."""
        return self.to_dict()

    # =========================================================================
    # Internal -- Initialization
    # =========================================================================

    def _validate_ready_state(self) -> None:
        """Ensure trainer is in READY state before training starts."""
        current = _TrainerStatus(self._runtime.status)
        if current != _TrainerStatus.READY:
            raise TrainerError(
                "train", "state_validation",
                received=current.value,
                expected=_TrainerStatus.READY.value,
                resolution="Trainer can only train() from READY state.",
            )

    def _capture_reproducibility(self) -> None:
        """Capture reproducibility snapshot once at training start."""
        if self._repro is not None:
            return  # Already captured (e.g., after resume)

        self._repro = _ReproducibilitySnapshot(
            experiment_name=self._config.experiment_name,
            seed=self._config.seed,
            python_version=platform.python_version(),
            platform_info=platform.platform(),
            torch_version=torch.__version__,
            cuda_available=torch.cuda.is_available(),
            device=self._run_context.device,
            mixed_precision_requested=self._run_context.mixed_precision_requested,
            mixed_precision_available=self._run_context.mixed_precision_available,
            config_snapshot=self._config.as_dict(),
            run_context_snapshot=self._run_context.as_dict(),
            created_at=datetime.datetime.now().isoformat(),
        )

    def _prepare_model(self) -> None:
        """Move model bundle to the target device."""
        device = torch.device(self._run_context.device)
        self._model_bundle.to(device)

        # Warn if model has zero trainable parameters
        total_trainable = sum(
            p.numel() for p in self._model_bundle.parameters() if p.requires_grad
        )
        if total_trainable == 0:
            self._mark_warning(
                "Model has zero trainable parameters. "
                "Training will proceed but weights will not update."
            )

    def _prepare_amp(self) -> None:
        """Prepare AMP GradScaler if mixed precision is available and requested.

        Determines AMP capability once at initialization. Stores the chosen
        implementation for the entire run. Never branches on PyTorch version
        during the training loop.

        Records fallback reason when AMP is requested but unavailable.
        """
        if (self._run_context.mixed_precision_requested
                and self._run_context.mixed_precision_available):
            self._amp_enabled = True
            # Use modern torch.amp API when available, fallback to legacy
            self._amp_device_type = self._run_context.device_type
            try:
                self._scaler = torch.amp.GradScaler(self._amp_device_type, enabled=True)
            except (TypeError, AttributeError):
                self._scaler = torch.cuda.amp.GradScaler()
            self._runtime.amp_status = "enabled"
            self._runtime.amp_fallback_reason = None
        else:
            self._amp_enabled = False
            self._amp_device_type = self._run_context.device_type
            self._scaler = None
            if self._run_context.mixed_precision_requested:
                reason = ("CUDA not available" if not torch.cuda.is_available()
                          else "mixed precision not supported on this device")
                self._runtime.amp_status = "fallback_to_fp32"
                self._runtime.amp_fallback_reason = reason
                self._mark_warning(f"AMP requested but unavailable: {reason}. Falling back to FP32.")
            else:
                self._runtime.amp_status = "disabled"
                self._runtime.amp_fallback_reason = None

    # =========================================================================
    # Internal -- Event / State Machine
    # =========================================================================

    def _emit_event(
        self,
        event: _TrainingEvent,
        epoch: Optional[int] = None,
        batch: Optional[int] = None,
        **details: Any,
    ) -> None:
        """Record a training event."""
        self._runtime.last_event = event.value
        self._event_log.append(event.value)

    def _set_status(self, new_status: _TrainerStatus) -> None:
        """Transition the trainer to a new status, validating the transition."""
        current = _TrainerStatus(self._runtime.status)

        if current in _TERMINAL_STATUSES:
            raise TrainerError(
                "state_machine", "status_transition",
                received=f"{current.value} -> {new_status.value}",
                expected=f"{current.value} is terminal -- no transitions allowed",
                resolution="Trainer has already terminated.",
            )

        allowed = _VALID_TRANSITIONS.get(current, set())
        if new_status not in allowed:
            raise TrainerError(
                "state_machine", "status_transition",
                received=f"{current.value} -> {new_status.value}",
                expected=f"one of {sorted(s.value for s in allowed)}",
                resolution="Invalid state transition in trainer lifecycle.",
            )

        self._runtime.status = new_status.value

    def _mark_warning(self, message: str) -> None:
        """Record a warning without failing."""
        self._runtime.last_warning = message
        self._runtime.warning_count += 1
        logger.warning(message)

    def _record_failure(
        self,
        error: Exception,
        event: str = "",
    ) -> None:
        """Record failure metadata. Guards against double-recording.

        Sets status=failed, marks healthy=False, emits TRAINING_FAILED
        exactly once. KeyboardInterrupt is NOT routed here.
        """
        if self._runtime.failed:
            return  # Already recorded -- do not double-emit

        self._runtime.healthy = False
        self._runtime.failed = True
        self._runtime.last_error = str(error)[:500]
        self._runtime.completed_at = datetime.datetime.now().isoformat()

        # GPU snapshot at failure time
        gpu_snap = _gpu_memory_snapshot()
        if gpu_snap["max_allocated_mb"] is not None:
            self._runtime.peak_gpu_memory_mb = gpu_snap["max_allocated_mb"]

        # Transition to FAILED
        try:
            self._set_status(_TrainerStatus.FAILED)
        except TrainerError:
            self._runtime.status = _TrainerStatus.FAILED.value

        self._emit_event(_TrainingEvent.TRAINING_FAILED)

    # =========================================================================
    # Internal -- Batch Handling
    # =========================================================================

    def _validate_batch(self, batch: Any, batch_index: int, epoch: int) -> None:
        """Validate batch structure and contents."""
        if not isinstance(batch, dict):
            raise TrainerError(
                "batch_validation", "BATCH_STARTED",
                epoch=epoch, batch=batch_index,
                subsystem="data_pipeline",
                received=type(batch).__name__,
                expected="dict",
                resolution="DataLoader collate must return a dict.",
            )

        missing = _REQUIRED_BATCH_KEYS - set(batch.keys())
        if missing:
            raise TrainerError(
                "batch_validation", "BATCH_STARTED",
                epoch=epoch, batch=batch_index,
                subsystem="data_pipeline",
                received=f"keys {sorted(batch.keys())}",
                expected=f"required keys {sorted(_REQUIRED_BATCH_KEYS)}",
                resolution=f"Collate missing: {sorted(missing)}",
            )

        # Verify tensor keys are tensors and non-empty
        for key in _TENSOR_BATCH_KEYS:
            val = batch[key]
            if not isinstance(val, torch.Tensor):
                raise TrainerError(
                    "batch_validation", "BATCH_STARTED",
                    epoch=epoch, batch=batch_index,
                    subsystem="data_pipeline",
                    received=f"batch['{key}'] is {type(val).__name__}",
                    expected="torch.Tensor",
                    resolution=f"Collate must produce tensor for '{key}'.",
                )
            if val.numel() == 0:
                raise TrainerError(
                    "batch_validation", "BATCH_STARTED",
                    epoch=epoch, batch=batch_index,
                    subsystem="data_pipeline",
                    received=f"batch['{key}'] is empty (numel=0)",
                    expected="non-empty tensor",
                    resolution=f"Empty batch tensor for '{key}'.",
                )

        # Check consistent batch sizes across tensor keys
        batch_sizes = {key: batch[key].shape[0] for key in _TENSOR_BATCH_KEYS}
        unique_sizes = set(batch_sizes.values())
        if len(unique_sizes) > 1:
            raise TrainerError(
                "batch_validation", "BATCH_STARTED",
                epoch=epoch, batch=batch_index,
                subsystem="data_pipeline",
                received=f"inconsistent batch sizes: {batch_sizes}",
                expected="same batch size across all tensor keys",
                resolution="Check collate function alignment.",
            )

        # Reject B=0 batch size
        B = batch["ratings"].shape[0]
        if B == 0:
            raise TrainerError(
                "batch_validation", "BATCH_STARTED",
                epoch=epoch, batch=batch_index,
                subsystem="data_pipeline",
                received="batch_size=0",
                expected="batch_size >= 1",
                resolution="DataLoader produced an empty batch.",
            )

        # Ratings must be floating-point
        ratings = batch["ratings"]
        if not ratings.dtype.is_floating_point:
            raise TrainerError(
                "batch_validation", "BATCH_STARTED",
                epoch=epoch, batch=batch_index,
                subsystem="data_pipeline",
                received=f"ratings.dtype={ratings.dtype}",
                expected="floating-point dtype (float32, float64)",
                resolution="Cast ratings to float before collation.",
            )

        # Ratings must be finite
        if not torch.isfinite(ratings).all():
            raise TrainerError(
                "batch_validation", "BATCH_STARTED",
                epoch=epoch, batch=batch_index,
                subsystem="data_pipeline",
                received="ratings contain NaN or Inf",
                expected="all finite values",
                resolution="Check dataset for corrupt ratings.",
            )

    def _validate_prediction_contract(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        epoch: int,
        batch_index: int,
    ) -> None:
        """Validate prediction tensor contract after forward pass.

        Checks:
        - predictions is a tensor on the same device as targets
        - batch sizes match
        - shape is [B] or [B,1]
        - dtype is not bool or complex
        - values are finite
        """
        if not isinstance(predictions, torch.Tensor):
            raise TrainerError(
                "prediction_validation", "FORWARD_COMPLETED",
                epoch=epoch, batch=batch_index,
                subsystem="fusion_model",
                received=type(predictions).__name__,
                expected="torch.Tensor",
                resolution="extract_prediction must return a tensor.",
            )

        if predictions.device != targets.device:
            raise TrainerError(
                "prediction_validation", "FORWARD_COMPLETED",
                epoch=epoch, batch=batch_index,
                subsystem="fusion_model",
                received=f"pred.device={predictions.device}, target.device={targets.device}",
                expected="same device",
                resolution="Prediction and target device mismatch.",
            )

        pred_B = predictions.shape[0]
        tgt_B = targets.shape[0]
        if pred_B != tgt_B:
            raise TrainerError(
                "prediction_validation", "FORWARD_COMPLETED",
                epoch=epoch, batch=batch_index,
                subsystem="fusion_model",
                received=f"pred.shape[0]={pred_B}, target.shape[0]={tgt_B}",
                expected="matching batch sizes",
                resolution="FusionModel output batch size mismatch.",
            )

        # Shape must be [B] or [B,1]
        if predictions.dim() > 2 or (predictions.dim() == 2 and predictions.shape[1] != 1):
            raise TrainerError(
                "prediction_validation", "FORWARD_COMPLETED",
                epoch=epoch, batch=batch_index,
                subsystem="fusion_model",
                received=f"pred.shape={tuple(predictions.shape)}",
                expected="[B] or [B,1]",
                resolution="FusionModel output has unexpected shape.",
            )

        # Reject bool and complex dtypes
        if predictions.dtype == torch.bool:
            raise TrainerError(
                "prediction_validation", "FORWARD_COMPLETED",
                epoch=epoch, batch=batch_index,
                subsystem="fusion_model",
                received=f"pred.dtype={predictions.dtype}",
                expected="numeric non-bool dtype",
                resolution="FusionModel returned boolean predictions.",
            )
        if predictions.dtype.is_complex:
            raise TrainerError(
                "prediction_validation", "FORWARD_COMPLETED",
                epoch=epoch, batch=batch_index,
                subsystem="fusion_model",
                received=f"pred.dtype={predictions.dtype}",
                expected="real-valued dtype",
                resolution="FusionModel returned complex predictions.",
            )

        # Finite check
        if not torch.isfinite(predictions).all():
            raise TrainerError(
                "prediction_validation", "FORWARD_COMPLETED",
                epoch=epoch, batch=batch_index,
                subsystem="fusion_model",
                received="predictions contain NaN or Inf",
                expected="all finite values",
                resolution="Check model output for numerical instability.",
            )

    def _validate_model_device(self) -> None:
        """Validate every model parameter and buffer is on the expected device.

        Stops immediately on first mismatch for deterministic errors.
        """
        expected_device = torch.device(self._run_context.device)
        # Check all parameters
        for name, param in self._model_bundle.named_parameters():
            if param.device != expected_device:
                raise TrainerError(
                    "device_validation", "model_device_check",
                    subsystem="model_bundle",
                    received=f"param '{name}' on {param.device}",
                    expected=f"all params on {expected_device}",
                    resolution="Call model.to(device) before training.",
                )
        # Check all buffers
        for name, buf in self._model_bundle.named_buffers():
            if buf.device != expected_device:
                raise TrainerError(
                    "device_validation", "model_device_check",
                    subsystem="model_bundle",
                    received=f"buffer '{name}' on {buf.device}",
                    expected=f"all buffers on {expected_device}",
                    resolution="Call model.to(device) before training.",
                )

    def _validate_batch_device(
        self, batch: Dict[str, Any], epoch: int, batch_index: int,
    ) -> None:
        """Verify batch tensors are on the expected device after transfer."""
        expected_device = torch.device(self._run_context.device)
        for key in _TENSOR_BATCH_KEYS:
            val = batch.get(key)
            if isinstance(val, torch.Tensor) and val.device != expected_device:
                raise TrainerError(
                    "device_validation", "BATCH_TRANSFERRED",
                    epoch=epoch, batch=batch_index,
                    subsystem="data_pipeline",
                    received=f"batch['{key}'] on {val.device}",
                    expected=f"all tensors on {expected_device}",
                    resolution="Batch transfer to device failed.",
                )

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move tensor fields to the target device. Returns a new dict.

        Wraps per-tensor transfer errors with actionable TrainerError.
        CUDA OOM during transfer produces a specific resolution message.
        """
        device = torch.device(self._run_context.device)
        non_blocking = (self._run_context.device_type == "cuda")

        moved = {}
        for key, val in batch.items():
            if key in _TENSOR_BATCH_KEYS and isinstance(val, torch.Tensor):
                try:
                    moved[key] = val.to(device, non_blocking=non_blocking)
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        raise TrainerError(
                            "batch_transfer", "BATCH_TRANSFERRED",
                            subsystem="cuda",
                            received=f"key='{key}', shape={tuple(val.shape)}, dtype={val.dtype}",
                            expected=f"successful transfer to {device}",
                            resolution=(
                                "Reduce batch_size, enable AMP, reduce image resolution, "
                                "reduce dataloader workers, or restart runtime to clear "
                                "GPU memory fragmentation."
                            ),
                        ) from exc
                    raise TrainerError(
                        "batch_transfer", "BATCH_TRANSFERRED",
                        subsystem="data_pipeline",
                        received=f"key='{key}', shape={tuple(val.shape)}, dtype={val.dtype}, error={str(exc)[:150]}",
                        expected=f"successful transfer to {device}",
                        resolution="Check tensor compatibility with target device.",
                    ) from exc
            else:
                moved[key] = val  # metadata stays on CPU
        return moved

    def _forward_batch(
        self, batch: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """Run the model bundle forward pass. Returns model output dict."""
        images = batch["images"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        tabular = batch["tabular"]

        try:
            image_embedding = self._model_bundle["image_encoder"](images)
        except Exception as exc:
            raise TrainerError(
                "forward", "FORWARD_STARTED",
                subsystem="image_encoder",
                received=str(exc)[:200],
                expected="valid image embedding",
                resolution="Check ImageEncoder input contract.",
            ) from exc

        try:
            text_embedding = self._model_bundle["text_encoder"](
                input_ids, attention_mask,
            )
        except Exception as exc:
            raise TrainerError(
                "forward", "FORWARD_STARTED",
                subsystem="text_encoder",
                received=str(exc)[:200],
                expected="valid text embedding",
                resolution="Check TextEncoder input contract.",
            ) from exc

        try:
            tabular_embedding = self._model_bundle["tabular_encoder"](tabular)
        except Exception as exc:
            raise TrainerError(
                "forward", "FORWARD_STARTED",
                subsystem="tabular_encoder",
                received=str(exc)[:200],
                expected="valid tabular embedding",
                resolution="Check TabularEncoder input contract.",
            ) from exc

        try:
            model_output = self._model_bundle["fusion_model"](
                image_embedding, text_embedding, tabular_embedding,
            )
        except Exception as exc:
            raise TrainerError(
                "forward", "FORWARD_STARTED",
                subsystem="fusion_model",
                received=str(exc)[:200],
                expected="valid fusion output dict",
                resolution="Check FusionModel forward contract.",
            ) from exc

        # Verify output contract
        if not isinstance(model_output, dict):
            raise TrainerError(
                "forward", "FORWARD_COMPLETED",
                subsystem="fusion_model",
                received=type(model_output).__name__,
                expected="dict with 'rating_prediction'",
                resolution="FusionModel must return a dict.",
            )
        if "rating_prediction" not in model_output:
            raise TrainerError(
                "forward", "FORWARD_COMPLETED",
                subsystem="fusion_model",
                received=f"keys {sorted(model_output.keys())}",
                expected="key 'rating_prediction'",
                resolution="FusionModel output missing 'rating_prediction'.",
            )

        return model_output

    def _train_one_batch(
        self,
        batch: Dict[str, Any],
        batch_index: int,
        epoch: int,
        is_accumulation_boundary: bool,
    ) -> float:
        """Process one training batch. Returns the raw loss value (float)."""
        self._emit_event(_TrainingEvent.BATCH_STARTED, epoch=epoch, batch=batch_index)
        self._runtime.current_batch = batch_index

        # Validate batch
        self._validate_batch(batch, batch_index, epoch)

        # Move to device
        batch = self._move_batch_to_device(batch)
        self._validate_batch_device(batch, epoch, batch_index)
        self._emit_event(_TrainingEvent.BATCH_TRANSFERRED, epoch=epoch, batch=batch_index)

        # Forward pass
        self._emit_event(_TrainingEvent.FORWARD_STARTED, epoch=epoch, batch=batch_index)

        targets = batch["ratings"]

        try:
            if self._amp_enabled:
                try:
                    amp_ctx = torch.amp.autocast(device_type=self._amp_device_type)
                except (TypeError, AttributeError):
                    amp_ctx = torch.cuda.amp.autocast()
                with amp_ctx:
                    model_output = self._forward_batch(batch)
                    predictions = extract_prediction(model_output)
                    self._validate_prediction_contract(predictions, targets, epoch, batch_index)
                    loss = compute_loss(predictions, targets, self._config)
            else:
                model_output = self._forward_batch(batch)
                predictions = extract_prediction(model_output)
                self._validate_prediction_contract(predictions, targets, epoch, batch_index)
                loss = compute_loss(predictions, targets, self._config)
        except TrainerError:
            raise
        except RuntimeError as exc:
            # Catch CUDA OOM and wrap as actionable TrainerError
            if "out of memory" in str(exc).lower():
                raise TrainerError(
                    "forward", "FORWARD_STARTED",
                    epoch=epoch, batch=batch_index,
                    subsystem="cuda",
                    received=str(exc)[:200],
                    expected="sufficient GPU memory",
                    resolution="Reduce batch_size or enable gradient_accumulation_steps.",
                ) from exc
            raise

        self._emit_event(_TrainingEvent.FORWARD_COMPLETED, epoch=epoch, batch=batch_index)

        # Loss safety checks
        self._validate_loss(loss, epoch, batch_index)
        self._emit_event(_TrainingEvent.LOSS_COMPUTED, epoch=epoch, batch=batch_index)

        # Scale by gradient accumulation
        accum_steps = self._config.gradient_accumulation_steps
        scaled_loss = loss / accum_steps

        # Backward
        try:
            if self._amp_enabled and self._scaler is not None:
                self._scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise TrainerError(
                    "backward", "BACKWARD_COMPLETED",
                    epoch=epoch, batch=batch_index,
                    subsystem="cuda",
                    received=str(exc)[:200],
                    expected="sufficient GPU memory for backward pass",
                    resolution="Reduce batch_size or enable gradient_accumulation_steps.",
                ) from exc
            raise

        self._emit_event(_TrainingEvent.BACKWARD_COMPLETED, epoch=epoch, batch=batch_index)

        # Optimizer step on accumulation boundary
        if is_accumulation_boundary:
            self._do_optimizer_step(epoch, batch_index)

        # Increment global step (per batch, not per optimizer step)
        self._runtime.global_step += 1

        return loss.detach().item()

    def _validate_loss(self, loss: torch.Tensor, epoch: int, batch: int) -> None:
        """Validate loss tensor integrity."""
        if loss.dim() != 0:
            raise TrainerError(
                "loss_validation", "LOSS_COMPUTED",
                epoch=epoch, batch=batch,
                subsystem="evaluation",
                received=f"loss.dim()={loss.dim()}, shape={tuple(loss.shape)}",
                expected="scalar loss (dim=0)",
                resolution="compute_loss must return a scalar.",
            )
        loss_val = loss.item()
        if not torch.isfinite(loss).item():
            raise TrainerError(
                "loss_validation", "LOSS_COMPUTED",
                epoch=epoch, batch=batch,
                subsystem="evaluation",
                received=f"loss={loss_val} (NaN or Inf)",
                expected="finite loss value",
                resolution="Check model output and targets for NaN/Inf.",
            )
        if not loss.requires_grad:
            raise TrainerError(
                "loss_validation", "LOSS_COMPUTED",
                epoch=epoch, batch=batch,
                subsystem="evaluation",
                received="loss.requires_grad=False",
                expected="loss.requires_grad=True (during training)",
                resolution="Ensure model is in train mode and inputs require grad.",
            )

    def _do_optimizer_step(self, epoch: int, batch_index: int) -> None:
        """Execute optimizer step with optional AMP unscale and gradient clipping."""
        try:
            if self._amp_enabled and self._scaler is not None:
                self._scaler.unscale_(self._optimizer)

            # Gradient clipping
            if self._config.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self._model_bundle.parameters(),
                    self._config.gradient_clip,
                )
                self._emit_event(
                    _TrainingEvent.GRADIENT_CLIPPED,
                    epoch=epoch, batch=batch_index,
                )

            if self._amp_enabled and self._scaler is not None:
                self._scaler.step(self._optimizer)
                self._scaler.update()
            else:
                self._optimizer.step()

            self._optimizer.zero_grad(set_to_none=True)
            self._runtime.optimizer_steps += 1
            self._emit_event(
                _TrainingEvent.OPTIMIZER_UPDATED,
                epoch=epoch, batch=batch_index,
            )

        except Exception as exc:
            raise TrainerError(
                "optimizer_step", "OPTIMIZER_UPDATED",
                epoch=epoch, batch=batch_index,
                subsystem="optimizer",
                received=str(exc)[:200],
                expected="successful optimizer step",
                resolution="Check optimizer and gradient state.",
            ) from exc

    # =========================================================================
    # Internal -- Epoch Handling
    # =========================================================================

    def _train_one_epoch(self, epoch: int) -> float:
        """Train for one full epoch. Returns average training loss."""
        self._model_bundle.train()
        self._optimizer.zero_grad(set_to_none=True)

        # Spot-check model device at epoch start
        self._validate_model_device()

        accum_steps = self._config.gradient_accumulation_steps
        total_loss = 0.0
        batch_count = 0

        for batch_index, batch in enumerate(self._train_loader, start=1):
            is_accumulation_boundary = (batch_index % accum_steps == 0)

            loss_val = self._train_one_batch(
                batch, batch_index, epoch, is_accumulation_boundary,
            )
            total_loss += loss_val
            batch_count += 1

        # Handle remainder: if the last batch wasn't an accumulation boundary,
        # do a final optimizer step
        if batch_count > 0 and (batch_count % accum_steps != 0):
            self._do_optimizer_step(epoch, batch_count)

        if batch_count == 0:
            raise TrainerError(
                "epoch", "EPOCH_STARTED",
                epoch=epoch,
                subsystem="train_loader",
                received="0 batches",
                expected="at least 1 batch",
                resolution="Train loader is empty.",
            )

        avg_loss = total_loss / batch_count
        return avg_loss

    def _validate_one_epoch(self, epoch: int) -> EvaluationResult:
        """Run validation for one epoch. Returns epoch-level EvaluationResult.

        Model mode is saved before and restored in a finally block to
        guarantee train mode is not lost after a validation failure.
        """
        self._set_status(_TrainerStatus.VALIDATING)
        self._emit_event(_TrainingEvent.VALIDATION_STARTED, epoch=epoch)

        was_training = self._model_bundle.training
        self._model_bundle.eval()

        all_predictions: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []
        val_batch_count = 0

        try:
            with torch.no_grad():
                for val_batch_index, batch in enumerate(self._val_loader, start=1):
                    self._validate_batch(batch, val_batch_index, epoch)
                    batch = self._move_batch_to_device(batch)
                    self._validate_batch_device(batch, epoch, val_batch_index)

                    targets = batch["ratings"]

                    if self._amp_enabled:
                        try:
                            amp_ctx = torch.amp.autocast(device_type=self._amp_device_type)
                        except (TypeError, AttributeError):
                            amp_ctx = torch.cuda.amp.autocast()
                        with amp_ctx:
                            model_output = self._forward_batch(batch)
                    else:
                        model_output = self._forward_batch(batch)

                    predictions = extract_prediction(model_output)

                    # Validate prediction contract BEFORE flatten/detach
                    self._validate_prediction_contract(
                        predictions, targets, epoch, val_batch_index,
                    )

                    # Flatten to 1D and detach to CPU
                    preds_1d = predictions.detach().cpu().view(-1)
                    tgts_1d = targets.detach().cpu().view(-1)

                    all_predictions.append(preds_1d)
                    all_targets.append(tgts_1d)
                    val_batch_count += 1

                    self._emit_event(
                        _TrainingEvent.VALIDATION_BATCH_COMPLETED,
                        epoch=epoch, batch=val_batch_index,
                    )

            if val_batch_count == 0:
                raise TrainerError(
                    "validation", "VALIDATION_STARTED",
                    epoch=epoch,
                    subsystem="val_loader",
                    received="0 batches",
                    expected="at least 1 batch",
                    resolution="Validation loader is empty.",
                )

            # Concatenate all predictions/targets for epoch-level evaluation
            epoch_preds = torch.cat(all_predictions, dim=0)
            epoch_targets = torch.cat(all_targets, dim=0)

            # Epoch-level evaluation
            val_result = self._evaluator.evaluate(
                epoch_preds, epoch_targets,
                split="validation", epoch=epoch,
            )

            self._emit_event(_TrainingEvent.VALIDATION_COMPLETED, epoch=epoch)

        finally:
            # Restore previous model mode unconditionally
            if was_training:
                self._model_bundle.train()
            else:
                self._model_bundle.eval()

        self._set_status(_TrainerStatus.RUNNING)

        return val_result

    def _update_history(
        self,
        epoch: int,
        train_loss: float,
        validation_result: Optional[EvaluationResult],
        epoch_duration: float,
        checkpoint_saved: bool,
    ) -> None:
        """Append one epoch's results to history. Validates length consistency."""
        # Prevent duplicate epoch entries
        if epoch in self._history.epoch:
            raise TrainerError(
                "history", "EPOCH_COMPLETED",
                epoch=epoch,
                received=f"duplicate epoch {epoch}",
                expected="unique epoch entries",
                resolution="History epoch duplication detected.",
            )

        # Current learning rate
        lr = self._optimizer.param_groups[0]["lr"]

        self._history.epoch.append(epoch)
        self._history.train_loss.append(train_loss)

        if validation_result is not None:
            self._history.validation_loss.append(validation_result.loss)
            self._history.validation_rmse.append(validation_result.rmse)
            self._history.validation_mae.append(validation_result.mae)
            self._history.validation_r2.append(validation_result.r2)
        else:
            self._history.validation_loss.append(float("nan"))
            self._history.validation_rmse.append(float("nan"))
            self._history.validation_mae.append(float("nan"))
            self._history.validation_r2.append(float("nan"))

        self._history.learning_rate.append(lr)
        self._history.epoch_duration_seconds.append(epoch_duration)
        self._history.checkpoint_saved.append(checkpoint_saved)

        # Update best tracking
        if (validation_result is not None
                and self._evaluator.state.best_validation_epoch is not None):
            self._history.best_epoch = self._evaluator.state.best_validation_epoch
            self._history.best_validation_loss = self._evaluator.state.best_validation_loss

        # Validate history consistency
        if not self._history._check_lengths():
            raise TrainerError(
                "history", "EPOCH_COMPLETED",
                epoch=epoch,
                received="mismatched history list lengths",
                expected="synchronized history arrays",
                resolution="Internal history corruption.",
            )

    def _check_epoch_invariants(self, epoch: int) -> None:
        """Verify runtime invariants before each epoch.

        Checks are ordered deterministically by dependency:
        config -> run_context -> model -> optimizer -> scheduler -> evaluator
        -> checkpoint -> device -> runtime state.

        Stops on first failure for deterministic error reporting.
        """
        # 1. Config must remain frozen
        if not self._config.is_frozen:
            raise TrainerError(
                "invariant", "EPOCH_STARTED",
                epoch=epoch,
                received="config.is_frozen=False",
                expected="config.is_frozen=True (immutable during training)",
                resolution="Config was unfrozen during training.",
            )

        # 2. Run context must still reference our config
        if self._run_context.config is not self._config:
            raise TrainerError(
                "invariant", "EPOCH_STARTED",
                epoch=epoch,
                received="run_context.config is not self._config",
                expected="run_context.config is self._config",
                resolution="RunContext was swapped during training.",
            )

        # 3. Model bundle still has required keys
        model_keys = set(self._model_bundle.keys()) if hasattr(self._model_bundle, 'keys') else set()
        missing_model = _REQUIRED_MODEL_KEYS - model_keys
        if missing_model:
            raise TrainerError(
                "invariant", "EPOCH_STARTED",
                epoch=epoch,
                received=f"model_bundle missing keys: {sorted(missing_model)}",
                expected=f"all required keys: {sorted(_REQUIRED_MODEL_KEYS)}",
                resolution="Model bundle was mutated during training.",
            )

        # 4. Optimizer is still valid
        if not isinstance(self._optimizer, torch.optim.Optimizer):
            raise TrainerError(
                "invariant", "EPOCH_STARTED",
                epoch=epoch,
                received=type(self._optimizer).__name__,
                expected="torch.optim.Optimizer",
                resolution="Optimizer was replaced during training.",
            )

        # 5. Optimizer-model parameter integrity
        self._validate_optimizer_model_integrity(epoch)

        # 6. Scheduler metadata still exists and is valid
        if self._scheduler_meta is None:
            raise TrainerError(
                "invariant", "EPOCH_STARTED",
                epoch=epoch,
                received="scheduler_meta is None",
                expected="valid SchedulerMetadata",
                resolution="Scheduler metadata was detached during training.",
            )
        if self._scheduler_meta.step_policy not in (STEP_POLICY_EPOCH, STEP_POLICY_VALIDATION_METRIC):
            raise TrainerError(
                "invariant", "EPOCH_STARTED",
                epoch=epoch,
                received=f"step_policy='{self._scheduler_meta.step_policy}'",
                expected=f"'{STEP_POLICY_EPOCH}' or '{STEP_POLICY_VALIDATION_METRIC}'",
                resolution="Scheduler metadata has invalid step_policy.",
            )

        # 7. Evaluator is still valid
        if not isinstance(self._evaluator, Evaluator):
            raise TrainerError(
                "invariant", "EPOCH_STARTED",
                epoch=epoch,
                received=type(self._evaluator).__name__,
                expected="Evaluator",
                resolution="Evaluator was replaced during training.",
            )

        # 8. Checkpoint directory safety
        _assert_child_path(
            self._checkpoint_dir.parent, self._checkpoint_dir, "invariant",
        )

        # 9. Epoch monotonicity
        if self._history.epoch and epoch <= self._history.epoch[-1]:
            raise TrainerError(
                "invariant", "EPOCH_STARTED",
                epoch=epoch,
                received=f"epoch {epoch} <= last recorded {self._history.epoch[-1]}",
                expected="strictly increasing epoch numbers",
                resolution="Epoch ordering corruption detected.",
            )

        # 10. Optimizer steps never exceed global steps
        if self._runtime.optimizer_steps > self._runtime.global_step:
            raise TrainerError(
                "invariant", "EPOCH_STARTED",
                epoch=epoch,
                received=f"optimizer_steps={self._runtime.optimizer_steps} > global_step={self._runtime.global_step}",
                expected="optimizer_steps <= global_step",
                resolution="Internal step counting corruption.",
            )

        # 11. History lengths synchronized
        if not self._history._check_lengths():
            raise TrainerError(
                "invariant", "EPOCH_STARTED",
                epoch=epoch,
                received="mismatched history list lengths",
                expected="synchronized history arrays",
                resolution="Internal history corruption detected.",
            )

        # 12. Model device validation (all params + buffers)
        self._validate_model_device()

    def _validate_optimizer_model_integrity(self, epoch: int) -> None:
        """Verify optimizer-model parameter ownership.

        Uses object identity (id(param)) -- never tensor equality.
        Stops on first category of mismatch for deterministic errors.

        Checks:
        - Every trainable model param appears in exactly one optimizer group.
        - No stale params in optimizer that are not in the model.
        - No duplicate params across optimizer groups.
        """
        # Collect active trainable model param ids
        model_param_ids = {
            id(p) for p in self._model_bundle.parameters() if p.requires_grad
        }

        # Collect optimizer param ids (with duplicate detection)
        optimizer_param_ids = set()
        duplicate_count = 0
        for group in self._optimizer.param_groups:
            for p in group["params"]:
                pid = id(p)
                if pid in optimizer_param_ids:
                    duplicate_count += 1
                optimizer_param_ids.add(pid)

        # Duplicate optimizer parameter ownership
        if duplicate_count > 0:
            raise TrainerError(
                "invariant", "EPOCH_STARTED",
                epoch=epoch,
                subsystem="optimizer",
                received=f"{duplicate_count} duplicate parameter(s) across optimizer groups",
                expected="every parameter appears in exactly one optimizer group",
                resolution="Rebuild optimizer after modifying model_bundle.",
            )

        # Missing trainable model params from optimizer
        missing = model_param_ids - optimizer_param_ids
        if missing:
            raise TrainerError(
                "invariant", "EPOCH_STARTED",
                epoch=epoch,
                subsystem="optimizer",
                received=f"{len(missing)} trainable model param(s) missing from optimizer",
                expected="every trainable model parameter appears in exactly one optimizer group",
                resolution="Rebuild optimizer after modifying model_bundle.",
            )

        # Stale optimizer params not in model
        stale = optimizer_param_ids - model_param_ids
        # Filter: stale params may be frozen params intentionally included
        # Only flag params whose id is not in ANY model parameter (trainable or frozen)
        all_model_param_ids = {id(p) for p in self._model_bundle.parameters()}
        truly_stale = stale - all_model_param_ids
        if truly_stale:
            raise TrainerError(
                "invariant", "EPOCH_STARTED",
                epoch=epoch,
                subsystem="optimizer",
                received=f"{len(truly_stale)} optimizer param(s) not in model_bundle",
                expected="all optimizer parameters belong to the active model",
                resolution="Rebuild optimizer after modifying model_bundle.",
            )

    # =========================================================================
    # Internal -- Scheduler
    # =========================================================================

    def _step_scheduler_after_epoch(
        self, validation_result: Optional[EvaluationResult],
    ) -> None:
        """Step the scheduler according to its metadata policy."""
        policy = self._scheduler_meta.step_policy

        if policy == STEP_POLICY_EPOCH:
            self._scheduler.step()
            self._emit_event(_TrainingEvent.SCHEDULER_UPDATED)

        elif policy == STEP_POLICY_VALIDATION_METRIC:
            if validation_result is None:
                # Plateau scheduler but no validation this epoch -- skip
                return
            self._scheduler.step(validation_result.loss)
            self._emit_event(_TrainingEvent.SCHEDULER_UPDATED)

        else:
            raise TrainerError(
                "scheduler", "SCHEDULER_UPDATED",
                subsystem="scheduler",
                received=f"step_policy='{policy}'",
                expected=f"'{STEP_POLICY_EPOCH}' or '{STEP_POLICY_VALIDATION_METRIC}'",
                resolution="Unknown scheduler step policy.",
            )

    # =========================================================================
    # Internal -- Checkpoint
    # =========================================================================

    def _should_save_latest(self, epoch: int) -> bool:
        """Determine if a latest checkpoint should be saved."""
        if not self._config.save_latest:
            return False
        if self._config.checkpoint_frequency <= 0:
            return False
        return epoch % self._config.checkpoint_frequency == 0

    def _should_save_best(self, is_best: bool) -> bool:
        """Determine if a best checkpoint should be saved."""
        return self._config.save_best and is_best

    def _get_checkpoint_path(self, kind: str, epoch: int) -> Path:
        """Build checkpoint file path. Validates path safety with relative_to."""
        if kind == "latest":
            filename = "latest.pt"
        elif kind == "best":
            filename = "best.pt"
        elif kind == "epoch":
            filename = f"epoch_{epoch:04d}.pt"
        elif kind == "interrupted":
            filename = f"interrupted_epoch_{epoch:04d}.pt"
        else:
            filename = f"{kind}_epoch_{epoch:04d}.pt"

        ckpt_path = self._checkpoint_dir / filename
        resolved = _assert_child_path(self._checkpoint_dir, ckpt_path, "checkpoint")
        return resolved

    def _save_checkpoint(self, kind: str, epoch: int) -> None:
        """Save a checkpoint atomically.

        Flow:
        1. Resolve safe checkpoint path.
        2. Build next_checkpoint_state before payload.
        3. Build checkpoint payload with next state.
        4. Save atomically (temp file -> rename).
        5. Only after success: update in-memory state, emit CHECKPOINT_SAVED.
        """
        self._set_status(_TrainerStatus.CHECKPOINTING)

        ckpt_path = self._get_checkpoint_path(kind, epoch)

        # Ensure directory exists
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        # Build next checkpoint state BEFORE payload creation
        next_cs = _CheckpointState(
            latest_checkpoint=(
                str(ckpt_path) if kind == "latest"
                else self._checkpoint_state.latest_checkpoint
            ),
            best_checkpoint=(
                str(ckpt_path) if kind == "best"
                else self._checkpoint_state.best_checkpoint
            ),
            last_saved_epoch=epoch,
            best_epoch=(
                epoch if kind == "best"
                else self._checkpoint_state.best_epoch
            ),
            checkpoint_count=self._checkpoint_state.checkpoint_count + 1,
            resume_checkpoint=self._checkpoint_state.resume_checkpoint,
            resume_epoch=self._checkpoint_state.resume_epoch,
        )

        # Scheduler state dict -- strip custom metadata to avoid pickle errors
        sched_sd = None
        if hasattr(self._scheduler, "state_dict"):
            sched_sd = dict(self._scheduler.state_dict())
            sched_sd.pop("_sched_metadata", None)

        scaler_sd = None
        if self._scaler is not None:
            scaler_sd = self._scaler.state_dict()

        # Build checkpoint payload with CURRENT history and NEXT checkpoint state
        checkpoint = {
            "checkpoint_version": _CHECKPOINT_VERSION,
            "trainer_schema_version": _TRAINER_SCHEMA_VERSION,
            "trainer_class": "Trainer",
            "kind": kind,
            "epoch": epoch,
            "global_step": self._runtime.global_step,
            "model_state_dict": self._model_bundle.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "scheduler_state_dict": sched_sd,
            "amp_scaler_state_dict": scaler_sd,
            "config": self._config.as_dict(),
            "run_context": self._run_context.as_dict(),
            "trainer_runtime": self.runtime_state(),
            "training_history": self.history(),
            "checkpoint_state": next_cs.as_dict(),
            "evaluation": self._evaluator.as_dict(),
            "reproducibility": self._repro.as_dict() if self._repro else None,
        }

        # Atomic save: write to temp file, then replace
        tmp_path = None
        try:
            # Ensure all MappingProxyType objects are converted to dicts
            from types import MappingProxyType as _MPT

            def _make_picklable(obj):
                if isinstance(obj, _MPT):
                    return {k: _make_picklable(v) for k, v in obj.items()}
                elif isinstance(obj, dict):
                    return {k: _make_picklable(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    converted = [_make_picklable(v) for v in obj]
                    return type(obj)(converted) if isinstance(obj, tuple) else converted
                return obj

            checkpoint = _make_picklable(checkpoint)

            fd, tmp_path = tempfile.mkstemp(
                dir=str(ckpt_path.parent),
                suffix=".pt.tmp",
            )
            os.close(fd)
            torch.save(checkpoint, tmp_path)

            # Atomic replace
            tmp_path_obj = Path(tmp_path)
            if ckpt_path.exists():
                ckpt_path.unlink()
            tmp_path_obj.rename(ckpt_path)

        except Exception as exc:
            # Clean up temp file on failure -- guard against tmp_path being None
            if tmp_path is not None:
                try:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                except OSError:
                    pass
            raise TrainerError(
                "checkpoint", "CHECKPOINT_SAVED",
                epoch=epoch,
                subsystem="checkpoint",
                received=str(exc)[:200],
                expected="successful checkpoint save",
                resolution="Check disk space and permissions.",
            ) from exc

        # Success: update in-memory state and emit event
        self._checkpoint_state = next_cs
        self._emit_event(_TrainingEvent.CHECKPOINT_SAVED, epoch=epoch)

        # Return to RUNNING (unless training is about to end)
        if _TrainerStatus(self._runtime.status) == _TrainerStatus.CHECKPOINTING:
            self._set_status(_TrainerStatus.RUNNING)

    def _load_checkpoint(self, path: Path) -> Dict[str, Any]:
        """Load a checkpoint file. Validates structure and schema version."""
        if not path.exists():
            raise TrainerError(
                "resume", "RESUME_STARTED",
                received=str(path),
                expected="existing checkpoint file",
                resolution="Checkpoint file not found.",
            )

        try:
            device = torch.device(self._run_context.device)
            checkpoint = torch.load(str(path), map_location=device, weights_only=False)
        except Exception as exc:
            raise TrainerError(
                "resume", "RESUME_STARTED",
                received=str(exc)[:200],
                expected="loadable checkpoint",
                resolution="Checkpoint file is corrupt or incompatible.",
            ) from exc

        if not isinstance(checkpoint, dict):
            raise TrainerError(
                "resume", "RESUME_STARTED",
                received=type(checkpoint).__name__,
                expected="dict",
                resolution="Checkpoint is not a valid dict.",
            )

        missing = _CHECKPOINT_REQUIRED_KEYS - set(checkpoint.keys())
        if missing:
            raise TrainerError(
                "resume", "RESUME_STARTED",
                received=f"keys {sorted(checkpoint.keys())}",
                expected=f"required keys {sorted(_CHECKPOINT_REQUIRED_KEYS)}",
                resolution=f"Missing checkpoint keys: {sorted(missing)}",
            )

        # Schema version validation
        schema_ver = checkpoint.get("trainer_schema_version")
        if schema_ver is not None and schema_ver != _TRAINER_SCHEMA_VERSION:
            raise TrainerError(
                "resume", "schema_validation",
                received=f"trainer_schema_version={schema_ver}",
                expected=f"trainer_schema_version={_TRAINER_SCHEMA_VERSION}",
                resolution="Checkpoint was created by an incompatible trainer version.",
            )

        return checkpoint

    def _validate_checkpoint_restore_contract(
        self, checkpoint: Dict[str, Any],
    ) -> None:
        """Phase 1 of two-phase resume: validate without mutating state.

        Checks:
        - epoch is an integer > 0
        - global_step is a non-negative integer
        - model_state_dict is a dict
        - optimizer_state_dict is a dict
        - training_history has consistent list lengths
        - checkpoint_version matches
        """
        epoch = checkpoint.get("epoch")
        if not isinstance(epoch, int) or epoch < 1:
            raise TrainerError(
                "resume", "contract_validation",
                received=f"epoch={epoch!r}",
                expected="integer >= 1",
                resolution="Checkpoint epoch is invalid.",
            )

        gs = checkpoint.get("global_step")
        if not isinstance(gs, int) or gs < 0:
            raise TrainerError(
                "resume", "contract_validation",
                received=f"global_step={gs!r}",
                expected="non-negative integer",
                resolution="Checkpoint global_step is invalid.",
            )

        if not isinstance(checkpoint.get("model_state_dict"), dict):
            raise TrainerError(
                "resume", "contract_validation",
                received=type(checkpoint.get("model_state_dict")).__name__,
                expected="dict",
                resolution="Checkpoint model_state_dict is not a dict.",
            )

        if not isinstance(checkpoint.get("optimizer_state_dict"), dict):
            raise TrainerError(
                "resume", "contract_validation",
                received=type(checkpoint.get("optimizer_state_dict")).__name__,
                expected="dict",
                resolution="Checkpoint optimizer_state_dict is not a dict.",
            )

        # History list lengths consistent
        hist = checkpoint.get("training_history", {})
        if isinstance(hist, dict):
            list_keys = [
                "epoch", "train_loss", "validation_loss",
                "validation_rmse", "validation_mae", "validation_r2",
                "learning_rate", "epoch_duration_seconds", "checkpoint_saved",
            ]
            lengths = [len(hist.get(k, [])) for k in list_keys]
            if len(set(lengths)) > 1:
                raise TrainerError(
                    "resume", "contract_validation",
                    received=f"history lengths={dict(zip(list_keys, lengths))}",
                    expected="all list fields same length",
                    resolution="Checkpoint history is inconsistent.",
                )

        # Checkpoint version
        cv = checkpoint.get("checkpoint_version")
        if cv != _CHECKPOINT_VERSION:
            raise TrainerError(
                "resume", "contract_validation",
                received=f"checkpoint_version={cv}",
                expected=f"checkpoint_version={_CHECKPOINT_VERSION}",
                resolution="Checkpoint version mismatch.",
            )

    def _restore_checkpoint_state(self, checkpoint: Dict[str, Any]) -> None:
        """Phase 2 of two-phase resume: restore training state from checkpoint."""
        try:
            self._model_bundle.load_state_dict(checkpoint["model_state_dict"])
        except Exception as exc:
            raise TrainerError(
                "resume", "model_restore",
                received=str(exc)[:200],
                expected="compatible model state",
                resolution="Model architecture does not match checkpoint.",
            ) from exc

        try:
            self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception as exc:
            raise TrainerError(
                "resume", "optimizer_restore",
                received=str(exc)[:200],
                expected="compatible optimizer state",
                resolution="Optimizer does not match checkpoint.",
            ) from exc

        # Scheduler
        sched_state = checkpoint.get("scheduler_state_dict")
        if sched_state is not None and hasattr(self._scheduler, "load_state_dict"):
            try:
                self._scheduler.load_state_dict(sched_state)
            except Exception as exc:
                raise TrainerError(
                    "resume", "scheduler_restore",
                    received=str(exc)[:200],
                    expected="compatible scheduler state",
                    resolution="Scheduler does not match checkpoint.",
                ) from exc

        # AMP scaler
        scaler_state = checkpoint.get("amp_scaler_state_dict")
        if scaler_state is not None and self._scaler is not None:
            try:
                self._scaler.load_state_dict(scaler_state)
            except Exception:
                self._mark_warning("AMP scaler restore failed, using fresh scaler.")

        # Runtime state
        rt = checkpoint.get("trainer_runtime", {})
        self._runtime.global_step = rt.get("global_step", 0)
        self._runtime.optimizer_steps = rt.get("optimizer_steps", 0)

        # History
        hist = checkpoint.get("training_history", {})
        self._history.epoch = list(hist.get("epoch", []))
        self._history.train_loss = list(hist.get("train_loss", []))
        self._history.validation_loss = list(hist.get("validation_loss", []))
        self._history.validation_rmse = list(hist.get("validation_rmse", []))
        self._history.validation_mae = list(hist.get("validation_mae", []))
        self._history.validation_r2 = list(hist.get("validation_r2", []))
        self._history.learning_rate = list(hist.get("learning_rate", []))
        self._history.epoch_duration_seconds = list(hist.get("epoch_duration_seconds", []))
        self._history.checkpoint_saved = list(hist.get("checkpoint_saved", []))
        self._history.best_epoch = hist.get("best_epoch")
        self._history.best_validation_loss = hist.get("best_validation_loss")

        # Checkpoint state
        cs = checkpoint.get("checkpoint_state", {})
        self._checkpoint_state.latest_checkpoint = cs.get("latest_checkpoint")
        self._checkpoint_state.best_checkpoint = cs.get("best_checkpoint")
        self._checkpoint_state.last_saved_epoch = cs.get("last_saved_epoch")
        self._checkpoint_state.best_epoch = cs.get("best_epoch")
        self._checkpoint_state.checkpoint_count = cs.get("checkpoint_count", 0)

        # Evaluator state restore
        self._restore_evaluator_state(checkpoint)

        # Start epoch
        self._start_epoch = checkpoint["epoch"] + 1
        self._checkpoint_state.resume_checkpoint = str(checkpoint.get("kind", "unknown"))
        self._checkpoint_state.resume_epoch = checkpoint["epoch"]

    def _restore_evaluator_state(self, checkpoint: Dict[str, Any]) -> None:
        """Restore evaluator best-tracking state from checkpoint.

        For current-schema checkpoints, malformed evaluation payloads are fatal.
        Sets fields directly on the mutable EvaluationRuntimeState dataclass.
        """
        eval_payload = checkpoint.get("evaluation")
        if not isinstance(eval_payload, dict):
            raise TrainerError(
                "resume", "evaluator_restore",
                received=type(eval_payload).__name__,
                expected="dict",
                resolution="Checkpoint 'evaluation' payload is missing or malformed.",
            )

        state_data = eval_payload.get("state")
        if not isinstance(state_data, dict):
            raise TrainerError(
                "resume", "evaluator_restore",
                received=type(state_data).__name__,
                expected="dict",
                resolution="Checkpoint 'evaluation.state' is missing or malformed.",
            )

        es = self._evaluator.state

        # Restore all supported fields with conservative type checking
        if "best_validation_loss" in state_data:
            v = state_data["best_validation_loss"]
            if v is not None and not isinstance(v, (int, float)):
                raise TrainerError(
                    "resume", "evaluator_restore",
                    received=f"best_validation_loss type={type(v).__name__}",
                    expected="float or None",
                    resolution="Checkpoint evaluator state is corrupt.",
                )
            es.best_validation_loss = v
        if "best_validation_epoch" in state_data:
            v = state_data["best_validation_epoch"]
            if v is not None and (not isinstance(v, int) or v < 0):
                raise TrainerError(
                    "resume", "evaluator_restore",
                    received=f"best_validation_epoch={v!r}",
                    expected="non-negative int or None",
                    resolution="Checkpoint evaluator state is corrupt.",
                )
            es.best_validation_epoch = v
        if "best_rmse" in state_data:
            es.best_rmse = state_data["best_rmse"]
        if "best_mae" in state_data:
            es.best_mae = state_data["best_mae"]
        if "best_r2" in state_data:
            es.best_r2 = state_data["best_r2"]
        if "epochs_without_improvement" in state_data:
            v = state_data["epochs_without_improvement"]
            if not isinstance(v, int) or v < 0:
                raise TrainerError(
                    "resume", "evaluator_restore",
                    received=f"epochs_without_improvement={v!r}",
                    expected="non-negative int",
                    resolution="Checkpoint evaluator state is corrupt.",
                )
            es.epochs_without_improvement = v
        if "last_best_update_epoch" in state_data:
            v = state_data["last_best_update_epoch"]
            if v is not None and (not isinstance(v, int) or v < 0):
                raise TrainerError(
                    "resume", "evaluator_restore",
                    received=f"last_best_update_epoch={v!r}",
                    expected="non-negative int or None",
                    resolution="Checkpoint evaluator state is corrupt.",
                )
            es.last_best_update_epoch = v
        if "latest_loss" in state_data:
            es.latest_loss = state_data["latest_loss"]
        if "latest_metrics" in state_data:
            v = state_data["latest_metrics"]
            if isinstance(v, dict):
                es.latest_metrics = dict(v)
        if "samples_evaluated" in state_data:
            v = state_data["samples_evaluated"]
            if isinstance(v, int) and v >= 0:
                es.samples_evaluated = v
        if "batches_evaluated" in state_data:
            v = state_data["batches_evaluated"]
            if isinstance(v, int) and v >= 0:
                es.batches_evaluated = v

    def _do_resume(self) -> None:
        """Execute the two-phase resume workflow.

        Phase 1: Validate checkpoint contract without mutating state.
        Phase 2: Restore training state from checkpoint.
        """
        self._emit_event(_TrainingEvent.RESUME_STARTED)

        ckpt_name = self._config.resume_checkpoint
        if ckpt_name is None:
            ckpt_name = "latest.pt"

        ckpt_path = self._checkpoint_dir / ckpt_name
        resolved = _assert_child_path(self._checkpoint_dir, ckpt_path, "resume")

        # Phase 1: Load and validate contract
        checkpoint = self._load_checkpoint(resolved)
        self._validate_checkpoint_restore_contract(checkpoint)

        # Phase 2: Restore state (only after Phase 1 passes)
        self._restore_checkpoint_state(checkpoint)

        self._emit_event(_TrainingEvent.RESUME_COMPLETED)

    # =========================================================================
    # Internal -- Dashboard
    # =========================================================================

    def _build_dashboard_string(self) -> str:
        """Build the full dashboard string from subsystem summaries."""
        sep = "=" * 64
        lines = [
            sep,
            "  TRAINING DASHBOARD",
            sep,
            "",
        ]

        # Experiment + Identity
        lines.append(f"  Experiment     : {self._config.experiment_name}")
        lines.append(f"  Created At     : {self._run_context.created_at}")
        lines.append(f"  Status         : {self._runtime.status}")
        lines.append(f"  Healthy        : {self._runtime.healthy}")
        lines.append(f"  Last Event     : {self._runtime.last_event}")
        lines.append("")

        # Device + AMP
        lines.append(f"  Device         : {self._run_context.device}")
        lines.append(f"  AMP Status     : {self._runtime.amp_status}")
        if self._runtime.amp_fallback_reason:
            lines.append(f"  AMP Fallback   : {self._runtime.amp_fallback_reason}")
        lines.append("")

        # Epoch / Batch / Step
        lines.append(f"  Epoch          : {self._runtime.current_epoch} / {self._runtime.total_epochs}")
        lines.append(f"  Global Step    : {self._runtime.global_step}")
        lines.append(f"  Optimizer Steps: {self._runtime.optimizer_steps}")
        lines.append("")

        # Timing
        lines.append(f"  Elapsed        : {self._runtime.elapsed_seconds:.1f}s")
        if self._runtime.estimated_remaining_seconds is not None:
            lines.append(f"  ETA            : {self._runtime.estimated_remaining_seconds:.1f}s")
        if self._runtime.train_time_seconds > 0:
            lines.append(f"  Train Time     : {self._runtime.train_time_seconds:.2f}s")
        if self._runtime.val_time_seconds > 0:
            lines.append(f"  Val Time       : {self._runtime.val_time_seconds:.2f}s")
        if self._runtime.checkpoint_time_seconds > 0:
            lines.append(f"  Ckpt Time      : {self._runtime.checkpoint_time_seconds:.2f}s")
        lines.append("")

        # Latest metrics from history
        if self._history.epoch:
            last_idx = -1
            lines.append(f"  Train Loss     : {self._history.train_loss[last_idx]:.6f}")
            val_loss = self._history.validation_loss[last_idx]
            if val_loss == val_loss:  # not NaN
                lines.append(f"  Val Loss       : {val_loss:.6f}")
                lines.append(f"  Val RMSE       : {self._history.validation_rmse[last_idx]:.6f}")
                lines.append(f"  Val MAE        : {self._history.validation_mae[last_idx]:.6f}")
                lines.append(f"  Val R2         : {self._history.validation_r2[last_idx]:.6f}")
            lines.append(f"  Learning Rate  : {self._history.learning_rate[last_idx]:.2e}")
            lines.append("")

        # Best
        if self._history.best_epoch is not None:
            lines.append(f"  Best Epoch     : {self._history.best_epoch}")
            if self._history.best_validation_loss is not None:
                lines.append(f"  Best Val Loss  : {self._history.best_validation_loss:.6f}")
            es = self._evaluator.state
            if es.best_rmse is not None:
                lines.append(f"  Best RMSE      : {es.best_rmse:.6f}")
            if es.best_mae is not None:
                lines.append(f"  Best MAE       : {es.best_mae:.6f}")
            if es.best_r2 is not None:
                lines.append(f"  Best R2        : {es.best_r2:.6f}")
            lines.append("")

        # Optimizer / Scheduler
        opt_type = type(self._optimizer).__name__
        sched_name = self._scheduler_meta.scheduler_type if self._scheduler_meta else "unknown"
        lines.append(f"  Optimizer      : {opt_type}")
        lines.append(f"  Scheduler      : {sched_name}")
        lines.append("")

        # Checkpoint
        cs = self._checkpoint_state
        lines.append(f"  Checkpoints    : {cs.checkpoint_count} saved")
        if cs.latest_checkpoint:
            lines.append(f"  Latest         : {Path(cs.latest_checkpoint).name}")
        if cs.best_checkpoint:
            lines.append(f"  Best           : {Path(cs.best_checkpoint).name}")
        if cs.resume_checkpoint:
            lines.append(f"  Resumed From   : {cs.resume_checkpoint}")
        lines.append("")

        # GPU Memory
        if self._runtime.peak_gpu_memory_mb is not None:
            lines.append(f"  Peak GPU Mem   : {self._runtime.peak_gpu_memory_mb:.1f} MB")
            lines.append("")

        # Warnings / Errors
        if self._runtime.warning_count > 0:
            lines.append(f"  Warnings       : {self._runtime.warning_count}")
        if self._runtime.last_warning:
            lines.append(f"  [WARN] {self._runtime.last_warning}")
        if self._runtime.last_error:
            lines.append(f"  [ERROR] {self._runtime.last_error}")
        if self._runtime.warning_count > 0 or self._runtime.last_error:
            lines.append("")

        lines.append(sep)
        return "\n".join(lines)

    def _render_dashboard(self) -> None:
        """Print the dashboard. Only method allowed to print."""
        dashboard = self._build_dashboard_string()
        print(dashboard, flush=True)

    # =========================================================================
    # Internal -- Timing
    # =========================================================================

    def _update_elapsed_and_eta(self, current_epoch: int) -> None:
        """Update elapsed time and ETA."""
        if self._train_start_time is None:
            return

        elapsed = time.perf_counter() - self._train_start_time
        self._runtime.elapsed_seconds = elapsed

        completed_epochs = current_epoch - self._start_epoch + 1
        remaining_epochs = self._config.epochs - current_epoch

        if completed_epochs > 0 and remaining_epochs >= 0:
            avg_epoch_time = elapsed / completed_epochs
            self._runtime.estimated_remaining_seconds = max(
                0.0, avg_epoch_time * remaining_epochs,
            )

    # =========================================================================
    # Internal -- Failure / Exit
    # =========================================================================

    def _handle_keyboard_interrupt(self) -> None:
        """Handle KeyboardInterrupt gracefully."""
        self._runtime.interrupted = True
        self._runtime.last_warning = "Training interrupted by user."

        # Attempt to save an interrupted checkpoint
        epoch = self._runtime.current_epoch
        if epoch > 0:
            try:
                self._save_checkpoint("interrupted", epoch)
            except Exception:
                self._mark_warning("Failed to save interrupted checkpoint.")

        try:
            self._set_status(_TrainerStatus.INTERRUPTED)
        except TrainerError:
            self._runtime.status = _TrainerStatus.INTERRUPTED.value

        self._emit_event(_TrainingEvent.TRAINING_INTERRUPTED)

        if self._render_dashboard_flag:
            self._render_dashboard()

    def _handle_failure(self, exception: Exception) -> None:
        """Handle unexpected exceptions."""
        self._record_failure(exception, self._runtime.last_event)

        if self._render_dashboard_flag:
            self._render_dashboard()

        raise TrainerError(
            "training", self._runtime.last_event,
            epoch=self._runtime.current_epoch,
            batch=self._runtime.current_batch,
            subsystem="trainer",
            received=str(exception)[:300],
            expected="successful training",
            resolution="See traceback above for root cause.",
        ) from exception


# =============================================================================
# 7. Builder
# =============================================================================

def build_trainer(
    *,
    config: TrainConfig,
    run_context: RunContext,
    model_bundle: nn.ModuleDict,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    evaluator: Evaluator,
    train_loader: Iterable,
    val_loader: Optional[Iterable] = None,
    render_dashboard: bool = True,
) -> Trainer:
    """Build a Trainer instance with full input validation.

    Args:
        config:           Frozen TrainConfig.
        run_context:      RunContext built from the same config.
        model_bundle:     nn.ModuleDict with image_encoder, text_encoder,
                          tabular_encoder, fusion_model.
        optimizer:        Built optimizer from build_optimizer().
        scheduler:        Built scheduler from build_scheduler().
        evaluator:        Built evaluator from build_evaluator().
        train_loader:     Training DataLoader or iterable.
        val_loader:       Validation DataLoader or None.
        render_dashboard: Whether to print dashboard during training.

    Returns:
        Trainer ready to train().

    Raises:
        TrainerError: On any input contract violation.
    """
    return Trainer(
        config=config,
        run_context=run_context,
        model_bundle=model_bundle,
        optimizer=optimizer,
        scheduler=scheduler,
        evaluator=evaluator,
        train_loader=train_loader,
        val_loader=val_loader,
        render_dashboard=render_dashboard,
    )


# =============================================================================
# 8. Smoke Test
# =============================================================================

if __name__ == "__main__":

    import math

    logging.basicConfig(
        level=logging.WARNING,
        format="[%(levelname)s] %(name)s -- %(message)s",
    )

    print("=" * 64)
    print("  training/trainer.py -- smoke test")
    print("=" * 64)

    passed = 0
    failed = 0

    def check(name: str, condition: bool, detail: str = "") -> None:
        global passed, failed
        if condition:
            passed += 1
            print(f"    [PASS]     {name}")
        else:
            failed += 1
            msg = f"    [FAIL]     {name}"
            if detail:
                msg += f"  ({detail})"
            print(msg)

    def expect_error(name: str, exc_type, fn):
        global passed, failed
        try:
            fn()
            failed += 1
            print(f"    [FAIL]     {name}  (no exception raised)")
        except exc_type:
            passed += 1
            print(f"    [PASS]     {name}")
        except Exception as e:
            failed += 1
            print(f"    [FAIL]     {name}  (wrong error: {type(e).__name__}: {e})")

    # -------------------------------------------------------------------------
    # Infrastructure helpers
    # -------------------------------------------------------------------------

    class _DummyImageEncoder(nn.Module):
        def __init__(self, embed_dim: int = 512):
            super().__init__()
            self.proj = nn.Linear(3 * 4 * 4, embed_dim)

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            B = images.shape[0]
            flat = images.view(B, -1)[:, :3*4*4]
            return self.proj(flat)

        def get_embedding_dim(self) -> int:
            return 512

    class _DummyTextEncoder(nn.Module):
        def __init__(self, embed_dim: int = 512):
            super().__init__()
            self.proj = nn.Linear(16, embed_dim)

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            B = input_ids.shape[0]
            x = input_ids.float()[:, :16]
            if x.shape[1] < 16:
                x = torch.nn.functional.pad(x, (0, 16 - x.shape[1]))
            return self.proj(x)

        def get_embedding_dim(self) -> int:
            return 512

    class _DummyTabularEncoder(nn.Module):
        def __init__(self, input_dim: int = 8, embed_dim: int = 512):
            super().__init__()
            self.proj = nn.Linear(input_dim, embed_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj(x)

        def get_embedding_dim(self) -> int:
            return 512

    class _DummyFusionModel(nn.Module):
        def __init__(self, embed_dim: int = 512):
            super().__init__()
            self.head = nn.Linear(embed_dim * 3, 1)

        def forward(self, img_emb, txt_emb, tab_emb):
            combined = torch.cat([img_emb, txt_emb, tab_emb], dim=1)
            pred = self.head(combined).squeeze(-1)
            return {"rating_prediction": pred}

    class _NaNFusionModel(nn.Module):
        """Fusion model that produces NaN predictions for testing."""
        def __init__(self):
            super().__init__()
            self.dummy = nn.Linear(1, 1)

        def forward(self, img_emb, txt_emb, tab_emb):
            B = img_emb.shape[0]
            return {"rating_prediction": torch.tensor([float("nan")] * B)}

    def _make_model_bundle() -> nn.ModuleDict:
        return nn.ModuleDict({
            "image_encoder": _DummyImageEncoder(),
            "text_encoder": _DummyTextEncoder(),
            "tabular_encoder": _DummyTabularEncoder(),
            "fusion_model": _DummyFusionModel(),
        })

    def _make_batch(B: int = 4) -> Dict[str, Any]:
        return {
            "images": torch.randn(B, 3, 4, 4),
            "input_ids": torch.randint(0, 100, (B, 16)),
            "attention_mask": torch.ones(B, 16, dtype=torch.long),
            "tabular": torch.randn(B, 8),
            "ratings": torch.rand(B) * 5,
            "metadata": [{"id": i} for i in range(B)],
        }

    def _make_loader(num_batches: int = 3, batch_size: int = 4) -> List:
        return [_make_batch(batch_size) for _ in range(num_batches)]

    def _make_infra(
        epochs: int = 1,
        loss_name: str = "mse",
        scheduler_type: str = "cosine",
        save_best: bool = False,
        save_latest: bool = False,
        resume: bool = False,
    ):
        cfg = build_train_config(
            device="cpu", loss_name=loss_name,
            epochs=epochs, scheduler=scheduler_type,
            save_best=save_best, save_latest=save_latest,
            resume=resume,
            gradient_clip=1.0,
            gradient_accumulation_steps=1,
            mixed_precision=False,
            validation_frequency=1,
            logging_frequency=1,
            checkpoint_frequency=1,
            warmup_epochs=0,
        )
        cfg.freeze()
        ctx = build_run_context(cfg)
        model = _make_model_bundle()
        opt = build_optimizer(config=cfg, run_context=ctx, model=model)
        sched = build_scheduler(config=cfg, run_context=ctx, optimizer=opt)
        evl = build_evaluator(cfg, ctx)
        return cfg, ctx, model, opt, sched, evl

    # =========================================================================
    # Test Cases
    # =========================================================================

    # -- 1. Trainer construction succeeds ------------------------------------
    print("\n  1. Trainer construction...")
    cfg, ctx, model, opt, sched, evl = _make_infra()
    train_loader = _make_loader()
    val_loader = _make_loader(num_batches=2)
    trainer = build_trainer(
        config=cfg, run_context=ctx, model_bundle=model,
        optimizer=opt, scheduler=sched, evaluator=evl,
        train_loader=train_loader, val_loader=val_loader,
        render_dashboard=False,
    )
    check("trainer created", trainer is not None)
    check("status is ready", trainer.runtime_state()["status"] == "ready")

    # -- 2. Bad config rejected -----------------------------------------------
    print("\n  2. Bad config rejected...")
    expect_error("non-config type", TrainerError,
                 lambda: build_trainer(
                     config="not_a_config", run_context=ctx,
                     model_bundle=model, optimizer=opt, scheduler=sched,
                     evaluator=evl, train_loader=train_loader,
                 ))

    # -- 3. Mismatched config/context -----------------------------------------
    print("\n  3. Mismatched config/context...")
    cfg2, ctx2, _, _, _, _ = _make_infra(loss_name="mae")
    expect_error("config/context mismatch", TrainerError,
                 lambda: build_trainer(
                     config=cfg, run_context=ctx2,
                     model_bundle=model, optimizer=opt, scheduler=sched,
                     evaluator=evl, train_loader=train_loader,
                 ))

    # -- 4. Missing model bundle key ------------------------------------------
    print("\n  4. Missing model bundle key...")
    bad_model = nn.ModuleDict({
        "image_encoder": _DummyImageEncoder(),
        "text_encoder": _DummyTextEncoder(),
    })
    expect_error("missing keys", TrainerError,
                 lambda: build_trainer(
                     config=cfg, run_context=ctx,
                     model_bundle=bad_model, optimizer=opt, scheduler=sched,
                     evaluator=evl, train_loader=train_loader,
                 ))

    # -- 5. Train/val loader identity rejected ---------------------------------
    print("\n  5. Train/val loader same object...")
    same_loader = _make_loader()
    expect_error("loader identity", TrainerError,
                 lambda: build_trainer(
                     config=cfg, run_context=ctx,
                     model_bundle=model, optimizer=opt, scheduler=sched,
                     evaluator=evl, train_loader=same_loader,
                     val_loader=same_loader,
                 ))

    # -- 6. One-epoch training completes --------------------------------------
    print("\n  6. One-epoch training...")
    cfg6, ctx6, model6, opt6, sched6, evl6 = _make_infra(epochs=1)
    tl6 = _make_loader(num_batches=3)
    vl6 = _make_loader(num_batches=2)
    trainer6 = build_trainer(
        config=cfg6, run_context=ctx6, model_bundle=model6,
        optimizer=opt6, scheduler=sched6, evaluator=evl6,
        train_loader=tl6, val_loader=vl6, render_dashboard=False,
    )
    result6 = trainer6.train()
    check("train returned dict", isinstance(result6, dict))
    check("history in result", "history" in result6)
    check("runtime in result", "runtime_state" in result6)

    # -- 7. Runtime state transitions to completed ----------------------------
    print("\n  7. Runtime state completed...")
    rs6 = trainer6.runtime_state()
    check("status completed", rs6["status"] == "completed")
    check("not interrupted", rs6["interrupted"] is False)
    check("not failed", rs6["failed"] is False)
    check("healthy", rs6["healthy"] is True)
    check("started_at set", rs6["started_at"] is not None)
    check("completed_at set", rs6["completed_at"] is not None)

    # -- 8. Event order is valid -----------------------------------------------
    print("\n  8. Event order...")
    events = trainer6._event_log
    check("has initialized", "initialized" in events)
    check("has training_started", "training_started" in events)
    check("has epoch_started", "epoch_started" in events)
    check("has training_completed", "training_completed" in events)
    check("initialized first", events[0] == "initialized")
    check("completed last", events[-1] == "training_completed")

    # -- 9. History has one epoch ----------------------------------------------
    print("\n  9. History has one epoch...")
    h6 = trainer6.history()
    check("epoch count", len(h6["epoch"]) == 1)
    check("epoch is 1", h6["epoch"][0] == 1)
    check("train_loss recorded", isinstance(h6["train_loss"][0], float))
    check("lr recorded", isinstance(h6["learning_rate"][0], float))

    # -- 10. Validation result recorded ----------------------------------------
    print("\n  10. Validation result...")
    check("val_loss recorded", not math.isnan(h6["validation_loss"][0]))
    check("val_rmse recorded", not math.isnan(h6["validation_rmse"][0]))
    check("val_r2 recorded", not math.isnan(h6["validation_r2"][0]))

    # -- 11. Scheduler epoch policy steps --------------------------------------
    print("\n  11. Scheduler epoch policy...")
    check("scheduler_updated in events", "scheduler_updated" in events)

    # -- 12. Plateau scheduler requires validation ----------------------------
    print("\n  12. Plateau scheduler requires validation...")
    cfg_p, ctx_p, mdl_p, opt_p, sched_p, evl_p = _make_infra(
        scheduler_type="plateau", save_best=True,
    )
    expect_error("plateau without val", TrainerError,
                 lambda: build_trainer(
                     config=cfg_p, run_context=ctx_p,
                     model_bundle=mdl_p, optimizer=opt_p, scheduler=sched_p,
                     evaluator=evl_p, train_loader=_make_loader(),
                     val_loader=None,
                 ))

    # -- 13. Checkpoint decision logic ----------------------------------------
    print("\n  13. Checkpoint decisions...")
    cfg13, ctx13, m13, o13, s13, e13 = _make_infra(
        save_latest=True, save_best=True,
    )
    tl13 = _make_loader(num_batches=2)
    vl13 = _make_loader(num_batches=1)
    trainer13 = build_trainer(
        config=cfg13, run_context=ctx13, model_bundle=m13,
        optimizer=o13, scheduler=s13, evaluator=e13,
        train_loader=tl13, val_loader=vl13, render_dashboard=False,
    )
    check("should_save_latest True", trainer13._should_save_latest(1) is True)
    check("should_save_best False", trainer13._should_save_best(False) is False)
    check("should_save_best True", trainer13._should_save_best(True) is True)

    # -- 14. Checkpoint save/load using tempdir --------------------------------
    print("\n  14. Checkpoint save/load...")
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg14, ctx14, m14, o14, s14, e14 = _make_infra(
            save_latest=True, save_best=True,
        )
        tl14 = _make_loader(num_batches=2)
        vl14 = _make_loader(num_batches=1)
        trainer14 = build_trainer(
            config=cfg14, run_context=ctx14, model_bundle=m14,
            optimizer=o14, scheduler=s14, evaluator=e14,
            train_loader=tl14, val_loader=vl14, render_dashboard=False,
        )
        # Override checkpoint dir to temp
        trainer14._checkpoint_dir = Path(tmpdir)
        trainer14._capture_reproducibility()
        trainer14._prepare_model()
        trainer14._set_status(_TrainerStatus.RUNNING)
        trainer14._runtime.global_step = 42
        trainer14._runtime.current_epoch = 1

        # Save
        trainer14._save_checkpoint("latest", 1)
        ckpt_path = Path(tmpdir) / "latest.pt"
        check("checkpoint file exists", ckpt_path.exists())

        # Load
        loaded = trainer14._load_checkpoint(ckpt_path)
        check("checkpoint has version", loaded["checkpoint_version"] == _CHECKPOINT_VERSION)
        check("checkpoint has epoch", loaded["epoch"] == 1)
        check("checkpoint has global_step", loaded["global_step"] == 42)
        check("checkpoint has model state", "model_state_dict" in loaded)

    # -- 15. Resume restores epoch/global step/history -------------------------
    print("\n  15. Resume restores state...")
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg15, ctx15, m15, o15, s15, e15 = _make_infra(
            epochs=2, save_latest=True, resume=False,
        )
        tl15 = _make_loader(num_batches=2)
        vl15 = _make_loader(num_batches=1)
        t15 = build_trainer(
            config=cfg15, run_context=ctx15, model_bundle=m15,
            optimizer=o15, scheduler=s15, evaluator=e15,
            train_loader=tl15, val_loader=vl15, render_dashboard=False,
        )
        t15._checkpoint_dir = Path(tmpdir)
        t15._capture_reproducibility()
        t15._prepare_model()
        t15._set_status(_TrainerStatus.RUNNING)
        t15._runtime.global_step = 10
        t15._runtime.optimizer_steps = 5
        t15._runtime.current_epoch = 1
        t15._history.epoch = [1]
        t15._history.train_loss = [0.5]
        t15._history.validation_loss = [0.4]
        t15._history.validation_rmse = [0.63]
        t15._history.validation_mae = [0.3]
        t15._history.validation_r2 = [0.8]
        t15._history.learning_rate = [0.001]
        t15._history.epoch_duration_seconds = [1.0]
        t15._history.checkpoint_saved = [True]
        t15._save_checkpoint("latest", 1)

        # Now create a new trainer and restore
        cfg15b, ctx15b, m15b, o15b, s15b, e15b = _make_infra(
            epochs=2, save_latest=True, resume=False,
        )
        t15b = build_trainer(
            config=cfg15b, run_context=ctx15b, model_bundle=m15b,
            optimizer=o15b, scheduler=s15b, evaluator=e15b,
            train_loader=_make_loader(2), val_loader=_make_loader(1),
            render_dashboard=False,
        )
        t15b._checkpoint_dir = Path(tmpdir)
        ckpt_data = t15b._load_checkpoint(Path(tmpdir) / "latest.pt")
        t15b._restore_checkpoint_state(ckpt_data)
        check("resume start_epoch", t15b._start_epoch == 2)
        check("resume global_step", t15b._runtime.global_step == 10)
        check("resume history", len(t15b._history.epoch) == 1)
        check("resume history epoch", t15b._history.epoch[0] == 1)

    # -- 16. KeyboardInterrupt path -------------------------------------------
    print("\n  16. KeyboardInterrupt handling...")
    cfg16, ctx16, m16, o16, s16, e16 = _make_infra()

    class _InterruptLoader:
        def __iter__(self):
            raise KeyboardInterrupt

    t16 = build_trainer(
        config=cfg16, run_context=ctx16, model_bundle=m16,
        optimizer=o16, scheduler=s16, evaluator=e16,
        train_loader=_InterruptLoader(), render_dashboard=False,
    )
    res16 = t16.train()
    check("interrupted status", res16["runtime_state"]["status"] == "interrupted")
    check("interrupted flag", res16["runtime_state"]["interrupted"] is True)

    # -- 17. NaN loss fails loudly -------------------------------------------
    print("\n  17. NaN loss detection...")
    cfg17, ctx17, _, o17_dummy, s17, e17 = _make_infra()
    nan_model = nn.ModuleDict({
        "image_encoder": _DummyImageEncoder(),
        "text_encoder": _DummyTextEncoder(),
        "tabular_encoder": _DummyTabularEncoder(),
        "fusion_model": _NaNFusionModel(),
    })
    o17 = build_optimizer(config=cfg17, run_context=ctx17, model=nan_model)
    s17b = build_scheduler(config=cfg17, run_context=ctx17, optimizer=o17)
    e17b = build_evaluator(cfg17, ctx17)
    t17 = build_trainer(
        config=cfg17, run_context=ctx17, model_bundle=nan_model,
        optimizer=o17, scheduler=s17b, evaluator=e17b,
        train_loader=_make_loader(1), render_dashboard=False,
    )
    try:
        t17.train()
        check("NaN loss detected", False, "no error raised")
    except (TrainerError, Exception):
        check("NaN loss detected", True)

    # -- 18. Missing batch key fails ------------------------------------------
    print("\n  18. Missing batch key...")
    cfg18, ctx18, m18, o18, s18, e18 = _make_infra()
    bad_batch = [{"images": torch.randn(2, 3, 4, 4)}]  # Missing keys
    t18 = build_trainer(
        config=cfg18, run_context=ctx18, model_bundle=m18,
        optimizer=o18, scheduler=s18, evaluator=e18,
        train_loader=bad_batch, render_dashboard=False,
    )
    try:
        t18.train()
        check("missing key detected", False, "no error raised")
    except TrainerError:
        check("missing key detected", True)

    # -- 19. Summary returns string -------------------------------------------
    print("\n  19. Summary...")
    summ = trainer6.summary()
    check("summary is string", isinstance(summ, str) and len(summ) > 50)
    check("summary has experiment", cfg6.experiment_name in summ)

    # -- 20. as_dict returns serializable dict ---------------------------------
    print("\n  20. Serialization...")
    d = trainer6.as_dict()
    check("as_dict is dict", isinstance(d, dict))
    check("has runtime_state", "runtime_state" in d)
    check("has history", "history" in d)
    check("has checkpoint_state", "checkpoint_state" in d)
    check("has evaluator", "evaluator" in d)
    check("has reproducibility", "reproducibility" in d)

    # -- 21. No hardcoded paths -----------------------------------------------
    print("\n  21. No hardcoded paths...")
    import inspect
    src = inspect.getsource(Trainer)
    check("no /content/drive", "/content/drive" not in src)
    check("no D:/", "D:/" not in src and "D:\\" not in src)
    check("no C:/", "C:/" not in src and "C:\\" not in src)

    # -- 22. No .cuda() calls -------------------------------------------------
    print("\n  22. No .cuda() calls...")
    check("no .cuda()", ".cuda()" not in src)

    # -- 23. No dataset/model construction inside trainer ----------------------
    print("\n  23. No internal construction...")
    full_src = inspect.getsource(sys.modules[__name__])
    check("no Dataset()", "Dataset(" not in full_src.split("class _Dummy")[0])
    check("no ImageEncoder()", "ImageEncoder(" not in full_src.split("class _Dummy")[0])

    # -- 24. _TRAINER_SCHEMA_VERSION constant exists ----------------------------
    print("\n  24. Schema version constant...")
    check("schema version exists", _TRAINER_SCHEMA_VERSION >= 1)
    check("checkpoint keys has schema", "trainer_schema_version" in _CHECKPOINT_REQUIRED_KEYS)
    check("checkpoint keys has class", "trainer_class" in _CHECKPOINT_REQUIRED_KEYS)

    # -- 25. _assert_child_path rejects traversal ------------------------------
    print("\n  25. Path traversal guard...")
    from pathlib import Path as _P25
    _base25 = _P25(tempfile.mkdtemp())
    try:
        _assert_child_path(_base25, _base25 / "safe.pt", "test")
        check("safe path passes", True)
    except TrainerError:
        check("safe path passes", False, "raised on safe path")
    try:
        _assert_child_path(_base25, _base25 / ".." / "escape.pt", "test")
        check("traversal rejected", False, "no error for traversal")
    except TrainerError:
        check("traversal rejected", True)

    # -- 26. _gpu_memory_snapshot returns dict ----------------------------------
    print("\n  26. GPU snapshot...")
    snap26 = _gpu_memory_snapshot()
    check("snapshot is dict", isinstance(snap26, dict))
    check("has cuda_available", "cuda_available" in snap26)
    check("has max_allocated_mb", "max_allocated_mb" in snap26)

    # -- 27. _record_failure double-call guard ---------------------------------
    print("\n  27. Record failure guard...")
    cfg27, ctx27, m27, o27, s27, e27 = _make_infra()
    t27 = build_trainer(
        config=cfg27, run_context=ctx27, model_bundle=m27,
        optimizer=o27, scheduler=s27, evaluator=e27,
        train_loader=_make_loader(1), render_dashboard=False,
    )
    t27._record_failure(RuntimeError("first"))
    check("failed set", t27._runtime.failed is True)
    first_error = t27._runtime.last_error
    t27._record_failure(RuntimeError("second"))
    check("double-call no overwrite", t27._runtime.last_error == first_error)

    # -- 28. Runtime state has AMP + timing fields -----------------------------
    print("\n  28. Runtime state fields...")
    rt28 = trainer6.runtime_state()
    for fld in ("amp_status", "amp_fallback_reason", "train_time_seconds",
                "val_time_seconds", "checkpoint_time_seconds",
                "warning_count", "peak_gpu_memory_mb"):
        check(f"runtime has {fld}", fld in rt28)

    # -- 29. _validate_prediction_contract rejects bad shape -------------------
    print("\n  29. Prediction contract...")
    cfg29, ctx29, m29, o29, s29, e29 = _make_infra()
    t29 = build_trainer(
        config=cfg29, run_context=ctx29, model_bundle=m29,
        optimizer=o29, scheduler=s29, evaluator=e29,
        train_loader=_make_loader(1), render_dashboard=False,
    )
    good_pred = torch.randn(4)
    good_tgt = torch.randn(4)
    try:
        t29._validate_prediction_contract(good_pred, good_tgt, 1, 1)
        check("good pred passes", True)
    except TrainerError:
        check("good pred passes", False)
    bad_pred = torch.randn(4, 3)  # wrong shape [B, 3]
    try:
        t29._validate_prediction_contract(bad_pred, good_tgt, 1, 1)
        check("bad shape rejected", False, "no error")
    except TrainerError:
        check("bad shape rejected", True)
    complex_pred = torch.randn(4) + 1j * torch.randn(4)
    try:
        t29._validate_prediction_contract(complex_pred, good_tgt, 1, 1)
        check("complex rejected", False, "no error")
    except TrainerError:
        check("complex rejected", True)

    # -- 30. Dashboard includes new sections -----------------------------------
    print("\n  30. Enhanced dashboard...")
    cfg30, ctx30, m30, o30, s30, e30 = _make_infra()
    t30 = build_trainer(
        config=cfg30, run_context=ctx30, model_bundle=m30,
        optimizer=o30, scheduler=s30, evaluator=e30,
        train_loader=_make_loader(1), render_dashboard=False,
    )
    dash30 = t30._build_dashboard_string()
    check("dashboard has created at", "Created At" in dash30)
    check("dashboard has device", "Device" in dash30)
    check("dashboard has AMP status", "AMP Status" in dash30)
    check("dashboard has optimizer", "Optimizer" in dash30)
    check("dashboard has scheduler", "Scheduler" in dash30)

    # -- 31. _validate_batch rejects integer ratings ---------------------------
    print("\n  31. Batch ratings validation...")
    cfg31, ctx31, m31, o31, s31, e31 = _make_infra()
    t31 = build_trainer(
        config=cfg31, run_context=ctx31, model_bundle=m31,
        optimizer=o31, scheduler=s31, evaluator=e31,
        train_loader=_make_loader(1), render_dashboard=False,
    )
    int_batch = {
        "images": torch.randn(2, 3, 4, 4),
        "input_ids": torch.randint(0, 100, (2, 8)),
        "attention_mask": torch.ones(2, 8),
        "tabular": torch.randn(2, 5),
        "ratings": torch.tensor([3, 4]),  # integer, not float
    }
    try:
        t31._validate_batch(int_batch, 1, 1)
        check("integer ratings rejected", False, "no error")
    except TrainerError:
        check("integer ratings rejected", True)

    # -- 32. Non-iterable loader rejected --------------------------------------
    print("\n  32. Non-iterable loader rejection...")
    cfg32, ctx32, m32, o32, s32, e32 = _make_infra()
    for bad_loader in [123, "bad", {}, 3.14, True]:
        try:
            build_trainer(
                config=cfg32, run_context=ctx32, model_bundle=m32,
                optimizer=o32, scheduler=s32, evaluator=e32,
                train_loader=bad_loader, render_dashboard=False,
            )
            check(f"loader {type(bad_loader).__name__} rejected", False, "no error")
        except TrainerError:
            check(f"loader {type(bad_loader).__name__} rejected", True)

    # -- 33. Valid list-of-batches still accepted --------------------------------
    print("\n  33. Valid loader accepted...")
    cfg33, ctx33, m33, o33, s33, e33 = _make_infra()
    t33 = build_trainer(
        config=cfg33, run_context=ctx33, model_bundle=m33,
        optimizer=o33, scheduler=s33, evaluator=e33,
        train_loader=_make_loader(1), render_dashboard=False,
    )
    check("valid list loader accepted", t33 is not None)

    # -- 34. save_best=True without val_loader expected guard -------------------
    print("\n  34. save_best without val_loader...")
    from training.train_config import build_train_config as _btc34
    _tc34 = _btc34(device="cpu", warmup_epochs=0, save_best=True, epochs=1)
    _tc34.freeze()
    from training.run_context import build_run_context as _brc34
    _rc34 = _brc34(_tc34)
    from training.optimizer import build_optimizer as _bo34
    from training.scheduler import build_scheduler as _bs34
    from training.evaluation import build_evaluator as _be34
    _m34 = nn.ModuleDict({
        "image_encoder": _DummyImageEncoder(),
        "text_encoder": _DummyTextEncoder(),
        "tabular_encoder": _DummyTabularEncoder(),
        "fusion_model": _DummyFusionModel(),
    })
    _o34 = _bo34(config=_tc34, run_context=_rc34, model=_m34)
    _s34 = _bs34(config=_tc34, run_context=_rc34, optimizer=_o34)
    _e34 = _be34(_tc34, _rc34)
    try:
        build_trainer(
            config=_tc34, run_context=_rc34, model_bundle=_m34,
            optimizer=_o34, scheduler=_s34, evaluator=_e34,
            train_loader=_make_loader(1), val_loader=None,
            render_dashboard=False,
        )
        check("save_best no val_loader rejected", False, "no error")
    except TrainerError:
        check("save_best no val_loader rejected", True)

    # -- 35. Validation prediction contract catches [B,1,1] ---------------------
    print("\n  35. Validation prediction [B,1,1] contract...")
    cfg35, ctx35, m35, o35, s35, e35 = _make_infra()
    t35 = build_trainer(
        config=cfg35, run_context=ctx35, model_bundle=m35,
        optimizer=o35, scheduler=s35, evaluator=e35,
        train_loader=_make_loader(1), render_dashboard=False,
    )
    bad_pred_35 = torch.randn(4, 1, 1)
    tgt_35 = torch.randn(4)
    try:
        t35._validate_prediction_contract(bad_pred_35, tgt_35, 1, 1)
        check("[B,1,1] rejected", False, "no error")
    except TrainerError:
        check("[B,1,1] rejected", True)
    # [B] and [B,1] should pass
    for shape_name, pred in [("[B]", torch.randn(4)), ("[B,1]", torch.randn(4, 1))]:
        try:
            t35._validate_prediction_contract(pred, tgt_35, 1, 1)
            check(f"{shape_name} accepted", True)
        except TrainerError:
            check(f"{shape_name} accepted", False)

    # -- 36. Malformed evaluator restore raises TrainerError --------------------
    print("\n  36. Malformed evaluator restore...")
    cfg36, ctx36, m36, o36, s36, e36 = _make_infra()
    t36 = build_trainer(
        config=cfg36, run_context=ctx36, model_bundle=m36,
        optimizer=o36, scheduler=s36, evaluator=e36,
        train_loader=_make_loader(1), render_dashboard=False,
    )
    try:
        t36._restore_evaluator_state({"evaluation": "not_a_dict"})
        check("malformed eval payload rejected", False, "no error")
    except TrainerError:
        check("malformed eval payload rejected", True)
    try:
        t36._restore_evaluator_state({"evaluation": {"state": "not_a_dict"}})
        check("malformed eval state rejected", False, "no error")
    except TrainerError:
        check("malformed eval state rejected", True)
    # Missing evaluation key
    try:
        t36._restore_evaluator_state({})
        check("missing eval key rejected", False, "no error")
    except TrainerError:
        check("missing eval key rejected", True)
    # Valid restore
    valid_eval_ckpt = {
        "evaluation": {
            "state": {
                "best_validation_loss": 0.25,
                "best_validation_epoch": 5,
                "best_rmse": 0.5,
                "best_mae": 0.2,
                "best_r2": 0.9,
                "epochs_without_improvement": 3,
                "last_best_update_epoch": 5,
            }
        }
    }
    t36._restore_evaluator_state(valid_eval_ckpt)
    check("valid eval restore best_epoch", t36._evaluator.state.best_validation_epoch == 5)
    check("valid eval restore last_best", t36._evaluator.state.last_best_update_epoch == 5)

    # -- 37. Optimizer-model parameter integrity --------------------------------
    print("\n  37. Optimizer-model integrity...")
    cfg37, ctx37, m37, o37, s37, e37 = _make_infra()
    t37 = build_trainer(
        config=cfg37, run_context=ctx37, model_bundle=m37,
        optimizer=o37, scheduler=s37, evaluator=e37,
        train_loader=_make_loader(1), render_dashboard=False,
    )
    # Normal should pass
    try:
        t37._validate_optimizer_model_integrity(1)
        check("normal opt-model passes", True)
    except TrainerError:
        check("normal opt-model passes", False)
    # Replace fusion_model -> optimizer now has stale params
    old_fusion = m37["fusion_model"]
    m37["fusion_model"] = nn.Linear(5, 1)  # new params not in optimizer
    try:
        t37._validate_optimizer_model_integrity(1)
        check("replaced fusion rejected", False, "no error")
    except TrainerError:
        check("replaced fusion rejected", True)
    m37["fusion_model"] = old_fusion  # restore for safety

    # -- 38. Full device validation checks all params ---------------------------
    print("\n  38. Full model device validation...")
    cfg38, ctx38, m38, o38, s38, e38 = _make_infra()
    t38 = build_trainer(
        config=cfg38, run_context=ctx38, model_bundle=m38,
        optimizer=o38, scheduler=s38, evaluator=e38,
        train_loader=_make_loader(1), render_dashboard=False,
    )
    # All on CPU should pass
    try:
        t38._validate_model_device()
        check("all-CPU device passes", True)
    except TrainerError:
        check("all-CPU device passes", False)

    # -- 39. _validate_loader_like is importable and callable -------------------
    print("\n  39. Loader validation helper...")
    check("_validate_loader_like callable", callable(_validate_loader_like))
    try:
        _validate_loader_like(None, "test")
        check("None loader rejected", False)
    except TrainerError:
        check("None loader rejected", True)

    # -- Final -----------------------------------------------------------------
    total = passed + failed
    print(f"\n{'=' * 64}")
    if failed == 0:
        print(f"  [PASS]  {passed}/{total} checks passed")
    else:
        print(f"  [FAIL]  {passed}/{total} passed, {failed} failed")
    print("=" * 64)

    sys.exit(1 if failed > 0 else 0)
