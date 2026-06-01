# =============================================================================
# data_pipeline/dataloader_factory.py
# Multimodal Execution Scheduling Authority — Multimodal AI Pipeline
# =============================================================================
#
# Ownership (this file ONLY):
#   - DataLoader construction and lifecycle configuration
#   - Execution mode presets (debug/safe/colab/throughput)
#   - Hardware-aware num_workers selection
#   - pin_memory / prefetch_factor / persistent_workers strategy
#   - Worker seed initialization for reproducibility
#   - Environment detection (platform, CUDA, Colab)
#   - Worker risk analysis and prefetch memory estimation
#   - First-batch probing and health reporting
#   - Lightweight throughput telemetry
#
# What this file does NOT own:
#   +-----------------------------+---------------------------+
#   | Responsibility              | Correct File              |
#   +-----------------------------+---------------------------+
#   | image preprocessing         | data_pipeline/transforms  |
#   | text tokenization           | data_pipeline/tokenization|
#   | sample construction         | data_pipeline/dataset.py  |
#   | batch semantics / stacking  | data_pipeline/collate.py  |
#   | GPU transfer / .cuda()      | train.py                  |
#   | model forward passes        | models/*.py               |
#   | training loops / optimizer   | train.py                  |
#   | loss / metrics / checkpoints | train.py                  |
#   +-----------------------------+---------------------------+
#
# Design:
#   STATELESS   — factory functions, no mutable global state
#   CPU-ONLY    — no .cuda(), no .to(device), no manual .pin_memory()
#   COLAB-SAFE  — conservative defaults for unstable runtimes
#   OBSERVABLE  — health reports, risk flags, probe results
# =============================================================================

from __future__ import annotations

import os
import sys
import time
import random
import platform
import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------
# Project Routing
# ---------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _THIS_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from data_pipeline.dataset import DatasetConfig, MultimodalProductDataset
from data_pipeline.collate import CollateConfig, build_collate_fn

logger = logging.getLogger(__name__)

_VALID_MODES = frozenset({"debug", "safe", "colab", "throughput"})

_REQUIRED_BATCH_KEYS = frozenset({
    "images", "input_ids", "attention_mask",
    "tabular", "ratings", "sample_ids",
    "row_indices", "asins",
})


# =============================================================================
# 1. Error helper
# =============================================================================

def _dataloader_error(
    stage: str,
    message: str,
    config: Optional["DataLoaderConfig"] = None,
    env: Optional[Dict[str, Any]] = None,
    cause: Optional[Exception] = None,
    resolution: str = "",
) -> str:
    parts = [
        "[DATALOADER ERROR]",
        f"  Stage          : {stage}",
        f"  Message        : {message}",
    ]
    if config:
        parts.append(f"  Execution mode : {config.execution_mode}")
        parts.append(f"  Batch size     : {config.batch_size}")
    if env:
        parts.append(f"  Platform       : {env.get('platform', 'unknown')}")
    if resolution:
        parts.append(f"  Resolution     : {resolution}")
    if cause:
        parts.append(f"  Cause          : {type(cause).__name__}: {cause}")
    return "\n".join(parts)


# =============================================================================
# 2. DataLoaderConfig
# =============================================================================

@dataclass
class DataLoaderConfig:
    """Configuration for DataLoader construction."""

    batch_size: int = 16
    shuffle: bool = True
    drop_last: bool = False

    num_workers: Optional[int] = None
    pin_memory: Optional[bool] = None
    persistent_workers: Optional[bool] = None
    prefetch_factor: Optional[int] = None
    timeout: float = 0.0

    worker_init_seed: int = 42

    execution_mode: str = "safe"
    enable_health_report: bool = True
    enable_probe: bool = False
    enable_warnings: bool = True
    enable_timing: bool = True

    max_prefetch_memory_mb: float = 1024.0
    approx_sample_mb: float = 2.0

    dataset_size_hint: Optional[int] = 6400

    def __post_init__(self):
        # -- batch_size --
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise TypeError(
                f"DataLoaderConfig.batch_size must be positive int, "
                f"got {type(self.batch_size).__name__}: {self.batch_size!r}"
            )
        # -- bool fields --
        for name in ("shuffle", "drop_last", "enable_health_report",
                      "enable_probe", "enable_warnings", "enable_timing"):
            val = getattr(self, name)
            if not isinstance(val, bool):
                raise TypeError(
                    f"DataLoaderConfig.{name} must be bool, "
                    f"got {type(val).__name__}: {val!r}"
                )
        # -- num_workers --
        if self.num_workers is not None:
            if isinstance(self.num_workers, bool) or not isinstance(self.num_workers, int) or self.num_workers < 0:
                raise TypeError(
                    f"DataLoaderConfig.num_workers must be None or int >= 0, "
                    f"got {type(self.num_workers).__name__}: {self.num_workers!r}"
                )
        # -- pin_memory --
        if self.pin_memory is not None and not isinstance(self.pin_memory, bool):
            raise TypeError(
                f"DataLoaderConfig.pin_memory must be None or bool, "
                f"got {type(self.pin_memory).__name__}: {self.pin_memory!r}"
            )
        # -- persistent_workers --
        if self.persistent_workers is not None and not isinstance(self.persistent_workers, bool):
            raise TypeError(
                f"DataLoaderConfig.persistent_workers must be None or bool, "
                f"got {type(self.persistent_workers).__name__}: {self.persistent_workers!r}"
            )
        # -- prefetch_factor --
        if self.prefetch_factor is not None:
            if isinstance(self.prefetch_factor, bool) or not isinstance(self.prefetch_factor, int) or self.prefetch_factor <= 0:
                raise TypeError(
                    f"DataLoaderConfig.prefetch_factor must be None or positive int, "
                    f"got {type(self.prefetch_factor).__name__}: {self.prefetch_factor!r}"
                )
        # -- timeout --
        if isinstance(self.timeout, bool) or not isinstance(self.timeout, (int, float)) or self.timeout < 0:
            raise TypeError(
                f"DataLoaderConfig.timeout must be non-negative number, "
                f"got {type(self.timeout).__name__}: {self.timeout!r}"
            )
        # -- worker_init_seed --
        if isinstance(self.worker_init_seed, bool) or not isinstance(self.worker_init_seed, int):
            raise TypeError(
                f"DataLoaderConfig.worker_init_seed must be int, "
                f"got {type(self.worker_init_seed).__name__}: {self.worker_init_seed!r}"
            )
        # -- execution_mode --
        if not isinstance(self.execution_mode, str) or self.execution_mode not in _VALID_MODES:
            raise ValueError(
                f"DataLoaderConfig.execution_mode must be one of {sorted(_VALID_MODES)}, "
                f"got {self.execution_mode!r}"
            )
        # -- memory budget --
        for name in ("max_prefetch_memory_mb", "approx_sample_mb"):
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, (int, float)) or val <= 0:
                raise TypeError(
                    f"DataLoaderConfig.{name} must be positive number, "
                    f"got {type(val).__name__}: {val!r}"
                )
        # -- dataset_size_hint --
        if self.dataset_size_hint is not None:
            if isinstance(self.dataset_size_hint, bool) or not isinstance(self.dataset_size_hint, int) or self.dataset_size_hint <= 0:
                raise TypeError(
                    f"DataLoaderConfig.dataset_size_hint must be None or positive int, "
                    f"got {type(self.dataset_size_hint).__name__}: {self.dataset_size_hint!r}"
                )


# =============================================================================
# 3. Environment detection
# =============================================================================

def detect_execution_environment() -> Dict[str, Any]:
    """Detect runtime environment for worker/pin strategy decisions."""
    cpu_count = os.cpu_count() or 1
    cuda_available = torch.cuda.is_available()
    cuda_count = torch.cuda.device_count() if cuda_available else 0
    cuda_name = None
    if cuda_available and cuda_count > 0:
        try:
            cuda_name = torch.cuda.get_device_name(0)
        except Exception:
            cuda_name = "unknown"

    # Colab detection
    is_colab = False
    if "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ:
        is_colab = True
    elif os.path.isdir("/content"):
        try:
            import importlib
            importlib.import_module("google.colab")
            is_colab = True
        except (ImportError, ModuleNotFoundError):
            pass

    plat = platform.system()
    return {
        "platform": plat,
        "python_version": platform.python_version(),
        "cpu_count": cpu_count,
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_count,
        "cuda_device_name": cuda_name,
        "is_colab": is_colab,
        "is_windows": plat == "Windows",
        "is_linux": plat == "Linux",
    }


# =============================================================================
# 4. Dynamic worker selection
# =============================================================================

def select_num_workers(config: DataLoaderConfig, env: Dict[str, Any]) -> int:
    """Select safe num_workers based on config, mode, and environment."""
    # Explicit override
    if config.num_workers is not None:
        return config.num_workers

    mode = config.execution_mode
    cpu_count = env.get("cpu_count", 1) or 1

    if mode == "debug":
        return 0

    if mode == "colab":
        return min(2, max(0, cpu_count - 1))

    if mode == "throughput":
        # Cap throughput on Colab for runtime stability
        if env.get("is_colab"):
            return min(4, max(1, cpu_count - 1))
        if env.get("is_windows"):
            return min(4, max(0, cpu_count - 1))
        return min(8, max(0, cpu_count - 1))

    # safe mode
    if cpu_count <= 2:
        return max(0, cpu_count - 1)
    if cpu_count <= 4:
        return 2
    if env.get("is_windows"):
        return min(2, max(0, cpu_count - 1))
    return min(4, max(0, cpu_count - 1))


# =============================================================================
# 5. Prefetch memory estimation
# =============================================================================

def estimate_prefetch_memory(
    batch_size: int,
    num_workers: int,
    prefetch_factor: Optional[int],
    approx_sample_mb: float = 2.0,
) -> Dict[str, Any]:
    """Estimate memory used by DataLoader prefetch queues (heuristic)."""
    base = {
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor,
        "estimate_type": "heuristic",
        "is_exact": False,
        "assumptions": {
            "approx_sample_mb": approx_sample_mb,
            "does_not_include": [
                "Python object overhead",
                "worker process overhead",
                "tokenizer memory",
                "OS page cache",
                "pinned memory allocator overhead",
            ],
        },
    }
    if num_workers == 0 or prefetch_factor is None:
        base["estimated_prefetch_batches"] = 0
        base["estimated_prefetch_memory_mb"] = 0.0
        return base
    prefetch_batches = num_workers * prefetch_factor
    mem_mb = batch_size * prefetch_batches * approx_sample_mb
    base["estimated_prefetch_batches"] = prefetch_batches
    base["estimated_prefetch_memory_mb"] = round(mem_mb, 2)
    return base


# =============================================================================
# 6. Worker risk analysis
# =============================================================================

def analyze_worker_risk(
    config: DataLoaderConfig,
    env: Dict[str, Any],
    selected_workers: int,
    resolved_pin_memory: bool = False,
    resolved_persistent_workers: bool = False,
    resolved_prefetch_factor: Optional[int] = None,
    effective_dataset_size_hint: Optional[int] = None,
) -> List[str]:
    """Identify execution risks using actual resolved DataLoader settings."""
    risks: List[str] = []
    cpu_count = env.get("cpu_count", 1) or 1

    if selected_workers == 0:
        risks.append(
            "num_workers=0: no multiprocessing overlap; "
            "good for debugging but may bottleneck GPU."
        )
    if env.get("is_windows") and selected_workers > 4:
        risks.append(
            f"Windows with num_workers={selected_workers}: "
            "multiprocessing spawn overhead may be high."
        )
    if selected_workers > 0 and selected_workers >= cpu_count:
        risks.append(
            f"num_workers={selected_workers} >= cpu_count={cpu_count}: "
            "oversubscription risk."
        )
    # Prefetch memory (uses resolved prefetch_factor, not raw config)
    if selected_workers > 0 and resolved_prefetch_factor:
        est = estimate_prefetch_memory(
            config.batch_size, selected_workers,
            resolved_prefetch_factor, config.approx_sample_mb,
        )
        if est["estimated_prefetch_memory_mb"] > config.max_prefetch_memory_mb:
            risks.append(
                f"Prefetch memory estimate {est['estimated_prefetch_memory_mb']:.0f}MB "
                f"exceeds budget {config.max_prefetch_memory_mb:.0f}MB."
            )
    # pin_memory without CUDA (uses resolved value)
    if resolved_pin_memory and not env.get("cuda_available"):
        risks.append(
            "pin_memory=True but CUDA unavailable: no benefit for async transfer."
        )
    # Large batch on small dataset
    hint = effective_dataset_size_hint if effective_dataset_size_hint is not None else config.dataset_size_hint
    if hint and hint <= 10000 and config.batch_size >= 64:
        risks.append(
            f"batch_size={config.batch_size} on ~{hint} samples: "
            "large batch on small dataset may reduce generalization."
        )
    # Persistent workers (uses resolved value)
    if resolved_persistent_workers and selected_workers > 0:
        risks.append(
            "persistent_workers=True: worker memory persists between epochs."
        )
    return risks


# =============================================================================
# 7. Worker init function
# =============================================================================

def make_worker_init_fn(seed: int):
    """Create a deterministic worker init function for reproducibility."""
    def _worker_init_fn(worker_id: int) -> None:
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        try:
            import numpy as np
            np.random.seed(worker_seed % (2**32))
        except ImportError:
            pass
    return _worker_init_fn


# =============================================================================
# 8. Preset resolution
# =============================================================================

def _resolve_loader_params(
    config: DataLoaderConfig,
    env: Dict[str, Any],
    num_workers: int,
) -> Dict[str, Any]:
    """Resolve pin_memory, persistent_workers, prefetch_factor from mode.

    Returns dict with resolved values and normalization_notes.
    """
    notes: List[str] = []

    # -- pin_memory --
    if config.pin_memory is not None:
        pin = config.pin_memory
    else:
        pin = env.get("cuda_available", False)
    # Normalize: pin on CPU-only is useless
    if pin and not env.get("cuda_available"):
        notes.append("pin_memory=True normalized to False because CUDA is unavailable")
        if config.enable_warnings:
            warnings.warn(
                "pin_memory=True on CPU-only runtime has no benefit. "
                "Normalizing to False.",
                RuntimeWarning,
                stacklevel=3,
            )
        pin = False

    # -- persistent_workers --
    if num_workers == 0:
        persist = False
        if config.persistent_workers is True:
            notes.append("persistent_workers ignored because num_workers=0")
    elif config.persistent_workers is not None:
        persist = config.persistent_workers
    elif config.execution_mode == "debug":
        persist = False
    else:
        persist = True

    # Notebook/Colab persistent worker risk
    # Future: add safe shutdown helper if notebook worker lifecycle becomes painful.
    if persist and env.get("is_colab"):
        notes.append(
            "persistent_workers=True in Colab: workers may retain memory/state after interrupted runs"
        )

    # -- prefetch_factor --
    if num_workers == 0:
        prefetch = None
        if config.prefetch_factor is not None:
            notes.append("prefetch_factor ignored because num_workers=0")
    elif config.prefetch_factor is not None:
        prefetch = config.prefetch_factor
    elif config.execution_mode == "throughput":
        # Cap prefetch on Colab for stability
        prefetch = 2 if env.get("is_colab") else 4
        if env.get("is_colab"):
            notes.append("throughput prefetch capped to 2 for Colab runtime stability")
    else:
        prefetch = 2

    return {
        "pin_memory": pin,
        "persistent_workers": persist,
        "prefetch_factor": prefetch,
        "normalization_notes": notes,
    }


# =============================================================================
# 9. First-batch probe
# =============================================================================

def probe_dataloader(
    loader: DataLoader,
    num_batches: int = 2,
    fail_fast: bool = True,
    config: Optional[DataLoaderConfig] = None,
    env: Optional[Dict[str, Any]] = None,
    runtime_settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Probe DataLoader by fetching first N batches and validating structure."""
    # Validate probe arguments
    if isinstance(num_batches, bool) or not isinstance(num_batches, int) or num_batches <= 0:
        raise TypeError(_dataloader_error(
            "probe_args",
            f"num_batches must be positive int, got {type(num_batches).__name__}: {num_batches!r}",
            config=config, env=env,
            resolution="pass a positive integer for num_batches",
        ))
    if not isinstance(fail_fast, bool):
        raise TypeError(_dataloader_error(
            "probe_args",
            f"fail_fast must be bool, got {type(fail_fast).__name__}: {fail_fast!r}",
            config=config, env=env,
        ))

    report: Dict[str, Any] = {
        "ok": True,
        "num_batches_probed": 0,
        "first_batch_ms": None,
        "startup_batch_ms": None,
        "steady_state_avg_ms": None,
        "avg_batch_fetch_ms": None,
        "timing_note": "first batch may include worker startup/tokenizer initialization",
        "batch_sizes": [],
        "fallback_counts": [],
        "fingerprints": [],
        "transfer_readiness": [],
        "errors": [],
    }
    if runtime_settings:
        report["runtime_settings"] = runtime_settings
    fetch_times: List[float] = []
    got_any = False

    try:
        it = iter(loader)
        for i in range(num_batches):
            t0 = time.perf_counter() * 1000.0
            try:
                batch = next(it)
            except StopIteration:
                break
            got_any = True
            fetch_ms = time.perf_counter() * 1000.0 - t0
            fetch_times.append(fetch_ms)
            report["num_batches_probed"] = i + 1

            # Validate batch
            if not isinstance(batch, dict):
                msg = f"Batch {i} is not a dict (got {type(batch).__name__})"
                report["errors"].append(msg)
                report["ok"] = False
                if fail_fast:
                    raise RuntimeError(_dataloader_error(
                        "first_batch_probe", msg,
                        config=config, env=env,
                        resolution="check collate_fn output",
                    ))
                continue

            missing = _REQUIRED_BATCH_KEYS - set(batch.keys())
            if missing:
                msg = f"Batch {i} missing keys: {sorted(missing)}"
                report["errors"].append(msg)
                report["ok"] = False
                if fail_fast:
                    raise RuntimeError(_dataloader_error(
                        "first_batch_probe", msg,
                        config=config, env=env,
                        resolution="check collate_fn output contract",
                    ))
                continue

            # Collect stats
            bs = len(batch.get("sample_ids", []))
            report["batch_sizes"].append(bs)

            meta = batch.get("metadata")
            if isinstance(meta, dict):
                fb = meta.get("fallback_summary", {}).get("fallback_count", 0)
                report["fallback_counts"].append(fb)
                fp = meta.get("batch_fingerprint")
                if fp:
                    report["fingerprints"].append(fp)
                tr = meta.get("transfer_readiness")
                if tr:
                    report["transfer_readiness"].append(tr)

    except RuntimeError as e:
        # Wrap worker/DataLoader runtime failures with rich context
        if "DATALOADER ERROR" in str(e):
            raise  # Already wrapped
        report["ok"] = False
        report["errors"].append(f"RuntimeError: {e}")
        if fail_fast:
            raise RuntimeError(_dataloader_error(
                "first_batch_probe",
                "Worker/DataLoader runtime failure while fetching batch.",
                config=config, env=env, cause=e,
                resolution="retry with execution_mode='debug' or num_workers=0, inspect dataset/collate error",
            )) from e
    except Exception as e:
        report["ok"] = False
        report["errors"].append(f"{type(e).__name__}: {e}")
        if fail_fast:
            raise RuntimeError(_dataloader_error(
                "first_batch_probe",
                "Worker failed while fetching batch.",
                config=config, env=env, cause=e,
                resolution="try debug mode with num_workers=0, inspect dataset/collate errors",
            )) from e

    # Handle no batches produced
    if not got_any and num_batches > 0:
        report["ok"] = False
        report["errors"].append("DataLoader produced no batches")
        if fail_fast:
            raise RuntimeError(_dataloader_error(
                "first_batch_probe",
                "DataLoader produced no batches (dataset may be empty).",
                config=config, env=env,
                resolution="check dataset size and batch_size",
            ))

    if fetch_times:
        report["first_batch_ms"] = round(fetch_times[0], 2)
        report["startup_batch_ms"] = round(fetch_times[0], 2)
        report["avg_batch_fetch_ms"] = round(sum(fetch_times) / len(fetch_times), 2)
        if len(fetch_times) > 1:
            report["steady_state_avg_ms"] = round(
                sum(fetch_times[1:]) / len(fetch_times[1:]), 2
            )

    return report


# =============================================================================
# 10. Build DataLoader
# =============================================================================

def build_dataloader(
    dataset_config: Optional[DatasetConfig] = None,
    collate_config: Optional[CollateConfig] = None,
    loader_config: Optional[DataLoaderConfig] = None,
    dataset: Optional[Any] = None,
) -> Tuple[DataLoader, Dict[str, Any]]:
    """
    Build a production-ready DataLoader for the multimodal pipeline.

    Args:
        dataset_config: Config for MultimodalProductDataset. Ignored if
            ``dataset`` is provided.
        collate_config: Config for BatchCollator.
        loader_config: Config for DataLoader construction.
        dataset: Pre-built dataset instance (bypasses dataset_config).

    Returns:
        (loader, health_report) tuple.
    """
    if loader_config is None:
        loader_config = DataLoaderConfig()
    cfg = loader_config

    t_start = time.perf_counter() * 1000.0 if cfg.enable_timing else None
    report_warnings: List[str] = []

    # -- 1. Environment --
    env = detect_execution_environment()

    # -- 2. Dataset --
    if dataset is None:
        if dataset_config is None:
            dataset_config = DatasetConfig()
        dataset = MultimodalProductDataset(dataset_config)
    num_samples = len(dataset)

    # Effective hint (never mutate caller config)
    effective_size_hint = cfg.dataset_size_hint if cfg.dataset_size_hint is not None else num_samples

    # -- 3. Collate --
    if collate_config is None:
        collate_config = CollateConfig()
    collate_fn = build_collate_fn(collate_config)

    # -- 4. Worker count --
    num_workers = select_num_workers(cfg, env)

    # -- 5. Resolve params --
    params = _resolve_loader_params(cfg, env, num_workers)
    pin_memory = params["pin_memory"]
    persistent_workers = params["persistent_workers"]
    prefetch_factor = params["prefetch_factor"]
    normalization_notes = params["normalization_notes"]

    # -- 6. Risk analysis (uses resolved runtime values) --
    risk_flags = analyze_worker_risk(
        cfg, env, num_workers,
        resolved_pin_memory=pin_memory,
        resolved_persistent_workers=persistent_workers,
        resolved_prefetch_factor=prefetch_factor,
        effective_dataset_size_hint=effective_size_hint,
    )
    if cfg.enable_warnings:
        for r in risk_flags:
            warnings.warn(f"[DATALOADER RISK] {r}", RuntimeWarning, stacklevel=2)
            report_warnings.append(r)
        for n in normalization_notes:
            warnings.warn(f"[DATALOADER NOTE] {n}", RuntimeWarning, stacklevel=2)

    # Batch size vs dataset size
    if num_samples > 0 and cfg.batch_size > num_samples:
        w = (
            f"batch_size={cfg.batch_size} > dataset size={num_samples}. "
            "DataLoader will produce partial batches or empty epochs."
        )
        if cfg.enable_warnings:
            warnings.warn(f"[DATALOADER RISK] {w}", RuntimeWarning, stacklevel=2)
        report_warnings.append(w)

    # -- 7. Worker init --
    worker_init_fn = make_worker_init_fn(cfg.worker_init_seed) if num_workers > 0 else None

    # -- 8. Build DataLoader kwargs --
    dl_kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": cfg.batch_size,
        "shuffle": cfg.shuffle,
        "drop_last": cfg.drop_last,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_fn,
        "timeout": cfg.timeout,
    }
    if worker_init_fn is not None:
        dl_kwargs["worker_init_fn"] = worker_init_fn
    # Only pass these when num_workers > 0
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            dl_kwargs["prefetch_factor"] = prefetch_factor

    loader = DataLoader(**dl_kwargs)

    # -- 9. Prefetch estimate --
    prefetch_est = estimate_prefetch_memory(
        cfg.batch_size, num_workers, prefetch_factor, cfg.approx_sample_mb,
    )

    # -- 10. Health report --
    build_ms = (time.perf_counter() * 1000.0 - t_start) if t_start else None
    health: Dict[str, Any] = {}
    if cfg.enable_health_report:
        health = {
            "execution_mode": cfg.execution_mode,
            "environment": env,
            "dataset": {
                "class": type(dataset).__name__,
                "num_samples": num_samples,
            },
            "collate": {
                "class": "BatchCollator",
            },
            "loader": {
                "batch_size": cfg.batch_size,
                "shuffle": cfg.shuffle,
                "drop_last": cfg.drop_last,
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "persistent_workers": persistent_workers,
                "prefetch_factor": prefetch_factor,
                "timeout": cfg.timeout,
            },
            "prefetch": prefetch_est,
            "risk_flags": risk_flags,
            "warnings": report_warnings,
            "normalization_notes": normalization_notes,
            "effective_dataset_size_hint": effective_size_hint,
            "build_ms": round(build_ms, 2) if build_ms else None,
        }

    # -- 11. Probe --
    if cfg.enable_probe:
        probe_report = probe_dataloader(
            loader, num_batches=2, fail_fast=True,
            config=cfg, env=env,
            runtime_settings={
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "persistent_workers": persistent_workers,
                "prefetch_factor": prefetch_factor,
            },
        )
        health["probe"] = probe_report

    logger.info(
        f"DataLoader ready | mode={cfg.execution_mode} | "
        f"workers={num_workers} | pin={pin_memory} | "
        f"batch={cfg.batch_size} | samples={num_samples}"
    )
    return loader, health


# =============================================================================
# 11. Train/Val convenience
# =============================================================================

def build_train_val_dataloaders(
    train_dataset_config: Optional[DatasetConfig] = None,
    val_dataset_config: Optional[DatasetConfig] = None,
    train_loader_config: Optional[DataLoaderConfig] = None,
    val_loader_config: Optional[DataLoaderConfig] = None,
    collate_config: Optional[CollateConfig] = None,
) -> Dict[str, Any]:
    """
    Build train and validation DataLoaders.

    Does NOT create splits — expects pre-split CSV configs.
    Split creation/leakage detection belongs to future split orchestration.
    """
    if train_loader_config is None:
        train_loader_config = DataLoaderConfig(shuffle=True)
    if val_loader_config is None:
        val_loader_config = DataLoaderConfig(
            shuffle=False, drop_last=False,
            execution_mode=train_loader_config.execution_mode,
        )

    # Leakage guard: fail loudly if both configs resolve to same source
    t_cfg = train_dataset_config or DatasetConfig()
    v_cfg = val_dataset_config or DatasetConfig()
    t_csv = getattr(t_cfg, "csv_filename", None)
    v_csv = getattr(v_cfg, "csv_filename", None)
    if t_csv is not None and v_csv is not None and t_csv == v_csv:
        raise ValueError(_dataloader_error(
            "train_val_loader_build",
            "Train and validation datasets resolve to the same CSV source.",
            resolution=(
                "Provide separate train/val DatasetConfig objects with "
                "distinct csv_filename values or pre-split files."
            ),
        ))

    train_loader, train_report = build_dataloader(
        dataset_config=train_dataset_config,
        collate_config=collate_config,
        loader_config=train_loader_config,
    )
    val_loader, val_report = build_dataloader(
        dataset_config=val_dataset_config,
        collate_config=collate_config,
        loader_config=val_loader_config,
    )
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "train_report": train_report,
        "val_report": val_report,
    }


# =============================================================================
# 12. Smoke Tests
# =============================================================================

if __name__ == "__main__":
    import re, copy

    print("=" * 60)
    print("  data_pipeline/dataloader_factory.py -- smoke test")
    print("=" * 60)

    passed = 0
    total = 0

    def chk(label, ok):
        global passed, total
        total += 1
        if ok:
            passed += 1
        tag = "PASS" if ok else "FAIL"
        print(f"    [{tag}] {label}")

    try:
        # ---- 1. Config validation ----
        print("\n  1. Config validation...")
        cfg = DataLoaderConfig()
        chk("defaults", cfg.batch_size == 16 and cfg.execution_mode == "safe")
        try: DataLoaderConfig(batch_size=0); chk("bs=0", False)
        except TypeError: chk("bs=0", True)
        try: DataLoaderConfig(batch_size=True); chk("bs=bool", False)
        except TypeError: chk("bs=bool", True)
        try: DataLoaderConfig(execution_mode="turbo"); chk("bad mode", False)
        except ValueError: chk("bad mode", True)
        try: DataLoaderConfig(num_workers=-1); chk("neg w", False)
        except TypeError: chk("neg w", True)
        try: DataLoaderConfig(num_workers=True); chk("bool w", False)
        except TypeError: chk("bool w", True)
        try: DataLoaderConfig(prefetch_factor=0); chk("pf=0", False)
        except TypeError: chk("pf=0", True)
        try: DataLoaderConfig(timeout=-1); chk("neg to", False)
        except TypeError: chk("neg to", True)
        try: DataLoaderConfig(max_prefetch_memory_mb=True); chk("bool mem", False)
        except TypeError: chk("bool mem", True)
        try: DataLoaderConfig(dataset_size_hint=True); chk("bool hint", False)
        except TypeError: chk("bool hint", True)
        chk("None w ok", DataLoaderConfig(num_workers=None).num_workers is None)

        # ---- 2. Environment ----
        print("\n  2. Environment...")
        env = detect_execution_environment()
        chk("platform", isinstance(env["platform"], str))
        chk("cpu", isinstance(env["cpu_count"], int) and env["cpu_count"] >= 1)
        chk("cuda", isinstance(env["cuda_available"], bool))
        chk("is_win", isinstance(env["is_windows"], bool))
        chk("is_colab", isinstance(env["is_colab"], bool))

        # ---- 3. Workers ----
        print("\n  3. Workers...")
        chk("debug=0", select_num_workers(DataLoaderConfig(execution_mode="debug"), env) == 0)
        chk("override", select_num_workers(DataLoaderConfig(num_workers=3), env) == 3)
        chk("safe>=0", select_num_workers(DataLoaderConfig(execution_mode="safe"), env) >= 0)
        chk("colab<=2", select_num_workers(DataLoaderConfig(execution_mode="colab"), env) <= 2)

        # ---- 4. Prefetch estimate ----
        print("\n  4. Prefetch estimate...")
        est = estimate_prefetch_memory(16, 2, 2, 2.0)
        chk("batches=4", est["estimated_prefetch_batches"] == 4)
        chk("mem=128", est["estimated_prefetch_memory_mb"] == 128.0)
        chk("heuristic", est["estimate_type"] == "heuristic")
        chk("not exact", est["is_exact"] is False)
        chk("assumptions", "does_not_include" in est["assumptions"])
        est0 = estimate_prefetch_memory(16, 0, None, 2.0)
        chk("w=0 mem=0", est0["estimated_prefetch_memory_mb"] == 0.0)

        # ---- 5. Risk analysis (resolved values) ----
        print("\n  5. Risk analysis...")
        risks = analyze_worker_risk(
            DataLoaderConfig(num_workers=0), env, 0,
            resolved_pin_memory=False, resolved_persistent_workers=False)
        chk("w=0 risk", any("num_workers=0" in r for r in risks))
        risks2 = analyze_worker_risk(
            DataLoaderConfig(batch_size=128), env, 2,
            resolved_pin_memory=False, resolved_persistent_workers=True,
            resolved_prefetch_factor=2, effective_dataset_size_hint=5000)
        chk("large batch", any("generalization" in r for r in risks2))
        chk("persist risk", any("persistent_workers" in r for r in risks2))

        # ---- 6. Normalization notes ----
        print("\n  6. Normalization notes...")
        p = _resolve_loader_params(DataLoaderConfig(execution_mode="debug"), env, 0)
        chk("notes list", isinstance(p["normalization_notes"], list))
        chk("debug pf=None", p["prefetch_factor"] is None)
        chk("debug pw=False", p["persistent_workers"] is False)

        p2 = _resolve_loader_params(
            DataLoaderConfig(persistent_workers=True, prefetch_factor=4), env, 0)
        chk("pw note", any("persistent_workers ignored" in n for n in p2["normalization_notes"]))
        chk("pf note", any("prefetch_factor ignored" in n for n in p2["normalization_notes"]))

        p3 = _resolve_loader_params(DataLoaderConfig(execution_mode="safe"), env, 2)
        chk("safe pf=2", p3["prefetch_factor"] == 2)
        chk("safe pw=True", p3["persistent_workers"] is True)

        # Throughput on Colab
        colab_env = {**env, "is_colab": True}
        p4 = _resolve_loader_params(DataLoaderConfig(execution_mode="throughput"), colab_env, 4)
        chk("colab tp pf<=2", p4["prefetch_factor"] <= 2)

        # ---- 7. Probe arg validation ----
        print("\n  7. Probe arg validation...")
        class _DummyDS(torch.utils.data.Dataset):
            def __len__(self): return 4
            def __getitem__(self, i): return {}
        dl = DataLoader(_DummyDS(), batch_size=2)
        try: probe_dataloader(dl, num_batches=0); chk("nb=0", False)
        except TypeError: chk("nb=0", True)
        try: probe_dataloader(dl, fail_fast="yes"); chk("ff=str", False)
        except TypeError: chk("ff=str", True)

        # ---- 8. Synthetic probe + timing ----
        print("\n  8. Synthetic probe...")
        class _SynDS(torch.utils.data.Dataset):
            def __len__(self): return 8
            def __getitem__(self, idx):
                return {
                    "sample_id": f"{idx}:S{idx}", "row_index": idx, "asin": f"S{idx}",
                    "image": torch.randn(3, 224, 224), "image_path": f"/s/{idx}.jpg",
                    "raw_text": f"t{idx}", "sanitized_text": f"t{idx}",
                    "input_ids": torch.randint(0, 100, (1, 16), dtype=torch.long),
                    "attention_mask": torch.ones(1, 16, dtype=torch.long),
                    "tabular": torch.randn(2), "rating": torch.tensor(3.0),
                    "metadata": {"fallback_used": False, "fallback_reasons": [],
                                 "missing_modalities": [],
                                 "trace": [{"stage": "s", "status": "ok"}]},
                }
        syn_ds = _SynDS()
        syn_col = build_collate_fn(CollateConfig(expected_image_size=(224, 224)))
        syn_dl = DataLoader(syn_ds, batch_size=4, shuffle=False, num_workers=0, collate_fn=syn_col)
        prb = probe_dataloader(syn_dl, num_batches=2, fail_fast=True)
        chk("probe ok", prb["ok"] is True)
        chk("startup_ms", prb["startup_batch_ms"] is not None)
        chk("timing_note", "first batch" in prb["timing_note"])
        chk("batches=2", prb["num_batches_probed"] == 2)

        # ---- 9. Health report + config immutability ----
        print("\n  9. Health report + immutability...")
        orig_cfg = DataLoaderConfig(
            execution_mode="debug", enable_probe=False,
            enable_warnings=False, dataset_size_hint=None,
        )
        orig_hint = orig_cfg.dataset_size_hint  # None
        _, rpt = build_dataloader(
            loader_config=orig_cfg, dataset=syn_ds,
            collate_config=CollateConfig(expected_image_size=(224, 224)),
        )
        chk("cfg not mutated", orig_cfg.dataset_size_hint is orig_hint)
        chk("mode", rpt.get("execution_mode") == "debug")
        chk("env", "environment" in rpt)
        chk("loader", "loader" in rpt)
        chk("prefetch", "prefetch" in rpt)
        chk("norm notes", isinstance(rpt.get("normalization_notes"), list))
        chk("eff hint", rpt.get("effective_dataset_size_hint") is not None)
        chk("risk list", isinstance(rpt.get("risk_flags"), list))
        chk("heuristic", rpt["prefetch"]["estimate_type"] == "heuristic")

        # ---- 10. Source safety ----
        print("\n  10. Source safety...")
        src = open(__file__, encoding="utf-8").read()
        prod = src.split('if __name__')[0]
        pc = "\n".join(l for l in prod.splitlines() if not l.strip().startswith("#"))
        chk("no .cuda()", ".cuda()" not in pc)
        chk("no .to(device)", ".to(device)" not in pc)
        chk("no .pin_memory()", ".pin_memory()" not in pc)
        chk("no from models", "from models" not in pc)
        chk("no from train", "from train" not in pc)
        chk("no BatchCollator import", "BatchCollator" not in prod.split("from data_pipeline.collate")[1].split("\n")[0] if "from data_pipeline.collate" in prod else True)

        # ---- Summary ----
        print(f"\n{'='*60}")
        tag = "PASS" if passed == total else "FAIL"
        print(f"  [{tag}]  {passed}/{total} checks passed")
        print("=" * 60)
        if passed < total:
            sys.exit(1)

    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
