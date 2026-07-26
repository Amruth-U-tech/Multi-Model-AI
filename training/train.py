# =============================================================================
# training/train.py
# Training Bootloader + Mission Planner - Multimodal AI Pipeline
# =============================================================================
#
# Purpose:
#   Application-level entry point that assembles the full training pipeline.
#   Discovers datasets, selects and validates them, builds in-memory splits,
#   constructs models, optimizer, scheduler, evaluator, and trainer -- then
#   freezes an immutable ExecutionPlan.
#
# Philosophy:
#   TrainConfig declares.  train.py assembles and verifies.  Trainer trains.
#   Subsystems own their own domain.
#
# Application Contracts:
#   Exactly one TrainConfig
#   Exactly one RunContext
#   Exactly one registry snapshot
#   Exactly one dataset selection
#   Exactly one base dataset
#   Exactly one train subset
#   Exactly one validation subset
#   Exactly one train DataLoader
#   Exactly one validation DataLoader
#   Exactly one model bundle (ModuleDict)
#   Exactly one optimizer
#   Exactly one scheduler
#   Exactly one evaluator
#   Exactly one trainer
#   Exactly one execution plan
#   Exactly one active training session
#
# Lifecycle:
#   ExecutionPlan: Build -> Freeze -> Read Only -> Destroy at process exit
#
# Public API:
#   main()
#   build_execution_plan(...)
#   run_training(...)
#   print_execution_plan(...)
#   perform_preflight(...)
#   perform_dry_run(...)
#   write_run_manifest(...)
#   run_smoke_tests()
#
# What this file does NOT do:
#   - Train models (Trainer owns that)
#   - Collate batches (collate.py owns that)
#   - Transfer tensors to GPU (Trainer owns that)
#   - Compute losses or metrics (evaluation.py owns that)
#   - Step optimizer or scheduler (Trainer owns that)
#   - Write checkpoints (Trainer owns that)
#
# Usage:
#   python training/train.py --smoke       # smoke tests only
#   python training/train.py --train       # full training
#   python training/train.py --plan-only   # print plan, no training
#   python training/train.py --dry-run     # plan + one-batch dry run
#   python training/train.py --help        # show usage
# =============================================================================

from __future__ import annotations

import os
import sys
import json
import time
import math
import platform
import logging
import datetime
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# -- Project root bootstrap ----------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _THIS_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

import copy
import random
import torch
import torch.nn as nn
from torch.utils.data import Subset

from configs.paths import (
    PROJECT_ROOT, PREPROCESSED_DATASET_DIR, CHECKPOINT_DIR,
    EXPERIMENT_DIR, LOG_DIR,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Constants & Schema
# =============================================================================

MANIFEST_SCHEMA_VERSION = 1


# =============================================================================
# Global Reproducibility Seeding
# =============================================================================

def _enforce_determinism(seed: int, deterministic: bool) -> None:
    """Enforce global reproducibility seeding.

    Called once before any dataset, DataLoader, or model construction.
    Sets Python, NumPy, PyTorch, and CUDA seeds. Configures CuDNN
    deterministic policy when requested.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        import numpy as np
        np.random.seed(seed % (2**32))
    except ImportError:
        pass

    if deterministic:
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                # Older PyTorch without warn_only
                pass

_REQUIRED_MODEL_KEYS = frozenset({
    "image_encoder", "text_encoder", "tabular_encoder", "fusion_model",
})

# Exit codes
EXIT_SUCCESS = 0
EXIT_CONFIG_FAILURE = 1
EXIT_DATASET_FAILURE = 2
EXIT_MODEL_FAILURE = 3
EXIT_TRAINING_FAILURE = 4
EXIT_INTERRUPT = 130


# =============================================================================
# TrainAppError
# =============================================================================

class TrainAppError(RuntimeError):
    """Structured application-level training error.

    Every error identifies: stage, subsystem, received, expected, resolution.
    Failure order is deterministic:
    config -> run_context -> registry -> selection -> dataset ->
    split -> loaders -> models -> optimizer -> scheduler ->
    evaluator -> trainer -> preflight -> dry_run -> manifest -> training
    """

    def __init__(
        self,
        stage: str,
        subsystem: str,
        *,
        received: str = "",
        expected: str = "",
        resolution: str = "",
    ):
        self.stage = stage
        self.subsystem = subsystem
        self.received = received
        self.expected = expected
        self.resolution = resolution
        lines = [
            "",
            "=" * 64,
            "  TRAINING ERROR",
            "=" * 64,
            f"  Stage     : {stage}",
            f"  Subsystem : {subsystem}",
            f"  Received  : {received}",
            f"  Expected  : {expected}",
        ]
        if resolution:
            lines.append(f"  Resolution: {resolution}")
        lines.append("=" * 64)
        super().__init__("\n".join(lines))


# =============================================================================
# Internal Frozen Dataclasses
# =============================================================================

@dataclass(frozen=True)
class _RegistrySnapshot:
    """Immutable representation of dataset discovery results.

    Owner: train.py | Lifetime: entire application | Mutable: No
    Consumers: _DatasetSelection, _ExecutionPlan, Manifest
    """
    discovered_at: str
    dataset_count: int
    datasets: tuple
    dataset_names: Tuple[str, ...]
    dataset_files: Tuple[str, ...]
    total_rows: int


@dataclass(frozen=True)
class _DatasetSelection:
    """Immutable representation of user intent resolved into selected datasets.

    Owner: train.py | Lifetime: entire application | Mutable: No
    Consumers: _ExecutionPlan, dataset construction
    """
    mode: str  # "single" | "manual_multi" | "full"
    requested: Tuple[str, ...]
    selected_names: Tuple[str, ...]
    selected_files: Tuple[str, ...]
    ignored_train_datasets: Tuple[str, ...]
    total_rows: int
    duplicate_identity_count: int


@dataclass(frozen=True)
class _SplitPlan:
    """Immutable representation of deterministic in-memory split.

    Owner: train.py | Lifetime: entire application | Mutable: No
    Consumers: _ExecutionPlan, Subset construction
    """
    validation_split: float
    seed: int
    dataset_size: int
    train_size: int
    validation_size: int
    train_indices: Tuple[int, ...]
    validation_indices: Tuple[int, ...]


@dataclass(frozen=True)
class _ExecutionMetadata:
    """Lightweight runtime snapshot used by the execution plan.

    Owner: train.py | Lifetime: entire application | Mutable: No
    Consumers: _ExecutionPlan, Manifest
    """
    created_at: str
    python_version: str
    platform: str
    torch_version: str
    cuda_available: bool
    device: str
    mixed_precision: bool
    seed: int


@dataclass(frozen=True)
class _PreflightResult:
    """Preflight diagnostics result.

    Owner: train.py | Lifetime: entire application | Mutable: No
    Consumers: Manifest, terminal display
    """
    status: str = "not_implemented"  # "passed" | "warnings" | "failed" | "not_implemented"
    warnings: Tuple[str, ...] = ()
    errors: Tuple[str, ...] = ()
    verified_subsystems: Tuple[str, ...] = ()
    elapsed_ms: float = 0.0


@dataclass(frozen=True)
class _DryRunResult:
    """One-batch dry run result.

    Owner: train.py | Lifetime: entire application | Mutable: No
    Consumers: Manifest, terminal display

    batch_shape_summary uses Tuple[Tuple[str, str], ...] for true
    immutability instead of a mutable Dict.
    """
    status: str  # "passed" | "failed" | "skipped"
    batch_shape_summary: Tuple[Tuple[str, str], ...] = ()
    forward_ok: bool = False
    loss_ok: bool = False
    elapsed_ms: float = 0.0
    error: str = ""


class _ExecutionPlan:
    """Effectively immutable execution plan.

    Owner: train.py | Lifetime: entire application

    Immutable topology (frozen after construction):
        registry, selection, split, metadata, config, run_context,
        train_loader_report, validation_loader_report

    Mutable runtime handles (intentionally mutable by Trainer/training):
        dataset, subsets, loaders, model_bundle, optimizer,
        scheduler, evaluator, trainer
        These are not frozen because Trainer needs to mutate model weights,
        optimizer state, scheduler state, and evaluator state during training.

    Lifecycle: Build -> Freeze -> Read Only (topology) -> Destroy at exit
    """

    __slots__ = (
        "config", "run_context", "metadata", "registry", "selection",
        "split", "base_dataset", "train_subset", "validation_subset",
        "train_loader", "validation_loader", "train_loader_report",
        "validation_loader_report", "model_bundle", "optimizer",
        "scheduler", "evaluator", "trainer", "status",
        "preflight", "dry_run", "model_dim_summary",
        "_frozen",
    )

    def __init__(self, **kwargs: Any):
        for k, v in kwargs.items():
            object.__setattr__(self, k, v)
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            # Allow mutation of runtime handles only
            _MUTABLE = {
                "base_dataset", "train_subset", "validation_subset",
                "train_loader", "validation_loader", "model_bundle",
                "optimizer", "scheduler", "evaluator", "trainer",
                "status", "preflight", "dry_run",
            }
            if name not in _MUTABLE:
                raise AttributeError(
                    f"ExecutionPlan topology is frozen. Cannot modify '{name}'."
                )
        object.__setattr__(self, name, value)

    def __repr__(self) -> str:
        return f"<_ExecutionPlan status={self.status}>"


# =============================================================================
# Private Assembly Functions
# =============================================================================

def _build_metadata(config: Any, run_context: Any) -> _ExecutionMetadata:
    """Build a lightweight runtime metadata snapshot."""
    return _ExecutionMetadata(
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        python_version=platform.python_version(),
        platform=platform.platform(),
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        device=str(run_context.device),
        mixed_precision=config.mixed_precision,
        seed=config.seed,
    )


def _discover_registry() -> _RegistrySnapshot:
    """Discover datasets exactly once. Convert to immutable snapshot."""
    from data_pipeline.dataset_registry import discover_datasets

    t0 = time.monotonic()
    descriptors = discover_datasets()
    elapsed = time.monotonic() - t0

    if not descriptors:
        raise TrainAppError(
            "registry", "discovery",
            received="0 datasets",
            expected="at least 1 discoverable dataset",
            resolution=f"Place preprocessed CSVs in {PREPROCESSED_DATASET_DIR}.",
        )

    names: List[str] = []
    files: List[str] = []
    total_rows = 0
    seen_files: set = set()

    for i, d in enumerate(descriptors):
        if not hasattr(d, "filename") or not hasattr(d, "row_count"):
            raise TrainAppError(
                "registry", "discovery",
                received=f"descriptor[{i}] type={type(d).__name__}",
                expected="DatasetDescriptor with filename and row_count",
                resolution="dataset_registry.discover_datasets() returned invalid data.",
            )
        fname = d.filename
        if fname in seen_files:
            raise TrainAppError(
                "registry", "discovery",
                received=f"duplicate filename '{fname}'",
                expected="unique filenames across discovered datasets",
                resolution=f"Remove duplicate CSV files from {PREPROCESSED_DATASET_DIR}.",
            )
        seen_files.add(fname)
        name = fname.rsplit(".csv", 1)[0] if fname.endswith(".csv") else fname
        names.append(name)
        files.append(fname)
        total_rows += d.row_count

    logger.info(
        "Dataset discovery: %d datasets, %d total rows (%.0f ms)",
        len(descriptors), total_rows, elapsed * 1000,
    )

    return _RegistrySnapshot(
        discovered_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        dataset_count=len(descriptors),
        datasets=tuple(descriptors),
        dataset_names=tuple(names),
        dataset_files=tuple(files),
        total_rows=total_rows,
    )


def _select_datasets(config: Any, registry: _RegistrySnapshot) -> _DatasetSelection:
    """Select datasets based on config intent. Validate identity."""
    # V1: validation_dataset_name is not supported (split-based only)
    if getattr(config, 'validation_dataset_name', None) is not None:
        raise TrainAppError(
            "selection", "validation_dataset_name",
            received=f"validation_dataset_name={config.validation_dataset_name!r}",
            expected="None for V1 split-based validation",
            resolution=(
                "This train.py version uses validation_split with in-memory Subset routing. "
                "Unset validation_dataset_name or implement explicit validation dataset routing in a future pass."
            ),
        )

    ignored: Tuple[str, ...] = ()

    if config.train_all:
        mode = "full"
        requested = registry.dataset_names
        selected_names = registry.dataset_names
        selected_files = registry.dataset_files
        if config.train_datasets:
            ignored = config.train_datasets
            logger.info("train_all=True: ignoring train_datasets=%s", config.train_datasets)
    elif config.train_datasets:
        mode = "manual_multi"
        requested = config.train_datasets
        sn, sf = [], []
        for name in config.train_datasets:
            matched = False
            for rn, rf in zip(registry.dataset_names, registry.dataset_files):
                if name == rn or name == rf:
                    sn.append(rn)
                    sf.append(rf)
                    matched = True
                    break
            if not matched:
                available = ", ".join(registry.dataset_names[:10])
                raise TrainAppError(
                    "selection", "dataset_resolution",
                    received=f"unknown dataset '{name}'",
                    expected=f"one of: {available}",
                    resolution="Check dataset name or filename spelling.",
                )
        selected_names = tuple(sn)
        selected_files = tuple(sf)
    else:
        mode = "single"
        requested = (config.dataset_name,)
        matched = False
        selected_names = ()
        selected_files = ()
        for rn, rf in zip(registry.dataset_names, registry.dataset_files):
            if config.dataset_name == rn or config.dataset_name == rf:
                selected_names = (rn,)
                selected_files = (rf,)
                matched = True
                break
        if not matched:
            available = ", ".join(registry.dataset_names[:10])
            raise TrainAppError(
                "selection", "dataset_resolution",
                received=f"dataset_name='{config.dataset_name}'",
                expected=f"one of: {available}",
                resolution="Set dataset_name to a known dataset.",
            )

    # Check ASIN collisions in multi-source mode
    dup_count = 0
    if len(selected_files) > 1:
        from data_pipeline.dataset_registry import resolve_multi_source
        multi_desc = resolve_multi_source(selected_files)
        dup_count = multi_desc.cross_source_asin_collisions
        if dup_count > 0:
            examples = getattr(multi_desc, "collision_examples", [])[:5]
            raise TrainAppError(
                "selection", "identity_validation",
                received=f"{dup_count} duplicate ASINs across selected datasets",
                expected="zero cross-source ASIN collisions",
                resolution=(
                    f"Selected datasets share {dup_count} ASINs. "
                    f"Examples: {examples}. "
                    "Remove overlapping sources or implement a dedup policy."
                ),
            )

    total_rows = 0
    for sf in selected_files:
        for d in registry.datasets:
            if d.filename == sf:
                total_rows += d.row_count
                break

    return _DatasetSelection(
        mode=mode, requested=tuple(requested),
        selected_names=selected_names, selected_files=selected_files,
        ignored_train_datasets=ignored, total_rows=total_rows,
        duplicate_identity_count=dup_count,
    )


def _build_dataset(selection: _DatasetSelection) -> Any:
    """Build the base dataset through existing DatasetConfig."""
    from data_pipeline.dataset import DatasetConfig, build_dataset

    if len(selection.selected_files) == 1:
        ds_cfg = DatasetConfig(dataset_name=selection.selected_names[0], mode="train")
    else:
        ds_cfg = DatasetConfig(source_files=list(selection.selected_files), mode="train")

    try:
        dataset = build_dataset(ds_cfg)
    except Exception as exc:
        raise TrainAppError(
            "dataset", "construction",
            received=str(exc)[:300],
            expected="valid dataset construction",
            resolution="Check dataset files, schema, and configuration.",
        ) from exc

    if len(dataset) == 0:
        raise TrainAppError(
            "dataset", "construction",
            received="dataset size = 0",
            expected="non-empty dataset",
            resolution="Selected datasets contain no valid samples.",
        )
    return dataset


def _build_split(config: Any, dataset: Any) -> _SplitPlan:
    """Build deterministic in-memory split."""
    dataset_size = len(dataset)
    val_size = max(1, int(math.floor(dataset_size * config.validation_split)))
    train_size = dataset_size - val_size

    if train_size < 1:
        raise TrainAppError(
            "split", "split_validation",
            received=f"train_size={train_size} (dataset={dataset_size}, val_split={config.validation_split})",
            expected="train_size >= 1",
            resolution="Increase dataset size or decrease validation_split.",
        )
    if val_size < 1:
        raise TrainAppError(
            "split", "split_validation",
            received=f"val_size={val_size}",
            expected="val_size >= 1",
            resolution="Increase dataset size or increase validation_split.",
        )

    generator = torch.Generator().manual_seed(config.seed)
    perm = torch.randperm(dataset_size, generator=generator)

    train_indices = tuple(perm[:train_size].tolist())
    val_indices = tuple(perm[train_size:].tolist())

    # Split integrity checks -- explicit errors, not assertions
    overlap = set(train_indices) & set(val_indices)
    if overlap:
        raise TrainAppError(
            "split", "integrity",
            received=f"{len(overlap)} overlapping indices (examples: {sorted(list(overlap))[:5]})",
            expected="disjoint train/validation index sets",
            resolution="Bug in split generation. Check dataset size, seed, and randperm logic.",
        )
    if len(train_indices) + len(val_indices) != dataset_size:
        raise TrainAppError(
            "split", "integrity",
            received=f"train({len(train_indices)}) + val({len(val_indices)}) = {len(train_indices) + len(val_indices)}",
            expected=f"total = {dataset_size}",
            resolution="Index coverage mismatch. Check split generation logic.",
        )
    oob_train = [i for i in train_indices if i < 0 or i >= dataset_size]
    if oob_train:
        raise TrainAppError(
            "split", "integrity",
            received=f"train indices out of bounds: {oob_train[:5]}",
            expected=f"all indices in [0, {dataset_size})",
            resolution="Check split generation logic.",
        )
    oob_val = [i for i in val_indices if i < 0 or i >= dataset_size]
    if oob_val:
        raise TrainAppError(
            "split", "integrity",
            received=f"validation indices out of bounds: {oob_val[:5]}",
            expected=f"all indices in [0, {dataset_size})",
            resolution="Check split generation logic.",
        )

    return _SplitPlan(
        validation_split=config.validation_split, seed=config.seed,
        dataset_size=dataset_size, train_size=train_size,
        validation_size=val_size, train_indices=train_indices,
        validation_indices=val_indices,
    )


def _build_loaders(
    config: Any, train_subset: Subset, val_subset: Subset,
) -> Tuple[Any, Dict[str, Any], Any, Dict[str, Any]]:
    """Build train and validation DataLoaders through the existing factory."""
    from data_pipeline.dataloader_factory import DataLoaderConfig, build_dataloader
    from data_pipeline.collate import CollateConfig

    train_cfg = DataLoaderConfig(
        batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, dataset_size_hint=len(train_subset),
        worker_init_seed=config.seed,
    )
    val_cfg = DataLoaderConfig(
        batch_size=config.batch_size, shuffle=False, drop_last=False,
        num_workers=config.num_workers, dataset_size_hint=len(val_subset),
        worker_init_seed=config.seed,
    )
    collate_cfg = CollateConfig()

    try:
        train_loader, train_report = build_dataloader(
            dataset=train_subset, loader_config=train_cfg, collate_config=collate_cfg,
        )
    except Exception as exc:
        raise TrainAppError(
            "loaders", "train_loader",
            received=str(exc)[:200], expected="valid train DataLoader",
            resolution="Check dataset and collate configuration.",
        ) from exc

    try:
        val_loader, val_report = build_dataloader(
            dataset=val_subset, loader_config=val_cfg, collate_config=collate_cfg,
        )
    except Exception as exc:
        raise TrainAppError(
            "loaders", "validation_loader",
            received=str(exc)[:200], expected="valid validation DataLoader",
            resolution="Check dataset and collate configuration.",
        ) from exc

    # Snapshot reports as immutable dicts (deep copy to prevent mutation)
    import copy
    return train_loader, copy.deepcopy(train_report), val_loader, copy.deepcopy(val_report)


def _build_model_bundle(
    *,
    tabular_input_dim: int,
    tabular_columns: Tuple[str, ...],
) -> Tuple[nn.ModuleDict, Dict[str, Any]]:
    """Build all model components and assemble into ModuleDict.

    Args:
        tabular_input_dim: Number of raw tabular features from dataset.
        tabular_columns: Column names from dataset config.

    Returns (model_bundle, dimension_summary).
    """
    # Validate tabular contract
    if isinstance(tabular_input_dim, bool) or not isinstance(tabular_input_dim, int) or tabular_input_dim <= 0:
        raise TrainAppError(
            "models", "tabular_contract",
            received=f"tabular_input_dim={tabular_input_dim!r} (type={type(tabular_input_dim).__name__})",
            expected="positive int",
            resolution="Ensure dataset config has valid tabular_columns.",
        )
    if not isinstance(tabular_columns, tuple) or len(tabular_columns) != tabular_input_dim:
        raise TrainAppError(
            "models", "tabular_contract",
            received=f"tabular_columns={tabular_columns!r}, len={len(tabular_columns) if isinstance(tabular_columns, tuple) else 'N/A'}",
            expected=f"tuple of {tabular_input_dim} non-empty strings",
            resolution="tabular_columns must match tabular_input_dim.",
        )
    for col in tabular_columns:
        if not isinstance(col, str) or not col.strip():
            raise TrainAppError(
                "models", "tabular_contract",
                received=f"invalid column name: {col!r}",
                expected="non-empty string",
            )

    from models.image_encoder import ImageEncoderConfig, build_encoder
    from models.text_encoder import TextEncoderConfig, build_text_encoder
    from models.tabular_encoder import TabularEncoderConfig, build_tabular_encoder
    from models.fusion import FusionConfig, FusionModel

    try:
        image_encoder = build_encoder(ImageEncoderConfig())
    except Exception as exc:
        raise TrainAppError(
            "models", "image_encoder", received=str(exc)[:200],
            expected="valid ImageEncoder",
            resolution=(
                "Check internet/cache availability. Pre-download model weights or "
                "ensure cache paths are populated. Run models/image_encoder.py smoke test."
            ),
        ) from exc

    try:
        text_encoder = build_text_encoder(TextEncoderConfig())
    except Exception as exc:
        raise TrainAppError(
            "models", "text_encoder", received=str(exc)[:200],
            expected="valid TextEncoder",
            resolution=(
                "Check internet/cache availability. Pre-download model weights or "
                "ensure cache paths are populated. Run models/text_encoder.py smoke test."
            ),
        ) from exc

    try:
        tabular_encoder = build_tabular_encoder(
            TabularEncoderConfig(input_dim=tabular_input_dim)
        )
    except Exception as exc:
        raise TrainAppError(
            "models", "tabular_encoder", received=str(exc)[:200],
            expected="valid TabularEncoder",
            resolution="Check tabular encoder config and dependencies.",
        ) from exc

    fusion_cfg = FusionConfig()
    try:
        fusion_model = FusionModel(fusion_cfg)
    except Exception as exc:
        raise TrainAppError(
            "models", "fusion_model", received=str(exc)[:200],
            expected="valid FusionModel",
            resolution="Check fusion config and encoder embedding dims.",
        ) from exc

    # -- Model dimension contract validation -----------------------------------
    dim_summary: Dict[str, Any] = {}
    _dim_errors: List[str] = []

    if hasattr(image_encoder, "get_embedding_dim"):
        img_dim = image_encoder.get_embedding_dim()
        dim_summary["image_encoder_dim"] = img_dim
        if img_dim != fusion_cfg.image_dim:
            _dim_errors.append(
                f"image_encoder.get_embedding_dim()={img_dim} != FusionConfig.image_dim={fusion_cfg.image_dim}"
            )

    if hasattr(text_encoder, "get_embedding_dim"):
        txt_dim = text_encoder.get_embedding_dim()
        dim_summary["text_encoder_dim"] = txt_dim
        if txt_dim != fusion_cfg.text_dim:
            _dim_errors.append(
                f"text_encoder.get_embedding_dim()={txt_dim} != FusionConfig.text_dim={fusion_cfg.text_dim}"
            )

    if hasattr(tabular_encoder, "get_embedding_dim"):
        tab_dim = tabular_encoder.get_embedding_dim()
        dim_summary["tabular_encoder_dim"] = tab_dim
        if tab_dim != fusion_cfg.tabular_dim:
            _dim_errors.append(
                f"tabular_encoder.get_embedding_dim()={tab_dim} != FusionConfig.tabular_dim={fusion_cfg.tabular_dim}"
            )

    dim_summary["fusion_image_dim"] = fusion_cfg.image_dim
    dim_summary["fusion_text_dim"] = fusion_cfg.text_dim
    dim_summary["fusion_tabular_dim"] = fusion_cfg.tabular_dim
    dim_summary["fusion_dim"] = fusion_cfg.fusion_dim
    dim_summary["tabular_input_dim"] = tabular_input_dim
    dim_summary["tabular_columns"] = tabular_columns
    dim_summary["tabular_encoder_expected_input_dim"] = tabular_encoder.config.input_dim
    dim_summary["matched"] = len(_dim_errors) == 0

    if _dim_errors:
        raise TrainAppError(
            "models", "dimension_contract",
            received="; ".join(_dim_errors),
            expected="encoder embedding dims match FusionConfig modality dims",
            resolution="Align encoder latent_dim with FusionConfig image/text/tabular_dim.",
        )

    # Assemble bundle
    bundle = nn.ModuleDict({
        "image_encoder": image_encoder,
        "text_encoder": text_encoder,
        "tabular_encoder": tabular_encoder,
        "fusion_model": fusion_model,
    })

    missing = _REQUIRED_MODEL_KEYS - set(bundle.keys())
    if missing:
        raise TrainAppError(
            "models", "model_bundle",
            received=f"keys={sorted(bundle.keys())}",
            expected=f"required keys: {sorted(_REQUIRED_MODEL_KEYS)}",
            resolution=f"Missing: {sorted(missing)}.",
        )

    for key, mod in bundle.items():
        if not isinstance(mod, nn.Module):
            raise TrainAppError(
                "models", "model_bundle",
                received=f"bundle['{key}'] is {type(mod).__name__}",
                expected="nn.Module",
            )

    total_params = sum(p.numel() for p in bundle.parameters())
    trainable_params = sum(p.numel() for p in bundle.parameters() if p.requires_grad)
    dim_summary["total_params"] = total_params
    dim_summary["trainable_params"] = trainable_params

    if trainable_params == 0:
        raise TrainAppError(
            "models", "model_bundle",
            received="0 trainable parameters",
            expected="total trainable parameters > 0",
            resolution="All model parameters are frozen. Unfreeze at least some.",
        )

    logger.info("Model bundle: %d total params, %d trainable", total_params, trainable_params)
    return bundle, dim_summary


def _build_training_stack(
    config: Any, run_context: Any, model_bundle: nn.ModuleDict,
    train_loader: Any, val_loader: Any,
) -> Tuple[Any, Any, Any, Any]:
    """Build optimizer, scheduler, evaluator, and trainer."""
    from training.optimizer import build_optimizer
    from training.scheduler import build_scheduler
    from training.evaluation import build_evaluator
    from training.trainer import build_trainer

    try:
        optimizer = build_optimizer(config=config, run_context=run_context, model=model_bundle)
    except Exception as exc:
        raise TrainAppError(
            "optimizer", "build", received=str(exc)[:200],
            expected="valid optimizer", resolution="Check optimizer config.",
        ) from exc

    try:
        scheduler = build_scheduler(config=config, run_context=run_context, optimizer=optimizer)
    except Exception as exc:
        raise TrainAppError(
            "scheduler", "build", received=str(exc)[:200],
            expected="valid scheduler", resolution="Check scheduler config.",
        ) from exc

    try:
        evaluator = build_evaluator(config, run_context)
    except Exception as exc:
        raise TrainAppError(
            "evaluator", "build", received=str(exc)[:200],
            expected="valid evaluator", resolution="Check evaluator config.",
        ) from exc

    try:
        trainer = build_trainer(
            config=config, run_context=run_context, model_bundle=model_bundle,
            optimizer=optimizer, scheduler=scheduler, evaluator=evaluator,
            train_loader=train_loader, val_loader=val_loader,
        )
    except Exception as exc:
        raise TrainAppError(
            "trainer", "build", received=str(exc)[:200],
            expected="valid trainer", resolution="Check trainer config and dependencies.",
        ) from exc

    return optimizer, scheduler, evaluator, trainer


# =============================================================================
# Preflight Diagnostics
# =============================================================================

def perform_preflight(plan: _ExecutionPlan) -> _PreflightResult:
    """Run bootloader-level preflight diagnostics.

    Validates topology integrity without iterating DataLoaders, starting
    worker processes, running model forward, moving tensors, or writing files.

    Does NOT train, mutate history, step optimizer/scheduler, or checkpoint.
    """
    t0 = time.monotonic()
    warnings: List[str] = []
    errors: List[str] = []
    verified: List[str] = []

    # Config frozen
    if hasattr(plan.config, '_frozen') and not plan.config._frozen:
        errors.append("Config is not frozen.")
    else:
        verified.append("config")

    # Run context
    if plan.run_context is None:
        errors.append("Run context is None.")
    else:
        verified.append("run_context")

    # Registry has at least one dataset
    if plan.registry.dataset_count == 0:
        errors.append("No datasets discovered.")
    else:
        verified.append(f"registry({plan.registry.dataset_count} datasets)")

    # Selected dataset count > 0
    if len(plan.selection.selected_names) == 0:
        errors.append("No datasets selected.")
    else:
        verified.append(f"selection({plan.selection.mode})")

    # Split sizes match dataset size
    total_from_split = plan.split.train_size + plan.split.validation_size
    if total_from_split != plan.split.dataset_size:
        errors.append(
            f"Split sizes ({plan.split.train_size}+{plan.split.validation_size})"
            f" != dataset size ({plan.split.dataset_size})."
        )

    # Train and val index sets are disjoint
    overlap = set(plan.split.train_indices) & set(plan.split.validation_indices)
    if overlap:
        errors.append(f"Train/val index overlap: {len(overlap)} indices.")
    else:
        verified.append("split")

    # Size warnings
    if plan.split.train_size < plan.config.batch_size:
        warnings.append(f"Train size ({plan.split.train_size}) < batch_size ({plan.config.batch_size})")
    if plan.split.validation_size < plan.config.batch_size:
        warnings.append(f"Val size ({plan.split.validation_size}) < batch_size ({plan.config.batch_size})")

    # Loaders exist
    if plan.train_loader is None:
        errors.append("Train loader is None.")
    else:
        verified.append("train_loader")
    if plan.validation_loader is None:
        warnings.append("Validation loader is None (may be intentional).")
    else:
        verified.append("validation_loader")

    # Model dimension summary
    dim_summary = plan.model_dim_summary
    if isinstance(dim_summary, dict) and not dim_summary.get("matched", True):
        errors.append(f"Model dimension mismatch: {dim_summary}")
    else:
        verified.append("model_bundle")

    # Optimizer, scheduler, evaluator, trainer exist
    if plan.optimizer is None:
        errors.append("Optimizer is None.")
    else:
        verified.append("optimizer")
    if plan.scheduler is None:
        errors.append("Scheduler is None.")
    else:
        verified.append("scheduler")
    if plan.evaluator is None:
        errors.append("Evaluator is None.")
    else:
        verified.append("evaluator")
    if plan.trainer is None:
        errors.append("Trainer is None.")
    else:
        verified.append("trainer")

    # Checkpoint directory parent exists
    ckpt_dir = plan.run_context.checkpoint_dir
    ckpt_parent = Path(ckpt_dir).parent
    if not ckpt_parent.exists():
        warnings.append(f"Checkpoint parent dir does not exist: {ckpt_parent}")
    verified.append("paths")

    # Tabular input dim vs encoder input dim
    tab_enc = plan.model_bundle.get("tabular_encoder", None) if hasattr(plan.model_bundle, 'get') else None
    if tab_enc is not None and hasattr(tab_enc, 'config') and hasattr(plan, 'base_dataset'):
        ds_cols = len(plan.base_dataset.config.tabular_columns)
        enc_dim = tab_enc.config.input_dim
        if ds_cols != enc_dim:
            errors.append(
                f"Tabular dim mismatch: dataset has {ds_cols} columns "
                f"but TabularEncoder expects input_dim={enc_dim}."
            )
        else:
            verified.append(f"tabular_dim({ds_cols})")

    elapsed = (time.monotonic() - t0) * 1000

    if errors:
        status = "failed"
    elif warnings:
        status = "warnings"
    else:
        status = "passed"

    return _PreflightResult(
        status=status, warnings=tuple(warnings), errors=tuple(errors),
        verified_subsystems=tuple(verified), elapsed_ms=elapsed,
    )


def _print_preflight(plan: _ExecutionPlan, pf: _PreflightResult) -> None:
    """Print a cockpit-style preflight report."""
    sel = plan.selection
    sp = plan.split
    meta = plan.metadata

    lines = [
        "",
        "=" * 64,
        "  PREFLIGHT DIAGNOSTICS",
        "=" * 64,
        "",
        "  Environment",
        f"    Python          : {meta.python_version}",
        f"    PyTorch         : {meta.torch_version}",
        f"    CUDA            : {meta.cuda_available}",
        f"    Device          : {meta.device}",
        f"    AMP             : {meta.mixed_precision}",
        "",
        "  Dataset",
        f"    Mode            : {sel.mode}",
        f"    Selected        : {', '.join(sel.selected_names)}",
        f"    Total Rows      : {sel.total_rows:,}",
    ]
    if sel.ignored_train_datasets:
        lines.append(f"    Ignored         : {', '.join(sel.ignored_train_datasets)}")

    lines.extend([
        "",
        "  Split",
        f"    Validation %    : {sp.validation_split:.1%}",
        f"    Train Samples   : {sp.train_size:,}",
        f"    Val Samples     : {sp.validation_size:,}",
        f"    Seed            : {sp.seed}",
        "",
        "  Training",
        f"    Epochs          : {plan.config.epochs}",
        f"    Batch Size      : {plan.config.batch_size}",
        f"    Workers         : {plan.config.num_workers}",
        f"    Optimizer       : {plan.config.optimizer} (lr={plan.config.learning_rate})",
        f"    Scheduler       : {plan.config.scheduler}",
        f"    Loss            : {plan.config.loss_name}",
        f"    Gradient Clip   : {plan.config.gradient_clip}",
        "",
        "  Model",
    ])
    if plan.model_dim_summary:
        ds = plan.model_dim_summary
        lines.append(f"    Total Params    : {ds.get('total_params', '?'):,}")
        lines.append(f"    Trainable       : {ds.get('trainable_params', '?'):,}")
        lines.append(f"    Dims Matched    : {ds.get('matched', '?')}")

    lines.extend([
        "",
        "  Checkpoints",
        f"    Save Best       : {plan.config.save_best}",
        f"    Save Latest     : {plan.config.save_latest}",
        f"    Directory       : {plan.run_context.checkpoint_dir}",
        "",
        "  Preflight",
        f"    Status          : {pf.status.upper()}",
        f"    Verified        : {len(pf.verified_subsystems)} subsystems",
        f"    Elapsed         : {pf.elapsed_ms:.0f} ms",
    ])

    if pf.warnings:
        lines.append("")
        lines.append("  Warnings:")
        for w in pf.warnings:
            lines.append(f"    [!] {w}")

    lines.extend(["", "=" * 64, ""])
    print("\n".join(lines))


# =============================================================================
# One-Batch Dry Run
# =============================================================================

def perform_dry_run(plan: _ExecutionPlan) -> _DryRunResult:
    """Execute a one-batch dry run by delegating to Trainer.dry_run_batch.

    train.py owns: getting one batch, building shape summary, returning result.
    Trainer owns: device transfer, forward, loss, prediction validation.

    Does NOT step optimizer, scheduler, or write checkpoints.
    Does NOT mutate training history.
    """
    t0 = time.monotonic()

    try:
        # Build one batch directly from dataset + collate to avoid
        # starting DataLoader worker processes (Windows pickle issue).
        loader = plan.train_loader
        subset = loader.dataset  # the Subset backing the loader
        batch_size = min(loader.batch_size or 1, len(subset))
        if batch_size == 0:
            return _DryRunResult(
                status="failed", elapsed_ms=(time.monotonic() - t0) * 1000,
                error="Train subset is empty.",
            )
        samples = [subset[i] for i in range(batch_size)]
        collate_fn = loader.collate_fn
        batch = collate_fn(samples)

        # Summarize batch shapes (train.py orchestration responsibility)
        shape_pairs: List[Tuple[str, str]] = []
        if isinstance(batch, dict):
            for k, v in sorted(batch.items()):
                if hasattr(v, "shape"):
                    shape_pairs.append((k, str(list(v.shape))))
                elif isinstance(v, (list, tuple)):
                    shape_pairs.append((k, f"len={len(v)}"))

        # Delegate to Trainer-owned diagnostic
        diag = plan.trainer.dry_run_batch(batch)

        forward_ok = diag.get("forward_ok", False)
        loss_ok = diag.get("loss_ok", False)
        error = diag.get("error", "")

        elapsed = (time.monotonic() - t0) * 1000
        return _DryRunResult(
            status="passed" if (forward_ok and loss_ok) else "failed",
            batch_shape_summary=tuple(shape_pairs),
            forward_ok=forward_ok, loss_ok=loss_ok,
            elapsed_ms=elapsed,
            error=error,
        )

    except Exception as exc:
        elapsed = (time.monotonic() - t0) * 1000
        return _DryRunResult(
            status="failed", elapsed_ms=elapsed,
            error=str(exc)[:300],
        )


def _print_dry_run(dr: _DryRunResult) -> None:
    """Print dry run results."""
    lines = [
        "",
        "-" * 64,
        "  DRY RUN RESULT",
        "-" * 64,
        f"  Status          : {dr.status.upper()}",
        f"  Forward OK      : {dr.forward_ok}",
        f"  Loss OK         : {dr.loss_ok}",
        f"  Elapsed         : {dr.elapsed_ms:.0f} ms",
    ]
    if dr.batch_shape_summary:
        lines.append("  Batch Shapes:")
        for k, v in dr.batch_shape_summary:
            lines.append(f"    {k:20s}: {v}")
    if dr.error:
        lines.append(f"  Error           : {dr.error}")
    lines.extend(["-" * 64, ""])
    print("\n".join(lines))


# =============================================================================
# Run Manifest
# =============================================================================

def write_run_manifest(plan: _ExecutionPlan, manifest_dir: Optional[Path] = None) -> Path:
    """Write run_manifest.json before training starts.

    Manifest ownership belongs to train.py, not Trainer.
    """
    if manifest_dir is None:
        manifest_dir = Path(plan.run_context.checkpoint_dir)

    manifest_path = manifest_dir / "run_manifest.json"

    manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "environment": {
            "python_version": plan.metadata.python_version,
            "platform": plan.metadata.platform,
            "torch_version": plan.metadata.torch_version,
            "cuda_available": plan.metadata.cuda_available,
            "device": plan.metadata.device,
            "mixed_precision": plan.metadata.mixed_precision,
        },
        "config": plan.config.as_dict(),
        "run_context": plan.run_context.as_dict(),
        "registry": {
            "discovered_at": plan.registry.discovered_at,
            "dataset_count": plan.registry.dataset_count,
            "dataset_names": list(plan.registry.dataset_names),
            "total_rows": plan.registry.total_rows,
        },
        "selection": {
            "mode": plan.selection.mode,
            "selected_names": list(plan.selection.selected_names),
            "ignored": list(plan.selection.ignored_train_datasets),
        },
        "split": {
            "validation_split": plan.split.validation_split,
            "seed": plan.split.seed,
            "dataset_size": plan.split.dataset_size,
            "train_size": plan.split.train_size,
            "validation_size": plan.split.validation_size,
        },
        "dataloader_reports": {
            "train": plan.train_loader_report,
            "validation": plan.validation_loader_report,
        },
        "model_dimensions": plan.model_dim_summary,
        "paths": {
            "checkpoint_dir": str(plan.run_context.checkpoint_dir),
            "experiment_dir": str(EXPERIMENT_DIR),
            "log_dir": str(LOG_DIR),
        },
        "preflight": {
            "status": plan.preflight.status if plan.preflight else "not_run",
        },
        "dry_run": {
            "status": plan.dry_run.status if plan.dry_run else "not_run",
        },
    }

    tmp_path = manifest_path.with_suffix(".json.tmp")
    try:
        manifest_dir.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp_path), str(manifest_path))
    except Exception as exc:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        raise TrainAppError(
            "manifest", "write",
            received=str(exc)[:200],
            expected=f"writable manifest at {manifest_path}",
            resolution="Check disk space and permissions.",
        ) from exc

    logger.info("Manifest written: %s", manifest_path)
    return manifest_path


# =============================================================================
# Execution Plan Display
# =============================================================================

def print_execution_plan(plan: _ExecutionPlan) -> None:
    """Print a compact, deterministic execution plan summary."""
    sel = plan.selection
    sp = plan.split
    meta = plan.metadata

    lines = [
        "",
        "=" * 64,
        "  EXECUTION PLAN",
        "=" * 64,
        f"  Mode              : {sel.mode}",
        f"  Datasets          : {', '.join(sel.selected_names)}",
    ]
    if sel.ignored_train_datasets:
        lines.append(f"  Ignored Datasets  : {', '.join(sel.ignored_train_datasets)}")
    lines.extend([
        f"  Validation Split  : {sp.validation_split:.1%}",
        f"  Train Samples     : {sp.train_size:,}",
        f"  Validation Samples: {sp.validation_size:,}",
        f"  Batch Size        : {plan.config.batch_size}",
        f"  Workers           : {plan.config.num_workers}",
        f"  Device            : {meta.device}",
        f"  AMP               : {meta.mixed_precision}",
        f"  Optimizer         : {plan.config.optimizer} (lr={plan.config.learning_rate})",
        f"  Scheduler         : {plan.config.scheduler}",
        f"  Epochs            : {plan.config.epochs}",
        f"  Save Best         : {plan.config.save_best}",
        f"  Save Latest       : {plan.config.save_latest}",
        f"  Checkpoint Dir    : {plan.run_context.checkpoint_dir}",
        f"  Seed              : {meta.seed}",
        f"  Status            : {plan.status}",
        "=" * 64,
        "",
    ])
    print("\n".join(lines))


# =============================================================================
# Final Summary
# =============================================================================

def _print_final_summary(
    result: Dict[str, Any], plan: _ExecutionPlan,
    manifest_path: Optional[Path], total_seconds: float,
) -> None:
    """Print compact final summary after training."""
    rt = result.get("runtime_state", {})
    hist = result.get("history", [])
    status = rt.get("status", "unknown")

    lines = [
        "",
        "=" * 64,
        "  TRAINING COMPLETE",
        "=" * 64,
        f"  Status            : {status.upper()}",
        f"  Total Runtime     : {total_seconds:.1f}s",
        f"  Epochs Completed  : {len(hist)}",
    ]

    # Best validation metric
    if hist:
        last = hist[-1]
        if "val_loss" in last:
            lines.append(f"  Final Val Loss    : {last['val_loss']:.6f}")
        if "val_rmse" in last:
            lines.append(f"  Final Val RMSE    : {last['val_rmse']:.6f}")
        if "val_r2" in last:
            lines.append(f"  Final Val R2      : {last['val_r2']:.6f}")
        if "lr" in last:
            lines.append(f"  Final LR          : {last['lr']:.2e}")

    lines.append(f"  Checkpoint Dir    : {plan.run_context.checkpoint_dir}")
    if manifest_path:
        lines.append(f"  Manifest          : {manifest_path}")

    # Next action
    if status == "completed":
        lines.append("  Next Action       : Evaluate best checkpoint on test set")
    elif status == "interrupted":
        lines.append("  Next Action       : Resume from latest checkpoint")
    else:
        lines.append("  Next Action       : Review error logs and retry")

    lines.extend(["=" * 64, ""])
    print("\n".join(lines))


# =============================================================================
# Public API
# =============================================================================

def build_execution_plan(
    *, config: Optional[Any] = None, **kwargs: Any,
) -> _ExecutionPlan:
    """Build and freeze an immutable execution plan.

    Assembles in dependency order. Does NOT call trainer.train().
    """
    from training.train_config import TrainConfig, build_train_config
    from training.run_context import build_run_context

    # -- 1. Config
    if config is None:
        try:
            config = build_train_config(**kwargs)
        except Exception as exc:
            raise TrainAppError(
                "config", "build", received=str(exc)[:200],
                expected="valid TrainConfig", resolution="Fix config parameters.",
            ) from exc

    if not isinstance(config, TrainConfig):
        raise TrainAppError(
            "config", "type_check", received=type(config).__name__,
            expected="TrainConfig", resolution="Pass a TrainConfig instance.",
        )

    if not config.is_frozen:
        try:
            config.validate().freeze()
        except Exception as exc:
            raise TrainAppError(
                "config", "validate_freeze", received=str(exc)[:200],
                expected="valid frozen config", resolution="Fix config parameters.",
            ) from exc

    # -- 1b. Global seeding (before any construction)
    _enforce_determinism(config.seed, getattr(config, 'deterministic', True))

    # -- 2. RunContext
    try:
        run_context = build_run_context(config)
    except Exception as exc:
        raise TrainAppError(
            "run_context", "build", received=str(exc)[:200],
            expected="valid RunContext", resolution="Check config and environment.",
        ) from exc

    # -- 3. Metadata
    metadata = _build_metadata(config, run_context)

    # -- 4. Registry
    registry = _discover_registry()

    # -- 5. Selection
    selection = _select_datasets(config, registry)
    logger.info("Selection: mode=%s, %d datasets", selection.mode, len(selection.selected_names))

    # -- 6. Dataset
    base_dataset = _build_dataset(selection)
    logger.info("Base dataset: %d samples", len(base_dataset))

    # -- 7. Split
    split = _build_split(config, base_dataset)
    logger.info("Split: train=%d, val=%d", split.train_size, split.validation_size)

    train_subset = Subset(base_dataset, list(split.train_indices))
    val_subset = Subset(base_dataset, list(split.validation_indices))

    # -- 8. Loaders
    train_loader, train_report, val_loader, val_report = _build_loaders(
        config, train_subset, val_subset,
    )

    # -- 9. Models
    tabular_columns = tuple(base_dataset.config.tabular_columns)
    tabular_input_dim = len(tabular_columns)
    model_bundle, dim_summary = _build_model_bundle(
        tabular_input_dim=tabular_input_dim,
        tabular_columns=tabular_columns,
    )

    # -- 10. Training Stack
    optimizer, scheduler, evaluator, trainer = _build_training_stack(
        config, run_context, model_bundle, train_loader, val_loader,
    )

    # -- 11. Freeze Plan
    plan = _ExecutionPlan(
        config=config, run_context=run_context, metadata=metadata,
        registry=registry, selection=selection, split=split,
        base_dataset=base_dataset, train_subset=train_subset,
        validation_subset=val_subset, train_loader=train_loader,
        validation_loader=val_loader,
        train_loader_report=train_report,
        validation_loader_report=val_report,
        model_bundle=model_bundle, model_dim_summary=dim_summary,
        optimizer=optimizer, scheduler=scheduler, evaluator=evaluator,
        trainer=trainer, status="READY",
        preflight=None, dry_run=None,
    )

    return plan


def run_training(
    *, config: Optional[Any] = None, **kwargs: Any,
) -> Dict[str, Any]:
    """Build execution plan, preflight, dry run, write manifest, and train."""
    t_start = time.monotonic()

    plan = build_execution_plan(config=config, **kwargs)
    print_execution_plan(plan)

    # Preflight
    pf = perform_preflight(plan)
    plan.preflight = pf
    _print_preflight(plan, pf)

    if pf.status == "failed":
        raise TrainAppError(
            "preflight", "validation",
            received="; ".join(pf.errors[:5]),
            expected="preflight status passed or warnings",
            resolution="Fix preflight errors before dry run/training.",
        )

    # Dry run
    dr = perform_dry_run(plan)
    plan.dry_run = dr
    _print_dry_run(dr)

    if dr.status == "failed":
        raise TrainAppError(
            "dry_run", "one_batch",
            received=dr.error or "forward/loss check failed",
            expected="successful one-batch dry run",
            resolution="Check model architecture, batch format, and loss function.",
        )

    # Manifest
    manifest_path = write_run_manifest(plan)

    # Training
    logger.info("Starting training...")
    plan.status = "TRAINING"
    result = plan.trainer.train()
    plan.status = "COMPLETED"

    total_seconds = time.monotonic() - t_start
    _print_final_summary(result, plan, manifest_path, total_seconds)

    return result


def main() -> int:
    """Application entry point with CLI mode separation."""
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="Multimodal AI Training Bootloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python training/train.py --smoke       # Run smoke tests\n"
            "  python training/train.py --train       # Full training\n"
            "  python training/train.py --plan-only   # Print execution plan\n"
            "  python training/train.py --dry-run     # Plan + one-batch dry run\n"
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true", help="Run smoke tests only")
    group.add_argument("--train", action="store_true", help="Full training pipeline")
    group.add_argument("--dry-run", action="store_true", help="Build plan + one-batch dry run")
    group.add_argument("--plan-only", action="store_true", help="Print execution plan and exit")

    args = parser.parse_args()

    if args.smoke:
        return run_smoke_tests()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s -- %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        if args.plan_only:
            plan = build_execution_plan()
            print_execution_plan(plan)
            pf = perform_preflight(plan)
            _print_preflight(plan, pf)
            return EXIT_SUCCESS

        if args.dry_run:
            plan = build_execution_plan()
            print_execution_plan(plan)
            pf = perform_preflight(plan)
            plan.preflight = pf
            _print_preflight(plan, pf)
            if pf.status == "failed":
                print(f"\n[FAILED] Preflight failed. Aborting dry run.", file=sys.stderr)
                for err in pf.errors:
                    print(f"  - {err}", file=sys.stderr)
                return EXIT_TRAINING_FAILURE
            dr = perform_dry_run(plan)
            plan.dry_run = dr
            _print_dry_run(dr)
            if dr.status == "failed":
                print(f"\n[FAILED] Dry run failed: {dr.error}")
                return EXIT_TRAINING_FAILURE
            print("\n[OK] Dry run passed. Ready for --train.")
            return EXIT_SUCCESS

        if args.train:
            result = run_training()
            status = result.get("runtime_state", {}).get("status", "unknown")
            if status == "completed":
                return EXIT_SUCCESS
            elif status == "interrupted":
                return EXIT_INTERRUPT
            return EXIT_TRAINING_FAILURE

    except TrainAppError as exc:
        print(str(exc), file=sys.stderr)
        stage = exc.stage
        if stage in ("config", "run_context"):
            return EXIT_CONFIG_FAILURE
        if stage in ("registry", "selection", "dataset", "split", "loaders"):
            return EXIT_DATASET_FAILURE
        if stage in ("models",):
            return EXIT_MODEL_FAILURE
        return EXIT_TRAINING_FAILURE
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] User cancelled.")
        return EXIT_INTERRUPT
    except Exception as exc:
        print(
            f"\n[FATAL TRAIN BOOTLOADER ERROR]\n"
            f"  Stage         : unhandled\n"
            f"  Likely area   : training/train.py main execution\n"
            f"  Error type    : {type(exc).__name__}\n"
            f"  Message       : {str(exc)[:300]}\n"
            f"  Resolution    : rerun with --smoke, then --plan-only; inspect error above",
            file=sys.stderr,
        )
        return EXIT_TRAINING_FAILURE

    return EXIT_SUCCESS


# =============================================================================
# Smoke Test
# =============================================================================

def run_smoke_tests() -> int:
    """Run local smoke tests without training."""
    logging.basicConfig(
        level=logging.WARNING,
        format="[%(asctime)s] [%(levelname)s] %(name)s -- %(message)s",
        datefmt="%H:%M:%S",
    )

    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            passed += 1
            print(f"    [PASS]     {name}")
        else:
            failed += 1
            msg = f"    [FAIL]     {name}"
            if detail:
                msg += f"  -- {detail}"
            print(msg)

    print("=" * 64)
    print("  training/train.py -- smoke test")
    print("=" * 64)

    # -- 1. Imports -----------------------------------------------------------
    print("\n  1. Imports...")
    check("TrainAppError importable", callable(TrainAppError))
    check("build_execution_plan importable", callable(build_execution_plan))
    check("run_training importable", callable(run_training))
    check("print_execution_plan importable", callable(print_execution_plan))
    check("perform_preflight importable", callable(perform_preflight))
    check("perform_dry_run importable", callable(perform_dry_run))
    check("write_run_manifest importable", callable(write_run_manifest))
    check("run_smoke_tests importable", callable(run_smoke_tests))
    check("main importable", callable(main))
    check("MANIFEST_SCHEMA_VERSION >= 1", MANIFEST_SCHEMA_VERSION >= 1)

    # -- 2. TrainAppError structured ------------------------------------------
    print("\n  2. TrainAppError structured...")
    err = TrainAppError("config", "test", received="bad", expected="good", resolution="fix it")
    check("error is RuntimeError", isinstance(err, RuntimeError))
    check("error has stage", err.stage == "config")
    check("error has subsystem", err.subsystem == "test")
    check("error message contains stage", "config" in str(err))

    # -- 3. Frozen dataclasses ------------------------------------------------
    print("\n  3. Frozen dataclasses...")
    snap = _RegistrySnapshot(
        discovered_at="now", dataset_count=2,
        datasets=(1, 2), dataset_names=("a", "b"),
        dataset_files=("a.csv", "b.csv"), total_rows=100,
    )
    try:
        snap.dataset_count = 5
        check("snapshot rejects mutation", False, "no error")
    except (AttributeError, TypeError):
        check("snapshot rejects mutation", True)

    sel = _DatasetSelection(
        mode="single", requested=("a",), selected_names=("a",),
        selected_files=("a.csv",), ignored_train_datasets=(),
        total_rows=50, duplicate_identity_count=0,
    )
    try:
        sel.mode = "full"
        check("selection rejects mutation", False, "no error")
    except (AttributeError, TypeError):
        check("selection rejects mutation", True)

    # -- 4. SplitPlan determinism ---------------------------------------------
    print("\n  4. SplitPlan determinism...")
    g1 = torch.Generator().manual_seed(42)
    p1 = torch.randperm(100, generator=g1).tolist()
    g2 = torch.Generator().manual_seed(42)
    p2 = torch.randperm(100, generator=g2).tolist()
    check("same seed -> same perm", p1 == p2)
    g3 = torch.Generator().manual_seed(99)
    p3 = torch.randperm(100, generator=g3).tolist()
    check("diff seed -> diff perm", p1 != p3)

    sp = _SplitPlan(
        validation_split=0.2, seed=42, dataset_size=100,
        train_size=80, validation_size=20,
        train_indices=tuple(p1[:80]), validation_indices=tuple(p1[80:]),
    )
    check("train+val = total", sp.train_size + sp.validation_size == sp.dataset_size)
    check("indices disjoint", len(set(sp.train_indices) & set(sp.validation_indices)) == 0)
    check("all indices in range", all(0 <= i < 100 for i in sp.train_indices + sp.validation_indices))

    # -- 5. ExecutionPlan immutability ----------------------------------------
    print("\n  5. ExecutionPlan immutability...")
    plan = _ExecutionPlan(
        status="READY", config=None, run_context=None,
        metadata=None, registry=None, selection=None, split=None,
        base_dataset=None, train_subset=None, validation_subset=None,
        train_loader=None, validation_loader=None,
        train_loader_report={}, validation_loader_report={},
        model_bundle=None, model_dim_summary={},
        optimizer=None, scheduler=None, evaluator=None, trainer=None,
        preflight=None, dry_run=None,
    )
    check("plan status", plan.status == "READY")
    # Topology mutation rejected
    try:
        plan.config = "bad"
        check("plan topology rejects mutation", False, "no error")
    except AttributeError:
        check("plan topology rejects mutation", True)
    try:
        plan.registry = "bad"
        check("plan registry rejects mutation", False, "no error")
    except AttributeError:
        check("plan registry rejects mutation", True)
    # Runtime handle mutation allowed
    plan.status = "RUNNING"
    check("plan runtime handle mutable", plan.status == "RUNNING")

    # -- 6. ExecutionMetadata -------------------------------------------------
    print("\n  6. ExecutionMetadata...")
    meta = _ExecutionMetadata(
        created_at="now", python_version="3.11", platform="test",
        torch_version="2.0", cuda_available=False, device="cpu",
        mixed_precision=False, seed=42,
    )
    try:
        meta.seed = 99
        check("metadata rejects mutation", False, "no error")
    except (AttributeError, TypeError):
        check("metadata rejects mutation", True)

    # -- 7. PreflightResult + DryRunResult ------------------------------------
    print("\n  7. Preflight + DryRun placeholders...")
    pf = _PreflightResult()
    check("preflight default status", pf.status == "not_implemented")
    dr = _DryRunResult(status="skipped")
    check("dryrun status", dr.status == "skipped")
    check("dryrun shape is tuple", isinstance(dr.batch_shape_summary, tuple))
    dr2 = _DryRunResult(status="passed", batch_shape_summary=(("a", "[4,3]"),))
    try:
        dr2.batch_shape_summary = ()
        check("dryrun rejects mutation", False)
    except (AttributeError, TypeError):
        check("dryrun rejects mutation", True)

    # -- 8. _build_metadata ---------------------------------------------------
    print("\n  8. _build_metadata...")
    from training.train_config import TrainConfig
    from training.run_context import build_run_context
    _tc = TrainConfig()
    _tc.validate().freeze()
    _rc = build_run_context(_tc)
    m = _build_metadata(_tc, _rc)
    check("metadata has created_at", bool(m.created_at))
    check("metadata has python_version", bool(m.python_version))
    check("metadata has torch_version", bool(m.torch_version))
    check("metadata device is string", isinstance(m.device, str))

    # -- 9. _discover_registry ------------------------------------------------
    print("\n  9. _discover_registry...")
    try:
        reg = _discover_registry()
        check("registry discovered", reg.dataset_count > 0)
        check("registry names tuple", isinstance(reg.dataset_names, tuple))
        check("registry files tuple", isinstance(reg.dataset_files, tuple))
        check("registry total_rows >= 0", reg.total_rows >= 0)
        check("registry count matches", reg.dataset_count == len(reg.dataset_names))
        try:
            reg.dataset_count = 0
            check("registry rejects mutation", False, "no error")
        except (AttributeError, TypeError):
            check("registry rejects mutation", True)
    except TrainAppError as e:
        check("registry discovered", False, str(e)[:100])

    # -- 10. _select_datasets -------------------------------------------------
    print("\n  10. _select_datasets...")
    if reg.dataset_count > 0:
        _tc10 = TrainConfig(dataset_name=reg.dataset_names[0])
        _tc10.validate().freeze()
        try:
            sel10 = _select_datasets(_tc10, reg)
            check("single mode", sel10.mode == "single")
            check("selected count = 1", len(sel10.selected_names) == 1)
        except TrainAppError as e:
            check("single mode", False, str(e)[:100])

        _tc10b = TrainConfig(dataset_name="nonexistent_zzzz")
        _tc10b.validate().freeze()
        try:
            _select_datasets(_tc10b, reg)
            check("unknown dataset rejected", False, "no error")
        except TrainAppError:
            check("unknown dataset rejected", True)

        # validation_dataset_name guard
        _tc10c = TrainConfig(validation_dataset_name="some_val_ds")
        _tc10c.validate().freeze()
        try:
            _select_datasets(_tc10c, reg)
            check("validation_dataset_name rejected", False, "no error")
        except TrainAppError:
            check("validation_dataset_name rejected", True)

    # -- 11. _build_split -----------------------------------------------------
    print("\n  11. _build_split...")
    class _MockDS:
        def __len__(self):
            return 50
    _mc = TrainConfig(validation_split=0.2)
    _mc.validate().freeze()
    sp11 = _build_split(_mc, _MockDS())
    check("split train_size", sp11.train_size == 40)
    check("split val_size", sp11.validation_size == 10)
    check("split disjoint", len(set(sp11.train_indices) & set(sp11.validation_indices)) == 0)
    check("split covers all", sp11.train_size + sp11.validation_size == 50)
    sp11b = _build_split(_mc, _MockDS())
    check("split deterministic", sp11.train_indices == sp11b.train_indices)
    _mc2 = TrainConfig(validation_split=0.2, seed=99)
    _mc2.validate().freeze()
    sp11c = _build_split(_mc2, _MockDS())
    check("diff seed -> diff split", sp11.train_indices != sp11c.train_indices)

    # -- 12. Split integrity (no raw assert) ----------------------------------
    print("\n  12. Split integrity checks...")
    import inspect
    src = inspect.getsource(_build_split)
    check("no raw assert in _build_split", "assert " not in src)
    check("uses TrainAppError in split", "TrainAppError" in src)

    # -- 13. No stale paths ---------------------------------------------------
    print("\n  13. No stale path wording...")
    _train_py_path = Path(__file__).resolve()
    _non_test_lines = []
    _in_smoke = False
    for _line in _train_py_path.read_text(encoding="utf-8").splitlines():
        if "def run_smoke_tests" in _line:
            _in_smoke = True
        if not _in_smoke:
            _non_test_lines.append(_line)
    _non_test_src = "\n".join(_non_test_lines)
    _stale_path = "data" + "/" + "preprocessed" + "/"
    check("no stale path wording", _stale_path not in _non_test_src)

    # -- 14. Model dimension validation exists --------------------------------
    print("\n  14. Model dimension validation...")
    check("_build_model_bundle callable", callable(_build_model_bundle))
    mb_src = inspect.getsource(_build_model_bundle)
    check("dim validation in model builder", "dimension_contract" in mb_src)
    check("get_embedding_dim check", "get_embedding_dim" in mb_src)
    check("tabular_input_dim param", "tabular_input_dim" in mb_src)
    check("tabular_columns param", "tabular_columns" in mb_src)
    check("TabularEncoderConfig(input_dim=" in mb_src, True)

    # -- 15. Required model keys complete -------------------------------------
    print("\n  15. Required model keys...")
    check("_REQUIRED_MODEL_KEYS complete", _REQUIRED_MODEL_KEYS == {
        "image_encoder", "text_encoder", "tabular_encoder", "fusion_model"
    })

    # -- 16. Exit codes defined -----------------------------------------------
    print("\n  16. Exit codes...")
    check("EXIT_SUCCESS = 0", EXIT_SUCCESS == 0)
    check("EXIT_INTERRUPT = 130", EXIT_INTERRUPT == 130)

    # -- 17. perform_dry_run has no manual .to(device) -------------------------
    print("\n  17. Dry run ownership...")
    _dr_src = inspect.getsource(perform_dry_run)
    check("no .to(device) in dry_run", ".to(device)" not in _dr_src)
    check("no image_encoder in dry_run", "image_encoder" not in _dr_src)
    check("no fusion_model in dry_run", "fusion_model" not in _dr_src)
    check("no MSELoss in dry_run", "MSELoss" not in _dr_src)
    check("delegates to trainer", "dry_run_batch" in _dr_src)

    # -- 18. Manifest uses atomic write ----------------------------------------
    print("\n  18. Manifest atomic write...")
    _mw_src = inspect.getsource(write_run_manifest)
    check("manifest uses os.replace", "os.replace" in _mw_src)
    check("manifest uses fsync", "os.fsync" in _mw_src)
    check("manifest uses tmp file", ".json.tmp" in _mw_src)

    # -- 19. No encoding artifacts ---------------------------------------------
    print("\n  19. Encoding artifacts...")
    _train_py_path = Path(__file__).resolve()
    _full_text = _train_py_path.read_text(encoding="utf-8")
    _bad_chars = ["\u2014", "\u26a0", "\u00b2"]  # em dash, warning sign, superscript 2
    _found_bad = [c for c in _bad_chars if c in _full_text]
    check("no encoding artifacts", len(_found_bad) == 0, f"found: {_found_bad}")

    # -- 20. Preflight validates topology --------------------------------------
    print("\n  20. Preflight strength...")
    _pf_src = inspect.getsource(perform_preflight)
    check("preflight checks split overlap", "overlap" in _pf_src.lower() or "&" in _pf_src)
    check("preflight checks dim matched", "matched" in _pf_src)
    check("preflight can return failed", '"failed"' in _pf_src)

    # -- 21. Global seeding exists ---------------------------------------------
    print("\n  21. Global seeding...")
    check("_enforce_determinism callable", callable(_enforce_determinism))
    _ed_src = inspect.getsource(_enforce_determinism)
    check("seeds random", "random.seed" in _ed_src)
    check("seeds torch", "torch.manual_seed" in _ed_src)
    check("seeds cuda", "cuda.manual_seed_all" in _ed_src)
    check("seeds numpy", "np.random.seed" in _ed_src)
    check("cudnn deterministic", "cudnn.deterministic" in _ed_src)
    check("cudnn benchmark", "cudnn.benchmark" in _ed_src)

    # -- 22. worker_init_seed wiring -------------------------------------------
    print("\n  22. Worker seed wiring...")
    _bl_src = inspect.getsource(_build_loaders)
    check("worker_init_seed in train loader", "worker_init_seed=config.seed" in _bl_src
          or "worker_init_seed" in _bl_src)

    # -- 23. Seeding called before construction ---------------------------------
    print("\n  23. Seeding order...")
    _bep_src = inspect.getsource(build_execution_plan)
    _seed_pos = _bep_src.find("_enforce_determinism")
    _reg_pos = _bep_src.find("_discover_registry")
    check("seeding before registry", 0 < _seed_pos < _reg_pos)

    # -- 24. Dry-run uses subset not iter(loader) -------------------------------
    print("\n  24. Dry-run strategy...")
    _dr_src = inspect.getsource(perform_dry_run)
    check("no iter(train_loader) in dry_run", "iter(" not in _dr_src)
    check("uses subset[i]", "subset[i]" in _dr_src or "loader.dataset" in _dr_src)
    check("delegates to trainer", "dry_run_batch" in _dr_src)

    # -- Final ----------------------------------------------------------------
    total = passed + failed
    print(f"\n{'=' * 64}")
    if failed == 0:
        print(f"  [PASS]  {passed}/{total} checks passed")
    else:
        print(f"  [FAIL]  {passed}/{total} passed, {failed} failed")
    print("=" * 64)

    return 1 if failed > 0 else 0


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    sys.exit(main())
    #the complete training comes to life