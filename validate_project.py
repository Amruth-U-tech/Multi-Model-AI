#!/usr/bin/env python3
# =============================================================================
# validate_project.py
# Global Validation Orchestrator -- Multimodal AI Pipeline
# =============================================================================
#
# Purpose:
#   Single-command end-to-end readiness validation for the entire project.
#   Coordinates staged contract checks across all layers without replacing
#   local smoke tests. Produces a clear pass/fail readiness score.
#
# Usage:
#   python validate_project.py              # default (safe, read-only)
#   python validate_project.py --quick      # skip heavy model/dataset checks
#   python validate_project.py --full       # include all checks
#   python validate_project.py --run-smoke  # also run local smoke tests
#   python validate_project.py --json       # print JSON summary to stdout
#   python validate_project.py --json-out report.json  # write JSON to file
#
# Safety:
#   - Read-only: no files modified, no datasets created, no training
#   - No checkpoint writing, no cache deletion
#   - No forced model downloads beyond normal online-first behavior
#   - ASCII-only output for Windows console safety
# =============================================================================

import sys
import os
import time
import platform
import importlib
import argparse
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

# ── Project root bootstrap ────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# =============================================================================
# Validation Result Tracking
# =============================================================================

class ValidationResult:
    """Single validation check result."""
    __slots__ = ("section", "name", "status", "detail", "duration_ms")

    def __init__(self, section: str, name: str, status: str, detail: str = "", duration_ms: float = 0.0):
        self.section = section
        self.name = name
        self.status = status       # PASS, FAIL, EXPECTED, WARN, SKIP
        self.detail = detail
        self.duration_ms = duration_ms

    def to_dict(self) -> dict:
        return {
            "section": self.section,
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "duration_ms": round(self.duration_ms, 2),
        }


class ValidationTracker:
    """Accumulates results and produces summary with section timing."""

    def __init__(self):
        self.results: List[ValidationResult] = []
        self._current_section = ""
        self._section_start = 0.0
        self._total_start = time.perf_counter()
        self.section_durations_ms: Dict[str, float] = {}
        self._section_order: List[str] = []

    def _finalize_section(self):
        """Record duration for the current section before starting a new one."""
        if self._current_section:
            elapsed = (time.perf_counter() - self._section_start) * 1000.0
            self.section_durations_ms[self._current_section] = elapsed

    def section(self, name: str):
        self._finalize_section()
        self._current_section = name
        self._section_start = time.perf_counter()
        self._section_order.append(name)
        print(f"\n  {name}...")

    def finalize(self):
        """Finalize the last active section. Call before summary."""
        self._finalize_section()

    def check(self, name: str, condition: bool, detail: str = ""):
        status = "PASS" if condition else "FAIL"
        r = ValidationResult(self._current_section, name, status, detail)
        self.results.append(r)
        tag = f"[{status}]"
        msg = f"    {tag:10s} {name}"
        if detail and status == "FAIL":
            msg += f"  -- {detail[:120]}"
        print(msg)

    def expected(self, name: str, detail: str = ""):
        r = ValidationResult(self._current_section, name, "EXPECTED", detail)
        self.results.append(r)
        print(f"    {'[EXPECTED]':10s} {name}")

    def warn(self, name: str, detail: str = ""):
        r = ValidationResult(self._current_section, name, "WARN", detail)
        self.results.append(r)
        print(f"    {'[WARN]':10s} {name}  -- {detail[:120]}")

    def skip(self, name: str, detail: str = ""):
        r = ValidationResult(self._current_section, name, "SKIP", detail)
        self.results.append(r)
        print(f"    {'[SKIP]':10s} {name}")

    def fail(self, name: str, detail: str = ""):
        r = ValidationResult(self._current_section, name, "FAIL", detail)
        self.results.append(r)
        msg = f"    {'[FAIL]':10s} {name}"
        if detail:
            msg += f"  -- {detail[:120]}"
        print(msg)

    @property
    def counts(self) -> Dict[str, int]:
        c: Dict[str, int] = {"PASS": 0, "FAIL": 0, "EXPECTED": 0, "WARN": 0, "SKIP": 0}
        for r in self.results:
            c[r.status] = c.get(r.status, 0) + 1
        return c

    @property
    def total_time_ms(self) -> float:
        return (time.perf_counter() - self._total_start) * 1000.0

    def slowest_section(self) -> str:
        if not self.section_durations_ms:
            return "N/A"
        return max(self.section_durations_ms, key=self.section_durations_ms.get)

    def readiness_score(self) -> int:
        c = self.counts
        total = c["PASS"] + c["FAIL"] + c["EXPECTED"] + c["WARN"]
        if total == 0:
            return 0
        good = c["PASS"] + c["EXPECTED"]
        score = int(round(100.0 * good / total))
        # Warnings reduce by 1 point each (max 10)
        score -= min(c["WARN"], 10)
        # Each FAIL is -5 points
        score -= c["FAIL"] * 5
        return max(0, min(100, score))

    def to_json(self) -> str:
        return json.dumps({
            "results": [r.to_dict() for r in self.results],
            "summary": self.counts,
            "readiness_score": self.readiness_score(),
            "total_time_ms": round(self.total_time_ms, 2),
            "section_timings_ms": {k: round(v, 1) for k, v in self.section_durations_ms.items()},
            "slowest_section": self.slowest_section(),
        }, indent=2)


# =============================================================================
# Stage 1: Environment
# =============================================================================

def validate_environment(t: ValidationTracker):
    t.section("1. Environment")

    t.check("Python >= 3.8", sys.version_info >= (3, 8), f"got {sys.version_info}")
    t.check("Platform detected", bool(platform.system()), platform.system())

    # ── Machine Summary ──────────────────────────────────────────────────
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    print(f"           CPU cores       : {cpu_count}")
    print(f"           Platform        : {platform.platform()}")
    print(f"           Python          : {sys.version.split()[0]}")

    # RAM estimate (psutil optional)
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
        print(f"           RAM             : {ram_gb} GB")
    except ImportError:
        print(f"           RAM             : unknown (psutil not installed)")

    # ── CUDA / torch ─────────────────────────────────────────────────────
    try:
        import torch
        t.check("torch imported", True)
        print(f"           torch version   : {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        print(f"           CUDA available  : {cuda_avail}")
        if cuda_avail:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"           GPU             : {gpu_name}")
            print(f"           CUDA devices    : {gpu_count}")
            t.check("CUDA available", True)
        else:
            print(f"           GPU             : unavailable")
            t.warn("CUDA not available", "GPU training will not work")
    except ImportError:
        t.fail("torch imported", "torch not installed")


# =============================================================================
# Stage 2: Dependency Imports
# =============================================================================

def validate_imports(t: ValidationTracker):
    t.section("2. Dependency Imports")

    required = [
        "torch", "torchvision", "transformers", "timm",
        "PIL", "pandas", "numpy", "tqdm",
    ]
    for mod_name in required:
        try:
            importlib.import_module(mod_name)
            t.check(f"import {mod_name}", True)
        except ImportError:
            t.fail(f"import {mod_name}", f"{mod_name} not installed")


# =============================================================================
# Stage 3: Path Infrastructure
# =============================================================================

def validate_paths(t: ValidationTracker):
    t.section("3. Path Infrastructure")

    try:
        from configs.paths import (
            PROJECT_ROOT, PREPROCESSED_DATASET_DIR, IMAGE_DATASET_DIR,
            CACHE_DIR, resolve_image_file, resolve_preprocessed_csv,
        )
        t.check("configs.paths imported", True)
        t.check("PROJECT_ROOT exists", PROJECT_ROOT.exists(), str(PROJECT_ROOT))
        t.check("PREPROCESSED_DATASET_DIR", PREPROCESSED_DATASET_DIR.exists())
        t.check("IMAGE_DATASET_DIR", IMAGE_DATASET_DIR.exists())
        t.check("CACHE_DIR", CACHE_DIR.exists())

        # Traversal guards
        try:
            resolve_image_file("../escape.jpg")
            t.fail("image traversal guard", "should have raised ValueError")
        except ValueError:
            t.expected("image traversal guard blocked ../escape.jpg")

        try:
            resolve_preprocessed_csv("../evil.csv")
            t.fail("csv traversal guard", "should have raised ValueError")
        except (ValueError, FileNotFoundError):
            t.expected("csv traversal guard blocked ../evil.csv")

    except ImportError as e:
        t.fail("configs.paths imported", str(e))


# =============================================================================
# Stage 4: Dataset Registry
# =============================================================================

def validate_registry(t: ValidationTracker):
    t.section("4. Dataset Registry")

    try:
        from data_pipeline.dataset_registry import (
            discover_datasets, resolve_dataset, resolve_dataset_group,
            REGISTERED_DATASETS,
        )
        t.check("registry imported", True)

        descs = discover_datasets()
        t.check("discover_datasets() returns list", isinstance(descs, list))
        t.check("found CSVs > 0", len(descs) > 0, f"found {len(descs)}")

        # Find sample_100
        sample = None
        for d in descs:
            if d.filename == "sample_100.csv":
                sample = d
                break
        t.check("sample_100.csv found", sample is not None)

        if sample:
            t.check("sample schema valid", sample.schema_valid)
            t.check("sample has image_path", sample.has_image_path_column)
            t.check("sample coverage > 0", sample.image_coverage_count > 0)
            t.check("sample invalid_img_count is int", isinstance(sample.invalid_image_path_count, int))

        # Groups
        t.check("sample group registered", "sample" in REGISTERED_DATASETS)
        t.check("all_discovered registered", "all_discovered" in REGISTERED_DATASETS)

        sample_group = resolve_dataset_group("sample")
        t.check("sample group has 2 files", len(sample_group) == 2)

        # Normalized name
        norm = resolve_dataset_group(" sample ")
        t.check("group name normalized", norm == sample_group)

    except Exception as e:
        t.fail("registry validation", str(e)[:200])


# =============================================================================
# Stage 5: Dataset Sample Contract
# =============================================================================

def validate_dataset(t: ValidationTracker, quick: bool = False):
    t.section("5. Dataset Sample Contract")

    if quick:
        t.skip("dataset construction", "skipped in quick mode")
        return

    try:
        from data_pipeline.dataset import DatasetConfig, MultimodalProductDataset

        # sample config
        cfg = DatasetConfig(dataset_name="sample_100")
        t.check("DatasetConfig(sample_100) builds", True)

        ds = MultimodalProductDataset(cfg)
        t.check("dataset length > 0", len(ds) > 0, f"len={len(ds)}")

        # First sample
        import torch
        with torch.no_grad():
            sample = ds[0]

        required_keys = [
            "sample_id", "row_index", "asin", "image", "image_path",
            "raw_text", "sanitized_text", "input_ids", "attention_mask",
            "tabular", "rating", "metadata",
        ]
        for key in required_keys:
            t.check(f"sample has '{key}'", key in sample)

        # Tensor contracts
        if "image" in sample:
            img = sample["image"]
            t.check("image shape (3,224,224)", tuple(img.shape) == (3, 224, 224))

        if "input_ids" in sample:
            ids = sample["input_ids"]
            t.check("input_ids is 2D", ids.ndim == 2)
            t.check("input_ids seq_len=64", ids.shape[1] == 64)

        if "attention_mask" in sample:
            mask = sample["attention_mask"]
            t.check("attention_mask is 2D", mask.ndim == 2)

        if "tabular" in sample:
            tab = sample["tabular"]
            t.check("tabular is 1D", tab.ndim == 1)

        if "rating" in sample:
            rat = sample["rating"]
            t.check("rating is scalar-like", rat.numel() == 1 if hasattr(rat, 'numel') else True)

        # all_discovered should fail with dup ASINs
        try:
            cfg_all = DatasetConfig(dataset_name="all_discovered")
            ds_all = MultimodalProductDataset(cfg_all)
            t.warn("all_discovered built without error", "expected dup ASIN failure")
        except (ValueError, KeyError) as e:
            if "duplicate" in str(e).lower() or "ASIN" in str(e):
                t.expected("all_discovered dup ASIN guard triggered")
            else:
                t.expected(f"all_discovered guard: {str(e)[:80]}")

    except Exception as e:
        t.fail("dataset construction", str(e)[:200])


# =============================================================================
# Stage 6: Data Pipeline Contract
# =============================================================================

def validate_pipeline(t: ValidationTracker, quick: bool = False):
    t.section("6. Data Pipeline Contract")

    # Transforms
    try:
        from data_pipeline.transforms import (
            get_transforms, safe_load_image, validate_tensor_output, INPUT_SIZE,
        )
        t.check("transforms imported", True)

        tf_train = get_transforms(mode="train")
        tf_eval = get_transforms(mode="eval")
        t.check("train transforms built", tf_train is not None)
        t.check("eval transforms built", tf_eval is not None)
    except Exception as e:
        t.fail("transforms import", str(e)[:200])

    # Tokenization
    try:
        from data_pipeline.tokenization import (
            sanitize_text, load_tokenizer, tokenize_batch, FALLBACK_TEXT,
        )
        t.check("tokenization imported", True)
        t.check("sanitize_text(None) -> fallback", sanitize_text(None) == FALLBACK_TEXT)
    except Exception as e:
        t.fail("tokenization import", str(e)[:200])

    # Collate
    try:
        from data_pipeline.collate import build_collate_fn
        t.check("collate imported", True)

        collate_fn = build_collate_fn()
        t.check("build_collate_fn() ok", collate_fn is not None)
    except Exception as e:
        t.fail("collate import", str(e)[:200])

    # Dataloader factory
    try:
        from data_pipeline.dataloader_factory import DataLoaderConfig
        t.check("dataloader_factory imported", True)

        dl_cfg = DataLoaderConfig()
        t.check("DataLoaderConfig defaults ok", dl_cfg.batch_size > 0)
    except Exception as e:
        t.fail("dataloader_factory import", str(e)[:200])

    if quick:
        t.skip("dataloader probe", "skipped in quick mode")
        return

    # Full pipeline probe: build a debug-mode loader on the sample dataset
    try:
        from data_pipeline.dataset import DatasetConfig, MultimodalProductDataset
        from data_pipeline.dataloader_factory import build_dataloader, DataLoaderConfig

        ds_cfg = DatasetConfig(dataset_name="sample_100")
        ds = MultimodalProductDataset(ds_cfg)
        dl_cfg = DataLoaderConfig(batch_size=2, execution_mode="debug")
        loader, health = build_dataloader(dataset_config=ds_cfg, loader_config=dl_cfg, dataset=ds)
        t.check("debug dataloader built", loader is not None)

        import torch
        with torch.no_grad():
            batch = next(iter(loader))
        t.check("first batch retrieved", batch is not None)

        # Check batch keys (collated names may differ from sample names)
        for key in ["images", "input_ids", "attention_mask", "tabular", "ratings"]:
            t.check(f"batch has '{key}'", key in batch)

    except Exception as e:
        t.fail("dataloader probe", str(e)[:200])


# =============================================================================
# Stage 7: Model Contract
# =============================================================================

def validate_models(t: ValidationTracker, quick: bool = False):
    t.section("7. Model Contracts")

    import torch

    # Fusion
    try:
        from models.fusion import FusionModel, FusionConfig
        t.check("FusionModel imported", True)

        model = FusionModel()
        B = 2
        with torch.no_grad():
            out = model(torch.randn(B, 512), torch.randn(B, 512), torch.randn(B, 512))
        t.check("fusion forward (B=2, 512)", "fused_embedding" in out)
        t.check("fusion output shape", out["fused_embedding"].shape == (B, 512))

        # Dimension contract guard
        try:
            FusionModel(FusionConfig(image_dim=256, text_dim=512))
            t.fail("fusion dim guard", "should have raised ValueError")
        except ValueError:
            t.expected("fusion dim guard blocked mismatched dims")

    except ImportError as e:
        t.fail("FusionModel import", str(e))

    # Tabular
    try:
        from models.tabular_encoder import TabularEncoder, TabularEncoderConfig, build_tabular_encoder
        t.check("TabularEncoder imported", True)

        enc = build_tabular_encoder(TabularEncoderConfig(input_dim=8))
        enc.eval()
        with torch.no_grad():
            out = enc(torch.randn(2, 8))
        t.check("tabular forward (B=2,F=8)", out.shape == (2, 512))

    except ImportError as e:
        t.fail("TabularEncoder import", str(e))

    # Image encoder (pretrained=False for speed)
    try:
        from models.image_encoder import ImageEncoder, ImageEncoderConfig, build_encoder
        t.check("ImageEncoder imported", True)

        if quick:
            t.skip("image encoder forward", "skipped in quick mode")
        else:
            cfg = ImageEncoderConfig(pretrained=False)
            enc = build_encoder(cfg)
            enc.eval()
            with torch.no_grad():
                out = enc(torch.randn(2, 3, 224, 224))
            t.check("image forward (B=2,3,224,224)", out.shape == (2, 512))

    except ImportError as e:
        t.fail("ImageEncoder import", str(e))

    # Text encoder
    try:
        from models.text_encoder import TextEncoder, TextEncoderConfig, build_text_encoder
        t.check("TextEncoder imported", True)

        if quick:
            t.skip("text encoder forward", "skipped in quick mode")
        else:
            cfg = TextEncoderConfig(freeze_backbone=True)
            enc = build_text_encoder(cfg)
            enc.eval()
            with torch.no_grad():
                dummy_ids = torch.randint(0, 100, (2, 64))
                dummy_mask = torch.ones(2, 64, dtype=torch.long)
                out = enc(dummy_ids, dummy_mask)
            t.check("text forward (B=2, seq=64)", out.shape == (2, 512))

    except ImportError as e:
        t.fail("TextEncoder import", str(e))


# =============================================================================
# Stage 8: Training Contracts
# =============================================================================

def validate_training_contracts(t: ValidationTracker):
    t.section("8. Training Contracts")

    # -- TrainConfig checks (existing) -----------------------------------------
    try:
        from training.train_config import (
            TrainConfig, TrainConfigError, ConfigFrozenError,
            ConfigState, build_train_config,
        )
        t.check("TrainConfig imported", True)

        # Default construct + validate
        cfg = TrainConfig()
        cfg.validate()
        t.check("default config validates", cfg.state == ConfigState.VALIDATED)

        # Freeze
        cfg.freeze()
        t.check("config freezes", cfg.state == ConfigState.FROZEN)

        # Frozen mutation rejected
        try:
            cfg.learning_rate = 0.1
            t.fail("frozen mutation guard", "should have raised ConfigFrozenError")
        except ConfigFrozenError:
            t.expected("frozen mutation guard blocks assignment")

        # Checkpoint traversal rejected
        try:
            TrainConfig(
                resume=True,
                resume_checkpoint="../checkpoints_evil/model.pt"
            ).validate()
            t.fail("checkpoint traversal guard", "should have raised")
        except TrainConfigError:
            t.expected("checkpoint traversal guard blocks escape")

        # as_dict
        cfg2 = build_train_config()
        d = cfg2.as_dict()
        t.check("as_dict returns dict", isinstance(d, dict) and "epochs" in d)

        # New fields: validation_split, train_all, train_datasets
        t.check("has validation_split", hasattr(cfg2, "validation_split"))
        t.check("validation_split default", 0.0 < cfg2.validation_split < 1.0)
        t.check("has train_all", hasattr(cfg2, "train_all"))
        t.check("has train_datasets", hasattr(cfg2, "train_datasets"))
        t.check("train_datasets is tuple", isinstance(cfg2.train_datasets, tuple))

        # Reject bad validation_split
        try:
            TrainConfig(validation_split=0.0).validate()
            t.fail("val_split=0 rejected", "should have raised")
        except TrainConfigError:
            t.expected("val_split=0 rejected")

    except ImportError as e:
        t.fail("TrainConfig import", str(e)[:200])
    except Exception as e:
        t.fail("training contracts", str(e)[:200])

    # -- train.py bootloader checks --------------------------------------------
    try:
        from training.train import (
            TrainAppError, build_execution_plan, run_training,
            print_execution_plan, main, MANIFEST_SCHEMA_VERSION,
            perform_preflight, perform_dry_run, write_run_manifest,
            run_smoke_tests, _build_split,
            EXIT_SUCCESS, EXIT_INTERRUPT,
        )
        t.check("training.train imports", True)
        t.check("TrainAppError is RuntimeError", issubclass(TrainAppError, RuntimeError))
        t.check("build_execution_plan callable", callable(build_execution_plan))
        t.check("run_training callable", callable(run_training))
        t.check("print_execution_plan callable", callable(print_execution_plan))
        t.check("main callable", callable(main))
        t.check("MANIFEST_SCHEMA_VERSION >= 1", MANIFEST_SCHEMA_VERSION >= 1)
        # Part 3 APIs
        t.check("perform_preflight callable", callable(perform_preflight))
        t.check("perform_dry_run callable", callable(perform_dry_run))
        t.check("write_run_manifest callable", callable(write_run_manifest))
        t.check("run_smoke_tests callable", callable(run_smoke_tests))
        t.check("EXIT_SUCCESS = 0", EXIT_SUCCESS == 0)
        t.check("EXIT_INTERRUPT = 130", EXIT_INTERRUPT == 130)
        # Split integrity uses TrainAppError, not assert
        import inspect as _vi
        _split_src = _vi.getsource(_build_split)
        t.check("split no raw assert", "assert " not in _split_src)
        t.check("split uses TrainAppError", "TrainAppError" in _split_src)
        # Dry run delegates to Trainer
        _dr_src = _vi.getsource(perform_dry_run)
        t.check("dry_run no .to(device)", ".to(device)" not in _dr_src)
        t.check("dry_run delegates to trainer", "dry_run_batch" in _dr_src)
        # Manifest atomic
        _mw_src = _vi.getsource(write_run_manifest)
        t.check("manifest atomic (os.replace)", "os.replace" in _mw_src)
        t.check("manifest atomic (fsync)", "os.fsync" in _mw_src)
        # Preflight validates topology
        _pf_src = _vi.getsource(perform_preflight)
        t.check("preflight checks overlap", "overlap" in _pf_src.lower() or "&" in _pf_src)
        t.check("preflight can fail", '"failed"' in _pf_src)
        # Trainer.dry_run_batch exists
        from training.trainer import Trainer
        t.check("Trainer.dry_run_batch", hasattr(Trainer, "dry_run_batch"))
        # Dataset has no _torch (pickle safety)
        from data_pipeline.dataset import MultimodalProductDataset
        t.check("dataset no _torch attr", "_torch" not in MultimodalProductDataset.__init__.__code__.co_varnames)
        # run_training checks preflight failed
        _rt_src = _vi.getsource(run_training)
        t.check("run_training aborts on preflight fail", 'pf.status == "failed"' in _rt_src)
        # manifest mkdir inside try
        t.check("manifest mkdir inside try", "manifest_dir.mkdir" not in _mw_src.split("try:")[0] if "try:" in _mw_src else False)
        # tabular input dim contract
        t.check("model_bundle accepts tabular_input_dim", "tabular_input_dim" in _vi.getsource(build_execution_plan))
        # validation_dataset_name guard
        _sel_src = _vi.getsource(run_training).replace(" ", "")  # rough check
        t.check("validation_dataset_name not ignored",
                "validation_dataset_name" in _vi.getsource(build_execution_plan) or
                "validation_dataset_name" in open(str(Path(__file__).parent / "training" / "train.py"), encoding="utf-8").read().split("def _select_datasets")[1].split("def ")[0])
        # worker init picklable
        import pickle as _pkl
        from data_pipeline.dataloader_factory import make_worker_init_fn as _mwif
        _wfn = _mwif(42)
        try:
            _pkl.dumps(_wfn)
            t.check("worker_init_fn picklable", True)
        except Exception:
            t.check("worker_init_fn picklable", False)
    except ImportError as e:
        t.fail("training.train import", str(e)[:200])
    except Exception as e:
        t.fail("training.train contracts", str(e)[:200])

    # -- RunContext checks -----------------------------------------------------
    try:
        from training.run_context import (
            RunContext, RunContextError, build_run_context,
        )
        t.check("RunContext imported", True)
        t.check("build_run_context imported", True)

        # Package-level export
        try:
            from training import RunContext as _RC, build_run_context as _brc
            t.check("package export: RunContext", True)
            t.check("package export: build_run_context", True)
        except ImportError:
            t.fail("package export: RunContext", "not in training/__init__.py")

        # Build RunContext from frozen config (CPU path -- deterministic)
        from training.train_config import build_train_config as _btc
        rc_cfg = _btc(device="cpu")
        rc_cfg.freeze()
        ctx = build_run_context(rc_cfg)
        t.check("RunContext builds from frozen config", ctx is not None)
        t.check("runtime device is cpu", ctx.device == "cpu")

        # Serialization surface
        rc_dict = ctx.as_dict()
        t.check("RunContext as_dict returns dict",
                isinstance(rc_dict, dict) and "device" in rc_dict)

        rc_summary = ctx.summary()
        t.check("RunContext summary returns string",
                isinstance(rc_summary, str) and len(rc_summary) > 50)

        # Immutability guard
        try:
            ctx.device = "tpu"
            t.fail("RunContext immutability guard", "should have raised AttributeError")
        except AttributeError:
            t.expected("RunContext immutability guard blocks mutation")

        # Unfrozen config rejected
        try:
            unfrozen_cfg = _btc(device="cpu")  # validated but NOT frozen
            build_run_context(unfrozen_cfg)
            t.fail("RunContext rejects unfrozen config", "should have raised")
        except RunContextError:
            t.expected("RunContext rejects unfrozen config")

        # Bad config type rejected
        try:
            build_run_context({"device": "cpu"})
            t.fail("RunContext rejects bad config type", "should have raised")
        except RunContextError:
            t.expected("RunContext rejects bad config type")

        # CUDA guard on CPU-only machine
        import torch
        if not torch.cuda.is_available():
            try:
                cuda_cfg = _btc(device="cuda")
                cuda_cfg.freeze()
                build_run_context(cuda_cfg)
                t.fail("RunContext CUDA guard", "should have raised on CPU-only")
            except RunContextError:
                t.expected("RunContext CUDA guard rejects cuda on CPU-only")

    except ImportError as e:
        t.fail("RunContext import", str(e)[:200])
    except Exception as e:
        t.fail("RunContext contracts", str(e)[:200])

    # -- Optimizer checks ------------------------------------------------------
    try:
        from training.optimizer import (
            OptimizerError, build_optimizer,
            validate_optimizer_inputs, summarize_optimizer, optimizer_to_dict,
        )
        t.check("optimizer imported", True)

        # Package-level export
        try:
            from training import OptimizerError as _OE, build_optimizer as _bo
            t.check("package export: OptimizerError", True)
            t.check("package export: build_optimizer", True)
        except ImportError:
            t.fail("package export: build_optimizer", "not in training/__init__.py")

        # Build optimizer on dummy model (CPU, deterministic)
        import torch.nn as _nn
        class _DummyModel(_nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = _nn.Linear(4, 1)
            def forward(self, x):
                return self.fc(x)

        from training.train_config import build_train_config as _btc2
        from training.run_context import build_run_context as _brc2
        opt_cfg = _btc2(optimizer="adamw", device="cpu")
        opt_cfg.freeze()
        opt_ctx = _brc2(opt_cfg)
        dummy = _DummyModel()
        opt = build_optimizer(opt_cfg, opt_ctx, dummy)
        t.check("optimizer builds on dummy model", opt is not None)

        import torch.optim as _optim
        t.check("optimizer is AdamW", isinstance(opt, _optim.AdamW))

        # Summary helper
        opt_summary = summarize_optimizer(opt, model=dummy, config=opt_cfg)
        t.check("optimizer summary returns string",
                isinstance(opt_summary, str) and len(opt_summary) > 50)

        # Serialization helper
        opt_dict = optimizer_to_dict(opt, model=dummy, config=opt_cfg)
        t.check("optimizer_to_dict returns dict",
                isinstance(opt_dict, dict) and "optimizer_type" in opt_dict)

        # Bad config type rejected
        try:
            build_optimizer({"optimizer": "adamw"}, opt_ctx, dummy)
            t.fail("optimizer rejects bad config", "should have raised")
        except OptimizerError:
            t.expected("optimizer rejects bad config type")

        # Bad model type rejected
        try:
            build_optimizer(opt_cfg, opt_ctx, "not_a_model")
            t.fail("optimizer rejects bad model", "should have raised")
        except OptimizerError:
            t.expected("optimizer rejects bad model type")

        # Duplicate parameter guard
        class _DupGroupModel(_nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = _nn.Linear(4, 1)
            def forward(self, x):
                return self.fc(x)
            def get_optimizer_parameter_groups(self):
                params = list(self.fc.parameters())
                return [
                    {"name": "A", "params": params},
                    {"name": "B", "params": params},
                ]

        try:
            build_optimizer(opt_cfg, opt_ctx, _DupGroupModel())
            t.fail("optimizer duplicate param guard", "should have raised")
        except OptimizerError:
            t.expected("optimizer duplicate param guard blocks overlap")

        # Zero trainable params guard
        class _FrozenModel(_nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = _nn.Linear(4, 1)
                for p in self.parameters():
                    p.requires_grad = False
            def forward(self, x):
                return self.fc(x)

        try:
            build_optimizer(opt_cfg, opt_ctx, _FrozenModel())
            t.fail("optimizer frozen model guard", "should have raised")
        except OptimizerError:
            t.expected("optimizer frozen model guard blocks zero params")

        # Config/context mismatch rejected
        opt_cfg_b = _btc2(optimizer="adamw", device="cpu")
        opt_cfg_b.freeze()
        opt_ctx_b = _brc2(opt_cfg_b)
        try:
            build_optimizer(opt_cfg, opt_ctx_b, dummy)  # cfg A + ctx B
            t.fail("optimizer config/context mismatch", "should have raised")
        except OptimizerError:
            t.expected("optimizer config/context mismatch rejected")

        # Grouped model preserves semantic metadata
        from training.optimizer import get_optimizer_metadata as _gom
        class _GroupedModel(_nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = _nn.Linear(4, 8)
                self.head = _nn.Linear(8, 1)
            def forward(self, x):
                return self.head(self.backbone(x))
            def get_optimizer_parameter_groups(self):
                return [
                    {"name": "Backbone", "params": self.backbone.parameters()},
                    {"name": "Head", "params": self.head.parameters()},
                ]

        gopt = build_optimizer(opt_cfg, opt_ctx, _GroupedModel())
        gdict = optimizer_to_dict(gopt)
        t.check("grouped dict has semantic names",
                gdict["groups"][0].get("name") == "Backbone"
                and gdict["groups"][1].get("name") == "Head")
        t.check("grouped dict has used_model_api",
                gdict.get("used_model_api") is True)

        gsummary = summarize_optimizer(gopt)
        t.check("grouped summary has semantic names",
                "Backbone" in gsummary and "Head" in gsummary)

        # Fallback metadata
        fdict = optimizer_to_dict(opt)
        t.check("fallback dict has used_model_api=False",
                fdict.get("used_model_api") is False)
        t.check("fallback dict group named all_trainable",
                fdict["groups"][0].get("name") == "all_trainable")

    except ImportError as e:
        t.fail("optimizer import", str(e)[:200])
    except Exception as e:
        t.fail("optimizer contracts", str(e)[:200])

    # -- Scheduler checks ------------------------------------------------------
    try:
        from training.scheduler import (
            SchedulerError, build_scheduler,
            validate_scheduler_inputs, summarize_scheduler, scheduler_to_dict,
            get_scheduler_metadata,
        )
        t.check("scheduler imported", True)

        # Package-level export
        try:
            from training import SchedulerError as _SE, build_scheduler as _bs
            t.check("package export: SchedulerError", True)
            t.check("package export: build_scheduler", True)
        except ImportError:
            t.fail("package export: build_scheduler", "not in training/__init__.py")

        # Build scheduler on dummy model (CPU, deterministic)
        from training.train_config import build_train_config as _btc3
        from training.run_context import build_run_context as _brc3
        from training.optimizer import build_optimizer as _bo3
        import torch.nn as _nn3
        class _SchedDummy(_nn3.Module):
            def __init__(self):
                super().__init__()
                self.fc = _nn3.Linear(4, 1)
            def forward(self, x):
                return self.fc(x)

        s_cfg = _btc3(scheduler="cosine", device="cpu", warmup_epochs=2, epochs=20)
        s_cfg.freeze()
        s_ctx = _brc3(s_cfg)
        s_opt = _bo3(s_cfg, s_ctx, _SchedDummy())
        sched = build_scheduler(s_cfg, s_ctx, s_opt)
        t.check("scheduler builds on dummy model", sched is not None)

        s_meta = get_scheduler_metadata(sched)
        t.check("scheduler metadata attached",
                s_meta is not None and s_meta.scheduler_type == "cosine")
        t.check("scheduler step_policy is epoch",
                s_meta.step_policy == "epoch")

        # Summary helper
        s_summ = summarize_scheduler(sched)
        t.check("scheduler summary returns string",
                isinstance(s_summ, str) and len(s_summ) > 50)

        # Serialization helper
        s_dict = scheduler_to_dict(sched)
        t.check("scheduler_to_dict returns dict",
                isinstance(s_dict, dict) and "scheduler_type" in s_dict)

        # Plateau step policy
        p_cfg = _btc3(scheduler="plateau", device="cpu", warmup_epochs=0)
        p_cfg.freeze()
        p_ctx = _brc3(p_cfg)
        p_opt = _bo3(p_cfg, p_ctx, _SchedDummy())
        p_sched = build_scheduler(p_cfg, p_ctx, p_opt)
        p_meta = get_scheduler_metadata(p_sched)
        t.check("plateau step_policy is validation_metric",
                p_meta.step_policy == "validation_metric")

        # None scheduler no-op
        n_cfg = _btc3(scheduler="none", device="cpu")
        n_cfg.freeze()
        n_ctx = _brc3(n_cfg)
        n_opt = _bo3(n_cfg, n_ctx, _SchedDummy())
        n_sched = build_scheduler(n_cfg, n_ctx, n_opt)
        n_meta = get_scheduler_metadata(n_sched)
        t.check("none scheduler type", n_meta.scheduler_type == "none")

        # Config/context mismatch rejected
        try:
            build_scheduler(s_cfg, p_ctx, s_opt)
            t.fail("scheduler config/context mismatch", "should have raised")
        except SchedulerError:
            t.expected("scheduler config/context mismatch rejected")

        # Plateau + warmup rejection
        try:
            pw_cfg = _btc3(scheduler="plateau", device="cpu", warmup_epochs=3)
            pw_cfg.freeze()
            pw_ctx = _brc3(pw_cfg)
            pw_opt = _bo3(pw_cfg, pw_ctx, _SchedDummy())
            build_scheduler(pw_cfg, pw_ctx, pw_opt)
            t.fail("plateau+warmup rejection", "should have raised")
        except SchedulerError:
            t.expected("plateau+warmup rejected")

        # step_gamma > 1 rejection
        try:
            sg_cfg = _btc3(scheduler="step", device="cpu", warmup_epochs=0)
            sg_cfg.step_gamma = 1.5
            sg_cfg.freeze()
            sg_ctx = _brc3(sg_cfg)
            sg_opt = _bo3(sg_cfg, sg_ctx, _SchedDummy())
            build_scheduler(sg_cfg, sg_ctx, sg_opt)
            t.fail("step_gamma>1 rejection", "should have raised")
        except SchedulerError:
            t.expected("step_gamma>1 rejected")

        # plateau_factor >= 1 rejection
        try:
            pf_cfg = _btc3(scheduler="plateau", device="cpu", warmup_epochs=0)
            pf_cfg.plateau_factor = 1.0
            pf_cfg.freeze()
            pf_ctx = _brc3(pf_cfg)
            pf_opt = _bo3(pf_cfg, pf_ctx, _SchedDummy())
            build_scheduler(pf_cfg, pf_ctx, pf_opt)
            t.fail("plateau_factor>=1 rejection", "should have raised")
        except SchedulerError:
            t.expected("plateau_factor>=1 rejected")

        # Plateau metadata: step_policy and metric
        t.check("plateau meta step_policy",
                p_meta.step_policy == "validation_metric")
        t.check("plateau meta metric_name",
                p_meta.metric_name == "validation_loss")
        t.check("plateau meta warmup_enabled false",
                p_meta.warmup_enabled is False)

    except ImportError as e:
        t.fail("scheduler import", str(e)[:200])
    except Exception as e:
        t.fail("scheduler contracts", str(e)[:200])

    # -- Evaluation checks -----------------------------------------------------
    try:
        from training.evaluation import (
            EvaluationError, EvaluationMetadata, EvaluationRuntimeState,
            EvaluationResult, Evaluator, build_evaluator,
            compute_loss, compute_metrics, extract_prediction,
        )
        t.check("evaluation imported", True)

        # Package-level export
        try:
            from training import EvaluationError as _EE, build_evaluator as _be
            t.check("package export: EvaluationError", True)
            t.check("package export: build_evaluator", True)
        except ImportError:
            t.fail("package export: build_evaluator", "not in training/__init__.py")

        # Build evaluator on dummy config
        from training.train_config import build_train_config as _btc4
        from training.run_context import build_run_context as _brc4
        e_cfg = _btc4(loss_name="mse", device="cpu")
        e_cfg.freeze()
        e_ctx = _brc4(e_cfg)
        ev = build_evaluator(e_cfg, e_ctx)
        t.check("evaluator builds", ev is not None)
        t.check("evaluator metadata", ev.metadata.problem_type == "regression")
        t.check("evaluator loss_name", ev.metadata.loss_name == "mse")

        import torch as _t4

        # Valid MSE loss
        _p = _t4.tensor([1.0, 2.0, 3.0])
        _tgt = _t4.tensor([1.5, 2.5, 3.5])
        _loss = compute_loss(_p, _tgt, e_cfg)
        t.check("MSE loss computes", isinstance(_loss, _t4.Tensor) and _loss.dim() == 0)

        # Valid MAE loss
        e_cfg_mae = _btc4(loss_name="mae", device="cpu")
        e_cfg_mae.freeze()
        _loss_mae = compute_loss(_p, _tgt, e_cfg_mae)
        t.check("MAE loss computes", _loss_mae.dim() == 0)

        # Valid Huber loss
        e_cfg_hub = _btc4(loss_name="huber", device="cpu")
        e_cfg_hub.freeze()
        _loss_hub = compute_loss(_p, _tgt, e_cfg_hub)
        t.check("Huber loss computes", _loss_hub.dim() == 0)

        # Metrics
        _m = compute_metrics(_p, _tgt)
        t.check("metrics has r2", "r2" in _m and isinstance(_m["r2"], float))
        t.check("metrics has rmse", "rmse" in _m and isinstance(_m["rmse"], float))

        # Dict prediction extraction
        _pred_dict = {"rating_prediction": _t4.tensor([1.0, 2.0])}
        _ext = extract_prediction(_pred_dict)
        t.check("dict extraction works", isinstance(_ext, _t4.Tensor))

        # Missing prediction key
        try:
            extract_prediction({"embedding": _t4.randn(4)})
            t.fail("missing prediction key", "should have raised")
        except EvaluationError:
            t.expected("missing prediction key rejected")

        # Shape mismatch
        try:
            compute_metrics(_t4.tensor([1.0, 2.0]), _t4.tensor([1.0, 2.0, 3.0]))
            t.fail("shape mismatch", "should have raised")
        except EvaluationError:
            t.expected("shape mismatch rejected")

        # NaN prediction
        try:
            compute_metrics(_t4.tensor([1.0, float('nan')]), _t4.tensor([1.0, 2.0]))
            t.fail("NaN prediction", "should have raised")
        except EvaluationError:
            t.expected("NaN prediction rejected")

        # Inf target
        try:
            compute_metrics(_t4.tensor([1.0, 2.0]), _t4.tensor([1.0, float('inf')]))
            t.fail("Inf target", "should have raised")
        except EvaluationError:
            t.expected("Inf target rejected")

        # R2 constant-target stability
        _const = _t4.tensor([3.0, 3.0, 3.0])
        _m_const = compute_metrics(_const.clone(), _const)
        t.check("R2 constant perfect=1.0", _m_const["r2"] == 1.0)

        # Best-validation tracking
        r_val = ev.evaluate(_p, _tgt, split="validation", epoch=1)
        is_best = ev.update_best(r_val)
        t.check("first epoch is best", is_best is True)
        t.check("best epoch set", ev.state.best_validation_epoch == 1)

        # Non-validation rejected
        r_train = ev.evaluate(_p, _tgt, split="train", epoch=2)
        try:
            ev.update_best(r_train)
            t.fail("train update_best rejected", "should have raised")
        except EvaluationError:
            t.expected("train update_best rejected")

        # Config/context mismatch
        e_cfg_b = _btc4(loss_name="mae", device="cpu")
        e_cfg_b.freeze()
        e_ctx_b = _brc4(e_cfg_b)
        try:
            build_evaluator(e_cfg, e_ctx_b)
            t.fail("evaluation config/context mismatch", "should have raised")
        except EvaluationError:
            t.expected("evaluation config/context mismatch rejected")

        # Complex tensor rejection
        try:
            compute_metrics(_t4.tensor([1.0+0j, 2.0+0j]), _t4.tensor([1.0, 2.0]))
            t.fail("complex tensor rejection", "should have raised")
        except EvaluationError:
            t.expected("complex tensor rejected")

        # Unfrozen compute_loss rejection
        try:
            _unfr = _btc4(loss_name="mse", device="cpu")
            compute_loss(_p, _tgt, _unfr)
            t.fail("unfrozen compute_loss rejection", "should have raised")
        except EvaluationError:
            t.expected("unfrozen compute_loss rejected")

        # Batch-level update_best rejection
        try:
            ev2 = build_evaluator(e_cfg, e_ctx)
            br = ev2.evaluate(_p, _tgt, split="validation", epoch=1, batch_index=3)
            ev2.update_best(br)
            t.fail("batch update_best rejection", "should have raised")
        except EvaluationError:
            t.expected("batch update_best rejected")

        # Duplicate epoch update_best rejection
        try:
            ev3 = build_evaluator(e_cfg, e_ctx)
            dr1 = ev3.evaluate(_p, _tgt, split="validation", epoch=1)
            ev3.update_best(dr1)
            dr1b = ev3.evaluate(_p, _tgt, split="validation", epoch=1)
            ev3.update_best(dr1b)
            t.fail("duplicate epoch rejection", "should have raised")
        except EvaluationError:
            t.expected("duplicate epoch update_best rejected")

    except ImportError as e:
        t.fail("evaluation import", str(e)[:200])
    except Exception as e:
        t.fail("evaluation contracts", str(e)[:200])


# =============================================================================
# Stage 8b: Trainer Contract Validation
# =============================================================================

def validate_trainer_contracts(t: ValidationTracker):
    t.section("8b. Trainer Contracts")
    try:
        from training.trainer import (
            Trainer, TrainerError, build_trainer,
            _TrainingEvent, _TrainerStatus, _VALID_TRANSITIONS,
            _TERMINAL_STATUSES, _REQUIRED_MODEL_KEYS,
            _REQUIRED_BATCH_KEYS, _CHECKPOINT_REQUIRED_KEYS,
            _CHECKPOINT_VERSION, _TRAINER_SCHEMA_VERSION,
            _assert_child_path, _gpu_memory_snapshot,
            _validate_loader_like,
        )
        t.check("trainer imports", True)

        # TrainerError is RuntimeError
        t.check("TrainerError is RuntimeError", issubclass(TrainerError, RuntimeError))

        # State machine: terminal states have no transitions
        for ts in _TERMINAL_STATUSES:
            t.check(f"terminal {ts.value} has no transitions",
                    len(_VALID_TRANSITIONS.get(ts, set())) == 0)

        # Event enum has required events
        required_events = [
            "INITIALIZED", "TRAINING_STARTED", "EPOCH_STARTED",
            "TRAINING_COMPLETED", "TRAINING_INTERRUPTED", "TRAINING_FAILED",
            "CHECKPOINT_SAVED", "VALIDATION_STARTED", "VALIDATION_COMPLETED",
        ]
        for ev in required_events:
            t.check(f"event {ev} exists", hasattr(_TrainingEvent, ev))

        # Required model keys
        for key in ["image_encoder", "text_encoder", "tabular_encoder", "fusion_model"]:
            t.check(f"model key '{key}' required", key in _REQUIRED_MODEL_KEYS)

        # Required batch keys
        for key in ["images", "input_ids", "attention_mask", "tabular", "ratings"]:
            t.check(f"batch key '{key}' required", key in _REQUIRED_BATCH_KEYS)

        # Checkpoint required keys -- original + hardened
        for key in ["checkpoint_version", "model_state_dict", "optimizer_state_dict",
                     "epoch", "global_step", "trainer_schema_version", "trainer_class"]:
            t.check(f"checkpoint key '{key}' required", key in _CHECKPOINT_REQUIRED_KEYS)

        # Schema version constants
        t.check("TRAINER_SCHEMA_VERSION >= 1", _TRAINER_SCHEMA_VERSION >= 1)
        t.check("CHECKPOINT_VERSION >= 1", _CHECKPOINT_VERSION >= 1)

        # Module helpers exist and are callable
        t.check("_assert_child_path callable", callable(_assert_child_path))
        t.check("_gpu_memory_snapshot callable", callable(_gpu_memory_snapshot))
        t.check("_validate_loader_like callable", callable(_validate_loader_like))

        # _gpu_memory_snapshot returns expected keys
        snap = _gpu_memory_snapshot()
        for k in ("cuda_available", "device_name", "allocated_mb",
                   "reserved_mb", "max_allocated_mb"):
            t.check(f"gpu_snapshot has '{k}'", k in snap)

        # Trainer class has hardened methods
        for method in ("_validate_prediction_contract", "_validate_model_device",
                       "_validate_batch_device", "_record_failure",
                       "_restore_evaluator_state", "_validate_checkpoint_restore_contract",
                       "_validate_optimizer_model_integrity"):
            t.check(f"Trainer.{method} exists", hasattr(Trainer, method))

        # build_trainer rejects non-config
        import torch.nn as nn
        try:
            build_trainer(
                config="bad", run_context=None, model_bundle=None,
                optimizer=None, scheduler=None, evaluator=None,
                train_loader=None,
            )
            t.fail("non-config rejection", "should have raised")
        except TrainerError:
            t.expected("non-config rejected")

        # --- Common infra (save_best=False to avoid val_loader requirement) ---
        from training.train_config import build_train_config
        from training.run_context import build_run_context
        from training.optimizer import build_optimizer
        from training.scheduler import build_scheduler
        from training.evaluation import build_evaluator

        _tc = build_train_config(device="cpu", warmup_epochs=0, save_best=False)
        _tc.freeze()
        _rc = build_run_context(_tc)

        # Missing model keys rejected
        _md = nn.ModuleDict({"image_encoder": nn.Linear(2, 2)})
        _op = build_optimizer(config=_tc, run_context=_rc, model=_md)
        _sc = build_scheduler(config=_tc, run_context=_rc, optimizer=_op)
        _ev = build_evaluator(_tc, _rc)
        try:
            build_trainer(
                config=_tc, run_context=_rc, model_bundle=_md,
                optimizer=_op, scheduler=_sc, evaluator=_ev,
                train_loader=[{"x": 1}],
            )
            t.fail("missing model key rejection", "should have raised")
        except TrainerError:
            t.expected("missing model keys rejected")

        # Non-iterable loader rejected
        for bad_loader in [123, "bad", {}]:
            try:
                _validate_loader_like(bad_loader, "test")
                t.fail(f"loader {type(bad_loader).__name__} rejected", "should have raised")
            except TrainerError:
                t.expected(f"loader {type(bad_loader).__name__} rejected")

        # Path traversal rejected
        import tempfile
        from pathlib import Path as _VP
        _vbase = _VP(tempfile.mkdtemp())
        try:
            _assert_child_path(_vbase, _vbase / ".." / "escape.pt", "test")
            t.fail("path traversal rejection", "should have raised")
        except TrainerError:
            t.expected("path traversal rejected")

        # save_best=True without val_loader rejected (expected guard)
        _tc_sb = build_train_config(device="cpu", warmup_epochs=0, save_best=True)
        _tc_sb.freeze()
        _rc_sb = build_run_context(_tc_sb)
        _full_md = nn.ModuleDict({
            "image_encoder": nn.Linear(2, 2),
            "text_encoder": nn.Linear(2, 2),
            "tabular_encoder": nn.Linear(2, 2),
            "fusion_model": nn.Linear(2, 2),
        })
        _op_sb = build_optimizer(config=_tc_sb, run_context=_rc_sb, model=_full_md)
        _sc_sb = build_scheduler(config=_tc_sb, run_context=_rc_sb, optimizer=_op_sb)
        _ev_sb = build_evaluator(_tc_sb, _rc_sb)
        try:
            build_trainer(
                config=_tc_sb, run_context=_rc_sb, model_bundle=_full_md,
                optimizer=_op_sb, scheduler=_sc_sb, evaluator=_ev_sb,
                train_loader=[{"x": 1}], val_loader=None,
                render_dashboard=False,
            )
            t.fail("save_best no val_loader rejected", "should have raised")
        except TrainerError:
            t.expected("save_best without val_loader rejected")

        # Runtime state has AMP and timing fields
        _full_md2 = nn.ModuleDict({
            "image_encoder": nn.Linear(2, 2),
            "text_encoder": nn.Linear(2, 2),
            "tabular_encoder": nn.Linear(2, 2),
            "fusion_model": nn.Linear(2, 2),
        })
        _op2 = build_optimizer(config=_tc, run_context=_rc, model=_full_md2)
        _sc2 = build_scheduler(config=_tc, run_context=_rc, optimizer=_op2)
        _ev2 = build_evaluator(_tc, _rc)
        _tr = build_trainer(
            config=_tc, run_context=_rc, model_bundle=_full_md2,
            optimizer=_op2, scheduler=_sc2, evaluator=_ev2,
            train_loader=[{"x": 1}], render_dashboard=False,
        )
        rt = _tr.runtime_state()
        for fld in ("amp_status", "amp_fallback_reason", "train_time_seconds",
                     "val_time_seconds", "checkpoint_time_seconds",
                     "warning_count", "peak_gpu_memory_mb"):
            t.check(f"runtime_state has '{fld}'", fld in rt)

        # Prediction contract rejects bad shape
        import torch
        bad_pred = torch.randn(4, 1, 1)
        good_tgt = torch.randn(4)
        try:
            _tr._validate_prediction_contract(bad_pred, good_tgt, 1, 1)
            t.fail("bad prediction shape rejected", "should have raised")
        except TrainerError:
            t.expected("bad prediction shape rejected")

        # Malformed evaluator restore rejected
        try:
            _tr._restore_evaluator_state({"evaluation": "not_a_dict"})
            t.fail("malformed eval restore rejected", "should have raised")
        except TrainerError:
            t.expected("malformed eval restore rejected")

        # Optimizer-model integrity passes normally
        try:
            _tr._validate_optimizer_model_integrity(1)
            t.check("optimizer-model integrity normal", True)
        except TrainerError:
            t.fail("optimizer-model integrity normal", "unexpected error")

        # Optimizer-model integrity detects replaced submodule
        old_fm = _full_md2["fusion_model"]
        _full_md2["fusion_model"] = nn.Linear(5, 1)
        try:
            _tr._validate_optimizer_model_integrity(1)
            t.fail("replaced submodule rejected", "should have raised")
        except TrainerError:
            t.expected("replaced submodule rejected")
        _full_md2["fusion_model"] = old_fm

        # Broken iterable-like loader rejected
        class _VBrokenIter:
            def __iter__(self):
                raise ValueError("broken")
        try:
            _validate_loader_like(_VBrokenIter(), "test")
            t.fail("broken iterable rejected", "should have raised")
        except TrainerError:
            t.expected("broken iterable rejected")

        # Scalar prediction rejected
        import torch
        scalar_p = torch.tensor(3.14)
        tgt_1d = torch.randn(4)
        try:
            _tr._validate_prediction_contract(scalar_p, tgt_1d, 1, 1)
            t.fail("scalar prediction rejected", "should have raised")
        except TrainerError:
            t.expected("scalar prediction rejected")

        # _validate_evaluator_payload exists and works
        t.check("Trainer._validate_evaluator_payload exists",
                hasattr(Trainer, "_validate_evaluator_payload"))
        try:
            _tr._validate_evaluator_payload({"no_eval": True})
            t.fail("Phase1 eval payload rejected", "should have raised")
        except TrainerError:
            t.expected("Phase1 eval payload rejected")
        try:
            _tr._validate_evaluator_payload({
                "evaluation": {"state": {"samples_evaluated": -1}}
            })
            t.fail("Phase1 eval counter rejected", "should have raised")
        except TrainerError:
            t.expected("Phase1 eval counter rejected")

        # Batch semantic guards exist: NaN image
        good_batch = {
            "images": torch.randn(2, 3, 2, 2),
            "input_ids": torch.randint(0, 10, (2, 4)),
            "attention_mask": torch.ones(2, 4, dtype=torch.long),
            "tabular": torch.randn(2, 3),
            "ratings": torch.randn(2),
        }
        nan_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v
                     for k, v in good_batch.items()}
        nan_batch["images"][0, 0, 0, 0] = float('nan')
        try:
            _tr._validate_batch(nan_batch, 1, 1)
            t.fail("NaN image rejected", "should have raised")
        except TrainerError:
            t.expected("NaN image rejected")

        # float input_ids rejected
        bad_ids_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v
                        for k, v in good_batch.items()}
        bad_ids_batch["input_ids"] = torch.randn(2, 4)
        try:
            _tr._validate_batch(bad_ids_batch, 1, 1)
            t.fail("float input_ids rejected", "should have raised")
        except TrainerError:
            t.expected("float input_ids rejected")

        # bool attention_mask accepted
        bool_mask_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v
                          for k, v in good_batch.items()}
        bool_mask_batch["attention_mask"] = torch.ones(2, 4, dtype=torch.bool)
        try:
            _tr._validate_batch(bool_mask_batch, 1, 1)
            t.check("bool attention_mask accepted", True)
        except TrainerError:
            t.fail("bool attention_mask accepted", "unexpectedly rejected")

        # Strict evaluator restore rejects bad latest_loss
        try:
            _tr._restore_evaluator_state({
                "evaluation": {"state": {"latest_loss": "bad"}}
            })
            t.fail("bad latest_loss rejected", "should have raised")
        except TrainerError:
            t.expected("bad latest_loss rejected")

        # Checkpoint atomicity: no unsafe unlink+rename pattern
        import inspect as _vinsp
        _save_src = _vinsp.getsource(Trainer._save_checkpoint)
        t.check("no .unlink() in _save_checkpoint",
                ".unlink()" not in _save_src)
        t.check("os.replace used in _save_checkpoint",
                "os.replace" in _save_src)

    except ImportError as e:
        t.fail("trainer import", str(e)[:200])
    except Exception as e:
        t.fail("trainer contracts", str(e)[:200])


# =============================================================================
# Stage 9: Local Smoke Tests (optional subprocess)
# =============================================================================

_SMOKE_FILES = [
    "configs/paths.py",
    "data_pipeline/dataset_registry.py",
    "data_pipeline/transforms.py",
    "data_pipeline/tokenization.py",
    "data_pipeline/collate.py",
    "data_pipeline/dataloader_factory.py",
    "models/fusion.py",
    "models/tabular_encoder.py",
    "models/image_encoder.py",
    "models/text_encoder.py",
    "training/train_config.py",
    "training/run_context.py",
    "training/optimizer.py",
    "training/scheduler.py",
    "training/evaluation.py",
    "training/trainer.py",
]


def validate_smoke_tests(t: ValidationTracker, run_smoke: bool = False):
    t.section("9. Local Smoke Tests")

    if not run_smoke:
        t.skip("local smoke tests", "use --run-smoke to execute")
        return

    python = sys.executable
    for smoke_file in _SMOKE_FILES:
        path = _PROJECT_ROOT / smoke_file
        if not path.exists():
            t.fail(f"smoke {smoke_file}", "file not found")
            continue

        t0 = time.perf_counter()
        try:
            result = subprocess.run(
                [python, "-B", str(path)],
                capture_output=True, text=True, timeout=120,
                cwd=str(_PROJECT_ROOT),
            )
            dur = (time.perf_counter() - t0) * 1000.0
            if result.returncode == 0:
                t.check(f"smoke {smoke_file}", True)
            else:
                # Get last meaningful line from stderr or stdout
                output = result.stderr or result.stdout
                last_lines = [l for l in output.strip().splitlines() if l.strip()]
                last = last_lines[-1][:120] if last_lines else "unknown error"
                t.fail(f"smoke {smoke_file}", f"exit={result.returncode} | {last}")
        except subprocess.TimeoutExpired:
            t.fail(f"smoke {smoke_file}", "timed out after 120s")
        except Exception as e:
            t.fail(f"smoke {smoke_file}", str(e)[:120])


# =============================================================================
# Stage 9: Timing + Summary
# =============================================================================

def _validate_json_out_path(raw_path: str) -> Path:
    """
    Validate --json-out target path for filesystem safety.
    Rejects directories, traversal attempts, and ensures parent exists.
    """
    p = Path(raw_path)

    # Reject directory targets
    if p.is_dir():
        raise ValueError(
            f"--json-out target is a directory, not a file: {p}"
        )

    # Reject traversal in relative paths
    if not p.is_absolute():
        resolved = (_PROJECT_ROOT / p).resolve()
        if not str(resolved).startswith(str(_PROJECT_ROOT.resolve())):
            raise ValueError(
                f"--json-out path escapes project root via traversal: {raw_path}"
            )
        p = resolved
    else:
        p = p.resolve()

    # Ensure parent directory exists (do NOT create arbitrary dirs)
    if not p.parent.exists():
        raise ValueError(
            f"--json-out parent directory does not exist: {p.parent}"
        )

    return p


def print_summary(
    t: ValidationTracker,
    print_json: bool = False,
    json_out_path: Optional[str] = None,
):
    t.finalize()  # lock final section timing

    c = t.counts
    score = t.readiness_score()
    total_ms = t.total_time_ms
    slowest = t.slowest_section()
    slowest_ms = t.section_durations_ms.get(slowest, 0.0)

    # ── Validation Timing Summary ─────────────────────────────────────
    print("\n" + "-" * 64)
    print("  VALIDATION TIMING SUMMARY")
    print("-" * 64)
    print(f"  Total validation time : {total_ms:.0f} ms")
    print(f"  Slowest stage         : {slowest} ({slowest_ms:.0f} ms)")
    print(f"  Stage timings:")
    for section_name in t._section_order:
        dur = t.section_durations_ms.get(section_name, 0.0)
        marker = " <<" if section_name == slowest else ""
        print(f"    {section_name:40s} {dur:8.0f} ms{marker}")
    print("-" * 64)

    # ── Results Summary ───────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  PROJECT VALIDATION SUMMARY")
    print("=" * 64)
    print(f"  Passed           : {c['PASS']}")
    print(f"  Expected Guards  : {c['EXPECTED']}")
    print(f"  Warnings         : {c['WARN']}")
    print(f"  Skipped          : {c['SKIP']}")
    print(f"  Failures         : {c['FAIL']}")
    print(f"  Total Time       : {total_ms:.0f} ms")
    print(f"  Readiness Score  : {score}/100")

    if score >= 95:
        rec = "PRODUCTION-INFRASTRUCTURE READY for next phase"
    elif score >= 90:
        rec = "Ready with minor warnings -- review before training"
    elif score >= 80:
        rec = "Usable but fix warnings before training"
    else:
        rec = "NOT READY -- fix failures before proceeding"

    print(f"  Recommendation   : {rec}")
    print("=" * 64)

    if c["FAIL"] > 0:
        print("\n  FAILURES:")
        for r in t.results:
            if r.status == "FAIL":
                print(f"    - [{r.section}] {r.name}: {r.detail}")

    # ── JSON output ───────────────────────────────────────────────────
    if print_json:
        print("\n" + t.to_json())

    if json_out_path:
        try:
            safe_path = _validate_json_out_path(json_out_path)
            with open(safe_path, "w") as f:
                f.write(t.to_json())
            print(f"\n  JSON report saved to: {safe_path}")
        except ValueError as e:
            print(f"\n  [ERROR] Cannot write JSON: {e}")

    return score


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multimodal AI Pipeline -- Global Validation Orchestrator"
    )
    parser.add_argument("--quick", action="store_true", help="Skip heavy model/dataset checks")
    parser.add_argument("--full", action="store_true", help="Run all checks including slow ones")
    parser.add_argument("--run-smoke", action="store_true", help="Also run local smoke tests as subprocesses")
    parser.add_argument("--json", action="store_true", help="Print JSON summary to stdout (read-only)")
    parser.add_argument("--json-out", type=str, default=None, metavar="PATH",
                        help="Write JSON report to a specific file (explicit write)")
    args = parser.parse_args()

    quick = args.quick and not args.full

    print("=" * 64)
    print("  Multi-Model AI -- Global Project Validation")
    print(f"  Root: {_PROJECT_ROOT}")
    print(f"  Mode: {'quick' if quick else 'full'}")
    print("=" * 64)

    t = ValidationTracker()

    validate_environment(t)
    validate_imports(t)
    validate_paths(t)
    validate_registry(t)
    validate_dataset(t, quick=quick)
    validate_pipeline(t, quick=quick)
    validate_models(t, quick=quick)
    validate_training_contracts(t)
    validate_trainer_contracts(t)
    validate_smoke_tests(t, run_smoke=args.run_smoke)

    score = print_summary(t, print_json=args.json, json_out_path=args.json_out)

    sys.exit(0 if score >= 80 else 1)


if __name__ == "__main__":
    main()
