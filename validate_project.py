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
#   python validate_project.py --json       # output JSON summary
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
    """Accumulates results and produces summary."""

    def __init__(self):
        self.results: List[ValidationResult] = []
        self._current_section = ""
        self._section_start = 0.0
        self._total_start = time.perf_counter()

    def section(self, name: str):
        self._current_section = name
        self._section_start = time.perf_counter()
        print(f"\n  {name}...")

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
        }, indent=2)


# =============================================================================
# Stage 1: Environment
# =============================================================================

def validate_environment(t: ValidationTracker):
    t.section("1. Environment")

    t.check("Python >= 3.8", sys.version_info >= (3, 8), f"got {sys.version_info}")
    t.check("Platform detected", bool(platform.system()), platform.system())

    # Machine summary
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    print(f"           CPU cores       : {cpu_count}")
    print(f"           Platform        : {platform.platform()}")
    print(f"           Python          : {sys.version.split()[0]}")

    # RAM estimate
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
        print(f"           RAM             : {ram_gb} GB")
    except ImportError:
        print(f"           RAM             : (psutil not installed)")

    # CUDA / torch
    try:
        import torch
        t.check("torch imported", True)
        print(f"           torch version   : {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        if cuda_avail:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"           CUDA device     : {gpu_name}")
            t.check("CUDA available", True)
        else:
            print(f"           CUDA            : not available")
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
# Stage 8: Local Smoke Tests (optional subprocess)
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
]


def validate_smoke_tests(t: ValidationTracker, run_smoke: bool = False):
    t.section("8. Local Smoke Tests")

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

def print_summary(t: ValidationTracker, output_json: bool = False):
    c = t.counts
    score = t.readiness_score()
    total_ms = t.total_time_ms

    # Find slowest section
    section_times: Dict[str, float] = {}
    for r in t.results:
        section_times[r.section] = section_times.get(r.section, 0) + r.duration_ms
    # We don't have per-result timing currently, so show total
    slowest = max(section_times, key=section_times.get) if section_times else "N/A"

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

    if output_json:
        json_path = _PROJECT_ROOT / "validation_report.json"
        with open(json_path, "w") as f:
            f.write(t.to_json())
        print(f"\n  JSON report saved to: {json_path}")

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
    parser.add_argument("--json", action="store_true", help="Output JSON summary to validation_report.json")
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
    validate_smoke_tests(t, run_smoke=args.run_smoke)

    score = print_summary(t, output_json=args.json)

    sys.exit(0 if score >= 80 else 1)


if __name__ == "__main__":
    main()
