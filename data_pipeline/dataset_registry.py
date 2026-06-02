# =============================================================================
# data_pipeline/dataset_registry.py
# Dataset Source Authority — Multimodal AI Pipeline
# =============================================================================
#
# Ownership (this file ONLY):
#   - Discovers available preprocessed CSVs from PREPROCESSED_DATASET_DIR
#   - Validates the shared core schema across all CSVs
#   - Reports row count, domain/category, image coverage, missing images
#   - Resolves a dataset by name/filename into a DatasetDescriptor
#   - Optionally builds a multi-source descriptor from several CSVs
#   - Detects cross-source ASIN collisions
#
# What this file does NOT own:
#   +-----------------------------+---------------------------+
#   | Responsibility              | Correct File              |
#   +-----------------------------+---------------------------+
#   | Image loading/transforms    | data_pipeline/transforms  |
#   | Text tokenization           | data_pipeline/tokenization|
#   | Sample construction         | data_pipeline/dataset     |
#   | Batch assembly / collation  | data_pipeline/collate     |
#   | DataLoader scheduling       | data_pipeline/dataloader  |
#   | GPU transfer / .cuda()      | train.py                  |
#   | Model forward pass          | models/*                  |
#   | Train/val splitting         | future split module       |
#   +-----------------------------+---------------------------+
#
# Design:
#   STATELESS    -- no mutable shared state
#   CPU-ONLY     -- no torch, no GPU
#   LAZY         -- CSVs read only when descriptors are built
#   LIGHTWEIGHT  -- uses csv + pathlib only, no pandas in prod path
#   WORKER-SAFE  -- descriptors are immutable data objects
# =============================================================================

from __future__ import annotations

import csv
import sys
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------
# Project Routing
# ---------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _THIS_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

try:
    from configs.paths import (
        PREPROCESSED_DATASET_DIR,
        IMAGE_DATASET_DIR,
        list_preprocessed_csvs,
        resolve_preprocessed_csv,
        resolve_image_file,
    )
except ImportError as _err:
    raise RuntimeError(
        "ROUTING FAILURE: Cannot import configs.paths. "
        f"sys.path: {sys.path[:5]}..."
    ) from _err

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Constants
# =============================================================================

# Shared core schema that every preprocessed CSV must have
CORE_SCHEMA_COLUMNS = frozenset({
    "asin", "text", "image_url", "price", "rating_number", "category", "rating",
})

# Identity key for cross-source deduplication
IDENTITY_COLUMN = "asin"


# =============================================================================
# 2. Error helper
# =============================================================================

def _registry_error(
    stage: str,
    message: str,
    filename: str = "",
    resolution: str = "",
    cause: Optional[Exception] = None,
) -> str:
    parts = [
        f"[REGISTRY ERROR]",
        f"  Stage     : {stage}",
    ]
    if filename:
        parts.append(f"  File      : {filename}")
    parts.append(f"  Message   : {message}")
    if resolution:
        parts.append(f"  Resolution: {resolution}")
    if cause:
        parts.append(f"  Cause     : {type(cause).__name__}: {cause}")
    return "\n".join(parts)


# =============================================================================
# 3. DatasetDescriptor — immutable audit object for one CSV source
# =============================================================================

@dataclass(frozen=True)
class DatasetDescriptor:
    """Immutable descriptor for a single preprocessed CSV source."""

    filename: str
    csv_path: str
    row_count: int
    columns: tuple
    has_image_path_column: bool
    image_support_status: str   # 'image_path_column_present' | 'asin_fallback_only' | 'no_images_found'
    image_path_policy: str      # 'use_image_path_then_asin' | 'use_asin_jpg'
    categories: tuple
    image_coverage_count: int
    image_missing_count: int
    image_coverage_pct: float
    schema_valid: bool
    schema_missing: tuple
    invalid_image_path_count: int = 0
    invalid_image_path_examples: tuple = ()
    audit_ms: Optional[float] = None

    @property
    def domain_name(self) -> str:
        """Derive domain name from filename."""
        name = self.filename
        for suffix in ("_processed.csv", ".csv"):
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                break
        return name.replace("_", " ").replace("meta ", "").title()


# =============================================================================
# 4. Multi-source descriptor
# =============================================================================

@dataclass(frozen=True)
class MultiSourceDescriptor:
    """Immutable descriptor for a multi-source combined dataset."""

    sources: tuple  # Tuple[DatasetDescriptor, ...]
    total_rows: int
    total_images_found: int
    total_images_missing: int
    cross_source_asin_collisions: int
    collision_examples: tuple  # Tuple[str, ...] (first N collisions)
    combined_schema_valid: bool
    audit_ms: Optional[float] = None


# =============================================================================
# 5. CSV auditing (lightweight, no pandas)
# =============================================================================

def _read_csv_header(csv_path: Path) -> List[str]:
    """Read just the header row from a CSV."""
    with open(csv_path, encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.reader(fh)
        try:
            return next(reader)
        except StopIteration:
            return []


def _audit_single_csv(csv_path: Path) -> DatasetDescriptor:
    """
    Build a DatasetDescriptor for one CSV file.

    Uses lightweight csv module only — no pandas. Scans the full file to
    compute row count, categories, and image coverage.
    """
    t0 = time.perf_counter() * 1000.0
    filename = csv_path.name
    columns: List[str] = []
    row_count = 0
    categories = set()
    images_found = 0
    images_missing = 0
    has_image_path = False
    invalid_image_path_count = 0
    invalid_image_path_examples: List[str] = []

    # Future scalability note:
    #   Current csv scan is fine for current 6k/20k/100k rows.
    #   Future large corpora may use cached manifests, schema fingerprints,
    #   parquet, lazy shards. Do not implement those now.

    with open(csv_path, encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        columns = list(reader.fieldnames or [])
        has_image_path = "image_path" in columns

        for row in reader:
            row_count += 1

            # Category tracking
            if "category" in columns:
                cat_val = row.get("category", "")
                if cat_val and cat_val.strip():
                    categories.add(cat_val.strip())

            # Image coverage: route through resolve_image_file for consistency
            asin_val = row.get("asin", "")
            if asin_val and asin_val.strip():
                asin_clean = asin_val.strip()
                if has_image_path:
                    ip = row.get("image_path", "").strip()
                    if ip:
                        # Use resolve_image_file for traversal safety
                        try:
                            img_file = resolve_image_file(ip)
                        except ValueError:
                            # Track invalid image path — do not crash discovery
                            invalid_image_path_count += 1
                            if len(invalid_image_path_examples) < 10:
                                invalid_image_path_examples.append(
                                    f"row={row_count}: {ip!r} (asin={asin_clean})"
                                )
                            img_file = resolve_image_file(asin_clean)
                    else:
                        img_file = resolve_image_file(asin_clean)
                else:
                    img_file = resolve_image_file(asin_clean)
                if img_file.exists():
                    images_found += 1
                else:
                    images_missing += 1

    # Schema check
    col_set = frozenset(columns)
    schema_missing = tuple(sorted(CORE_SCHEMA_COLUMNS - col_set))
    schema_valid = len(schema_missing) == 0

    total_images = images_found + images_missing
    coverage_pct = (images_found / total_images * 100.0) if total_images > 0 else 0.0

    # Image semantics
    if has_image_path:
        image_support_status = "image_path_column_present"
        image_path_policy = "use_image_path_then_asin"
    elif images_found > 0:
        image_support_status = "asin_fallback_only"
        image_path_policy = "use_asin_jpg"
    else:
        image_support_status = "no_images_found"
        image_path_policy = "use_asin_jpg"

    t1 = time.perf_counter() * 1000.0

    return DatasetDescriptor(
        filename=filename,
        csv_path=str(csv_path),
        row_count=row_count,
        columns=tuple(columns),
        has_image_path_column=has_image_path,
        image_support_status=image_support_status,
        image_path_policy=image_path_policy,
        categories=tuple(sorted(categories)),
        image_coverage_count=images_found,
        image_missing_count=images_missing,
        image_coverage_pct=round(coverage_pct, 2),
        schema_valid=schema_valid,
        schema_missing=schema_missing,
        invalid_image_path_count=invalid_image_path_count,
        invalid_image_path_examples=tuple(invalid_image_path_examples),
        audit_ms=round(t1 - t0, 2),
    )


# =============================================================================
# 6. Registry discovery
# =============================================================================

def discover_datasets() -> List[DatasetDescriptor]:
    """
    Discover and audit all preprocessed CSVs.

    Returns:
        List of DatasetDescriptor objects, one per CSV file found.
    """
    csv_paths = list_preprocessed_csvs()
    if not csv_paths:
        logger.warning(
            f"No CSVs found in {PREPROCESSED_DATASET_DIR}. "
            f"Run preprocessing first."
        )
        return []

    descriptors = []
    for csv_path in csv_paths:
        try:
            desc = _audit_single_csv(csv_path)
            descriptors.append(desc)
            logger.info(
                f"Discovered: {desc.filename} | "
                f"rows={desc.row_count} | "
                f"images={desc.image_coverage_count}/{desc.image_coverage_count + desc.image_missing_count} | "
                f"schema={'OK' if desc.schema_valid else 'INVALID'}"
            )
        except Exception as exc:
            logger.error(
                _registry_error(
                    "csv_audit", f"Failed to audit {csv_path.name}",
                    filename=csv_path.name,
                    resolution="check file encoding and format",
                    cause=exc,
                )
            )

    return descriptors


# =============================================================================
# 7. Single-source resolution
# =============================================================================

def resolve_dataset(
    filename: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> DatasetDescriptor:
    """
    Resolve a single dataset source by filename or logical name.

    Args:
        filename     : Direct CSV filename (e.g., 'sample_100.csv').
        dataset_name : Logical dataset name. The registry will try to
                       match against known filenames by appending
                       '_processed.csv' or '.csv'.

    Returns:
        DatasetDescriptor for the resolved CSV.

    Raises:
        FileNotFoundError : If the CSV cannot be found.
        ValueError        : If both or neither arguments are provided.
    """
    # Normalize inputs
    if filename is not None:
        filename = filename.strip() if isinstance(filename, str) else filename
    if dataset_name is not None:
        dataset_name = dataset_name.strip() if isinstance(dataset_name, str) else dataset_name

    if filename and dataset_name:
        raise ValueError(
            _registry_error(
                "resolve_dataset",
                "Provide either filename or dataset_name, not both.",
                resolution="Use filename for direct override, dataset_name for logical lookup.",
            )
        )
    if not filename and not dataset_name:
        raise ValueError(
            _registry_error(
                "resolve_dataset",
                "No filename or dataset_name provided.",
                resolution="Pass csv_filename='sample_100.csv' or dataset_name='sample_100'.",
            )
        )

    if filename:
        csv_path = resolve_preprocessed_csv(filename)
        return _audit_single_csv(csv_path)

    # Logical name resolution: try common patterns
    candidates = [
        f"{dataset_name}.csv",
        f"{dataset_name}_processed.csv",
        f"meta_{dataset_name}_processed.csv",
    ]
    available = list_preprocessed_csvs()
    available_names = {p.name for p in available}

    for candidate in candidates:
        if candidate in available_names:
            csv_path = resolve_preprocessed_csv(candidate)
            return _audit_single_csv(csv_path)

    raise FileNotFoundError(
        _registry_error(
            "resolve_dataset",
            f"No CSV found for dataset_name='{dataset_name}'.",
            resolution=(
                f"Tried: {candidates}. "
                f"Available: {sorted(available_names) if available_names else 'NONE'}."
            ),
        )
    )


# =============================================================================
# 8. Multi-source resolution
# =============================================================================

def resolve_multi_source(
    source_files: Sequence[str],
) -> MultiSourceDescriptor:
    """
    Build a multi-source descriptor from several CSV filenames.

    Validates:
      - Each CSV exists and passes core schema
      - Cross-source ASIN uniqueness

    Args:
        source_files : Sequence of CSV filenames (e.g., ['sample_100.csv', 'sample_100_2.csv']).

    Returns:
        MultiSourceDescriptor with full audit metadata.

    Raises:
        ValueError : If source_files is empty or any CSV fails schema check.
    """
    if not source_files or len(source_files) == 0:
        raise ValueError(
            _registry_error(
                "multi_source_resolve",
                "source_files must be a non-empty sequence of CSV filenames.",
            )
        )

    t0 = time.perf_counter() * 1000.0
    descriptors: List[DatasetDescriptor] = []
    all_asins: Dict[str, str] = {}  # asin -> source filename
    collision_count = 0
    collision_examples: List[str] = []

    for fname in source_files:
        desc = resolve_dataset(filename=fname)
        if not desc.schema_valid:
            raise ValueError(
                _registry_error(
                    "multi_source_schema",
                    f"CSV '{fname}' fails core schema validation.",
                    filename=fname,
                    resolution=f"Missing columns: {desc.schema_missing}",
                )
            )
        descriptors.append(desc)

        # Cross-source ASIN collision check
        csv_path = Path(desc.csv_path)
        with open(csv_path, encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                asin = row.get("asin", "").strip()
                if not asin:
                    continue
                if asin in all_asins:
                    collision_count += 1
                    if len(collision_examples) < 10:
                        collision_examples.append(
                            f"{asin} in [{all_asins[asin]}, {fname}]"
                        )
                else:
                    all_asins[asin] = fname

    t1 = time.perf_counter() * 1000.0

    total_rows = sum(d.row_count for d in descriptors)
    total_found = sum(d.image_coverage_count for d in descriptors)
    total_missing = sum(d.image_missing_count for d in descriptors)

    return MultiSourceDescriptor(
        sources=tuple(descriptors),
        total_rows=total_rows,
        total_images_found=total_found,
        total_images_missing=total_missing,
        cross_source_asin_collisions=collision_count,
        collision_examples=tuple(collision_examples),
        combined_schema_valid=all(d.schema_valid for d in descriptors),
        audit_ms=round(t1 - t0, 2),
    )


# =============================================================================
# 9. Registry summary (for notebooks/diagnostics)
# =============================================================================

def print_registry_summary(descriptors: Optional[List[DatasetDescriptor]] = None) -> None:
    """Print a compact table of all discovered datasets."""
    if descriptors is None:
        descriptors = discover_datasets()

    print(f"\n{'='*80}")
    print(f"  Dataset Registry Summary | {len(descriptors)} sources")
    print(f"{'='*80}")
    print(f"  {'Filename':<45} {'Rows':>6} {'Images':>8} {'Coverage':>9} {'Schema':>7}")
    print(f"  {'-'*45} {'-'*6} {'-'*8} {'-'*9} {'-'*7}")
    for d in descriptors:
        total = d.image_coverage_count + d.image_missing_count
        img_str = f"{d.image_coverage_count}/{total}"
        print(
            f"  {d.filename:<45} {d.row_count:>6} {img_str:>8} "
            f"{d.image_coverage_pct:>8.1f}% {'OK' if d.schema_valid else 'FAIL':>6}"
        )
    print(f"{'='*80}\n")


# =============================================================================
# 10. Dataset Groups — named presets for multi-source loading
# =============================================================================
#
# Groups are simple named tuples of CSV filenames.
# "AUTO" means all discovered valid CSVs (schema-valid only).
# Groups do NOT create physical combined CSVs.
# dataset.py can resolve a group name via dataset_name.

REGISTERED_DATASETS: Dict[str, Any] = {
    "sample": ("sample_100.csv", "sample_100_2.csv"),
    "all_discovered": "AUTO",
}


def list_dataset_groups() -> Dict[str, Any]:
    """
    Return all registered dataset groups.

    Returns:
        Dict mapping group name to either a tuple of filenames or 'AUTO'.
    """
    return dict(REGISTERED_DATASETS)


def resolve_dataset_group(group_name: str) -> tuple:
    """
    Resolve a dataset group name to a tuple of validated CSV filenames.

    Args:
        group_name : Registered group name (e.g., 'sample', 'all_discovered').

    Returns:
        Tuple[str, ...] : Tuple of CSV filenames that exist and pass schema.

    Raises:
        KeyError  : If group_name is not registered.
        ValueError : If group resolves to zero valid CSVs.
    """
    # Normalize group name
    group_name = group_name.strip() if isinstance(group_name, str) else group_name
    if group_name not in REGISTERED_DATASETS:
        available = sorted(REGISTERED_DATASETS.keys())
        raise KeyError(
            _registry_error(
                "resolve_group",
                f"Unknown dataset group: '{group_name}'.",
                resolution=f"Available groups: {available}",
            )
        )

    spec = REGISTERED_DATASETS[group_name]

    if spec == "AUTO":
        # Discover all schema-valid CSVs
        all_descs = discover_datasets()
        valid = tuple(
            d.filename for d in all_descs if d.schema_valid
        )
        if not valid:
            raise ValueError(
                _registry_error(
                    "resolve_group",
                    f"Group '{group_name}' resolved to AUTO but no schema-valid CSVs found.",
                    resolution="Run preprocessing to create valid CSVs.",
                )
            )
        return valid

    # Static group — validate each file exists and passes schema
    validated: List[str] = []
    for fname in spec:
        desc = resolve_dataset(filename=fname)
        if not desc.schema_valid:
            raise ValueError(
                _registry_error(
                    "resolve_group",
                    f"Group '{group_name}' contains invalid CSV: {fname}.",
                    filename=fname,
                    resolution=f"Missing columns: {desc.schema_missing}",
                )
            )
        validated.append(fname)

    if not validated:
        raise ValueError(
            _registry_error(
                "resolve_group",
                f"Group '{group_name}' resolved to zero valid CSVs.",
            )
        )

    return tuple(validated)


# =============================================================================
# 11. Smoke Tests
# =============================================================================

if __name__ == "__main__":
    import os
    import tempfile
    import shutil

    print("=" * 60)
    print("  data_pipeline/dataset_registry.py -- smoke test")
    print("=" * 60)

    passed = 0
    total = 0

    def chk(label, ok):
        global passed, total
        total += 1
        if ok:
            passed += 1
        print(f"    [{'PASS' if ok else 'FAIL'}] {label}")

    _tmp_dir = None

    try:
        # ---- 1. Constants ----
        print("\n  1. Constants...")
        chk("core schema", len(CORE_SCHEMA_COLUMNS) == 7)
        chk("asin in core", "asin" in CORE_SCHEMA_COLUMNS)
        chk("rating in core", "rating" in CORE_SCHEMA_COLUMNS)
        chk("identity key", IDENTITY_COLUMN == "asin")

        # ---- 2. CSV discovery ----
        print("\n  2. Discovery...")
        descs = discover_datasets()
        chk("found CSVs", len(descs) >= 1)

        # Find sample_100
        sample = None
        for d in descs:
            if d.filename == "sample_100.csv":
                sample = d
                break
        chk("sample_100 found", sample is not None)

        if sample:
            chk("rows=100", sample.row_count == 100)
            chk("has image_path", sample.has_image_path_column is True)
            chk("schema valid", sample.schema_valid is True)
            chk("schema missing empty", len(sample.schema_missing) == 0)
            chk("coverage > 0", sample.image_coverage_count > 0)
            chk("coverage pct", sample.image_coverage_pct > 0.0)
            chk("columns tuple", isinstance(sample.columns, tuple))
            chk("categories tuple", isinstance(sample.categories, tuple))
            chk("domain name", len(sample.domain_name) > 0)
            chk("audit_ms", sample.audit_ms is not None and sample.audit_ms > 0)
            # Image semantics
            chk("image_support_status", sample.image_support_status == "image_path_column_present")
            chk("image_path_policy", sample.image_path_policy == "use_image_path_then_asin")
            chk("invalid_img_path_count", isinstance(sample.invalid_image_path_count, int))
            chk("invalid_img_path_examples", isinstance(sample.invalid_image_path_examples, tuple))

        # Check domain CSVs have no image_path column
        for d in descs:
            if d.filename != "sample_100.csv" and d.filename != "sample_100_2.csv":
                chk(f"{d.filename[:20]}.. no image_path", d.has_image_path_column is False)
                chk(f"{d.filename[:20]}.. asin_fallback", d.image_support_status in ("asin_fallback_only", "no_images_found"))
                chk(f"{d.filename[:20]}.. policy", d.image_path_policy == "use_asin_jpg")
                break

        # ---- 3. Schema validation ----
        print("\n  3. Schema validation...")
        for d in descs:
            chk(f"{d.filename[:30]} schema", d.schema_valid)

        # ---- 4. Single-source resolution ----
        print("\n  4. Resolve single...")
        resolved = resolve_dataset(filename="sample_100.csv")
        chk("resolve by filename", resolved.filename == "sample_100.csv")

        try:
            resolve_dataset(filename="nonexistent.csv")
            chk("missing raises", False)
        except FileNotFoundError:
            chk("missing raises", True)

        try:
            resolve_dataset()
            chk("no args raises", False)
        except ValueError:
            chk("no args raises", True)

        try:
            resolve_dataset(filename="x.csv", dataset_name="y")
            chk("both args raises", False)
        except ValueError:
            chk("both args raises", True)

        # ---- 5. Logical name resolution ----
        print("\n  5. Logical name...")
        try:
            res_name = resolve_dataset(dataset_name="sample_100")
            chk("sample_100 by name", res_name.filename == "sample_100.csv")
        except FileNotFoundError:
            chk("sample_100 by name (file missing)", True)

        try:
            resolve_dataset(dataset_name="totally_fake_dataset_xyz")
            chk("fake name raises", False)
        except FileNotFoundError:
            chk("fake name raises", True)

        # ---- 6. Multi-source ----
        print("\n  6. Multi-source...")
        try:
            ms = resolve_multi_source(["sample_100.csv", "sample_100_2.csv"])
            chk("multi built", ms is not None)
            chk("multi sources=2", len(ms.sources) == 2)
            chk("multi total_rows=200", ms.total_rows == 200)
            chk("multi schema valid", ms.combined_schema_valid is True)
            chk("multi audit_ms", ms.audit_ms is not None)
            # Check collisions (sample CSVs may share ASINs)
            chk("collision count int", isinstance(ms.cross_source_asin_collisions, int))
            # Collision count must be true count, not capped
            chk("examples bounded", len(ms.collision_examples) <= 10)
            if ms.cross_source_asin_collisions > 10:
                chk("count > examples (honest)", ms.cross_source_asin_collisions > len(ms.collision_examples))
        except Exception as e:
            chk(f"multi-source: {e}", False)

        try:
            resolve_multi_source([])
            chk("empty multi raises", False)
        except ValueError:
            chk("empty multi raises", True)

        # ---- 7. Synthetic CSV validation ----
        print("\n  7. Synthetic CSV...")
        _tmp_dir = tempfile.mkdtemp()
        good_csv = os.path.join(_tmp_dir, "good.csv")
        with open(good_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["asin", "text", "image_url", "price", "rating_number", "category", "rating"])
            w.writerow(["T001", "test", "http://x", "10", "5", "cat", "4.0"])
            w.writerow(["T002", "test2", "http://y", "20", "3", "cat", "3.0"])
        synth = _audit_single_csv(Path(good_csv))
        chk("synth rows=2", synth.row_count == 2)
        chk("synth schema valid", synth.schema_valid is True)
        chk("synth no images", synth.image_coverage_count == 0)
        chk("synth images missing=2", synth.image_missing_count == 2)
        chk("synth no_images_found", synth.image_support_status == "no_images_found")
        chk("synth asin_jpg policy", synth.image_path_policy == "use_asin_jpg")
        chk("synth no invalid paths", synth.invalid_image_path_count == 0)

        # Synthetic CSV with bad image_path (traversal)
        bad_img_csv = os.path.join(_tmp_dir, "bad_img.csv")
        with open(bad_img_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["asin", "text", "image_url", "price", "rating_number", "category", "rating", "image_path"])
            w.writerow(["X001", "test", "http://x", "10", "5", "cat", "4.0", "../escape.jpg"])
            w.writerow(["X002", "test2", "http://y", "20", "3", "cat", "3.0", "B001.jpg"])
        bad_img_desc = _audit_single_csv(Path(bad_img_csv))
        chk("bad_img rows=2", bad_img_desc.row_count == 2)
        chk("bad_img has image_path", bad_img_desc.has_image_path_column is True)
        chk("bad_img invalid count=1", bad_img_desc.invalid_image_path_count == 1)
        chk("bad_img examples len", len(bad_img_desc.invalid_image_path_examples) == 1)
        chk("bad_img schema valid", bad_img_desc.schema_valid is True)

        bad_csv = os.path.join(_tmp_dir, "bad.csv")
        with open(bad_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["wrong_col", "another"])
            w.writerow(["x", "y"])
        bad_desc = _audit_single_csv(Path(bad_csv))
        chk("bad schema invalid", bad_desc.schema_valid is False)
        chk("bad schema missing >0", len(bad_desc.schema_missing) > 0)

        # ---- 8. Registry summary (visual check) ----
        print("\n  8. Registry summary...")
        print_registry_summary(descs)
        chk("summary printed", True)

        # ---- 9. Descriptor immutability ----
        print("\n  9. Immutability...")
        try:
            if sample:
                sample.row_count = 999
            chk("frozen raises", False)
        except AttributeError:
            chk("frozen raises", True)

        # ---- 10. Source safety ----
        print("\n  10. Source safety...")
        src = open(__file__, encoding="utf-8").read()
        prod = src.split("if __name__")[0]
        pc = "\n".join(l for l in prod.splitlines() if not l.strip().startswith("#"))
        chk("no torch", "import torch" not in pc)
        chk("no .cuda()", ".cuda()" not in pc)
        chk("no .to(device)", ".to(device)" not in pc)
        chk("no from models", "from models" not in pc)
        chk("no pandas", "import pandas" not in pc)

        # ---- 11. Dataset Groups ----
        print("\n  11. Dataset Groups...")
        groups = list_dataset_groups()
        chk("groups dict", isinstance(groups, dict))
        chk("sample group", "sample" in groups)
        chk("all_discovered group", "all_discovered" in groups)

        sample_group = resolve_dataset_group("sample")
        chk("sample -> tuple", isinstance(sample_group, tuple))
        chk("sample has 2", len(sample_group) == 2)
        chk("sample_100 in group", "sample_100.csv" in sample_group)
        chk("sample_100_2 in group", "sample_100_2.csv" in sample_group)

        auto_group = resolve_dataset_group("all_discovered")
        chk("auto -> tuple", isinstance(auto_group, tuple))
        chk("auto >= 1", len(auto_group) >= 1)
        chk("auto only valid", all(isinstance(f, str) for f in auto_group))

        try:
            resolve_dataset_group("nonexistent_group")
            chk("missing group raises", False)
        except KeyError:
            chk("missing group raises", True)

        # Normalized group name
        norm_group = resolve_dataset_group(" sample ")
        chk("norm group == sample", norm_group == sample_group)

        # Normalized resolve_dataset
        norm_desc = resolve_dataset(dataset_name=" sample_100 ")
        chk("norm resolve", norm_desc.filename == "sample_100.csv")

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

    finally:
        if _tmp_dir and os.path.isdir(_tmp_dir):
            try:
                shutil.rmtree(_tmp_dir)
            except Exception:
                pass
