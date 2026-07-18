# =============================================================================
# data_pipeline/dataset.py
# Multimodal Sample Integrity Authority -- Multimodal AI Pipeline
# =============================================================================
#
# Ownership (this file ONLY):
#   - CSV loading via configs.paths
#   - Dataset schema validation
#   - Row identity and deterministic indexing
#   - Sample construction: image + text + tabular + rating
#   - Image path resolution
#   - Calls into data_pipeline.transforms (image preprocessing)
#   - Calls into data_pipeline.tokenization (text preprocessing)
#   - Tabular feature extraction and tensor conversion
#   - Sample-level metadata, timing, and trace events
#   - Dry-run sample validation
#
# What this file does NOT own:
#   +-----------------------------+---------------------------+
#   | Responsibility              | Correct File              |
#   +-----------------------------+---------------------------+
#   | model forward passes        | models/*.py               |
#   | GPU transfer / .cuda()      | train.py                  |
#   | training loops              | train.py                  |
#   | collation / batch assembly  | collate.py                |
#   | optimizers / losses         | train.py                  |
#   | async queues / CUDA streams | train.py                  |
#   | transform definitions       | data_pipeline/transforms  |
#   | tokenizer definitions       | data_pipeline/tokenization|
#   | fusion logic                | models/fusion.py          |
#   +-----------------------------+---------------------------+
#
# Design:
#   STATELESS    -- no mutable shared state across workers
#   CPU-ONLY     -- no .cuda(), no .to(device)
#   LAZY         -- images/tokens loaded per-sample, not preloaded
#   DETERMINISTIC-- same index always returns same identity
#   WORKER-SAFE  -- each DataLoader worker gets own instance copy
# =============================================================================

import sys
import math
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------
# Project Routing
# ---------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _THIS_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

try:
    from configs.paths import (
        IMAGE_DATASET_DIR,
        get_dataset_csv,
        resolve_image_file,
    )
except ImportError as _err:
    raise RuntimeError(
        "ROUTING FAILURE: Cannot import configs.paths. "
        f"sys.path: {sys.path[:5]}..."
    ) from _err

# ---------------------------------------------------------------
# Data pipeline authorities (lazy torch inside functions)
# ---------------------------------------------------------------
from data_pipeline.transforms import safe_load_image, get_transforms
import torch.utils.data as _torch_data
from data_pipeline.tokenization import (
    sanitize_text,
    tokenize_text,
    validate_tokenized_output,
    load_tokenizer,
    FALLBACK_TEXT,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Timing helper
# =============================================================================

def _now_ms() -> float:
    return time.perf_counter() * 1000.0


# =============================================================================
# Error helper
# =============================================================================

def _dataset_error(
    stage: str,
    sample_id: str,
    row_index: int,
    asin: str,
    expected: str,
    received: str,
    origin: str = "dataset.py",
    resolution: str = "",
    cause: Optional[Exception] = None,
) -> str:
    msg = (
        f"[DATASET ERROR]\n"
        f"  Stage     : {stage}\n"
        f"  Sample ID : {sample_id}\n"
        f"  Row index : {row_index}\n"
        f"  ASIN      : {asin}\n"
        f"  Expected  : {expected}\n"
        f"  Received  : {received}\n"
        f"  Origin    : {origin}"
    )
    if resolution:
        msg += f"\n  Resolution: {resolution}"
    if cause:
        msg += f"\n  Cause     : {type(cause).__name__}: {cause}"
    return msg


# =============================================================================
# DatasetConfig
# =============================================================================

@dataclass
class DatasetConfig:
    """
    Configuration for MultimodalProductDataset.

    Routing priority:
      1. source_files  -> multi-source dataset (concatenated CSVs)
      2. dataset_name  -> logical registry lookup
      3. csv_filename  -> direct raw CSV override (default)
    """

    # -- Source routing (priority order) --
    csv_filename: str = "sample_100.csv"
    dataset_name: Optional[str] = None
    source_files: Optional[Sequence[str]] = None

    # -- Dataset behavior --
    mode: str = "train"
    text_max_length: int = 64
    image_strict: bool = False
    strict_tabular: bool = False
    debug_trace: bool = True
    enable_timing: bool = True
    tabular_columns: Tuple[str, ...] = ("price", "rating_number")
    category_column: str = "category"
    image_path_column: str = "image_path"
    text_column: str = "text"
    target_column: str = "rating"
    asin_column: str = "asin"

    def __post_init__(self):
        # -- csv_filename: strip and validate --
        if not isinstance(self.csv_filename, str) or not self.csv_filename.strip():
            raise ValueError(f"csv_filename must be non-empty str, got {self.csv_filename!r}")
        self.csv_filename = self.csv_filename.strip()
        # -- dataset_name: strip and validate --
        if self.dataset_name is not None:
            if not isinstance(self.dataset_name, str) or not self.dataset_name.strip():
                raise ValueError(f"dataset_name must be non-empty str or None, got {self.dataset_name!r}")
            self.dataset_name = self.dataset_name.strip()
        # -- source_files: consume iterables, normalize to tuple, strip, reject dupes --
        if self.source_files is not None:
            if not hasattr(self.source_files, '__iter__') or isinstance(self.source_files, str):
                raise TypeError(f"source_files must be a sequence of str or None, got {type(self.source_files).__name__}")
            sf_list = list(self.source_files)  # safely consumes generators
            if len(sf_list) == 0:
                raise ValueError("source_files must be non-empty if provided.")
            for i, sf in enumerate(sf_list):
                if not isinstance(sf, str) or not sf.strip():
                    raise ValueError(f"source_files[{i}] must be non-empty str, got {sf!r}")
            sf_normalized = tuple(sf.strip() for sf in sf_list)
            # Reject duplicate filenames early
            seen = set()
            for sf in sf_normalized:
                if sf in seen:
                    raise ValueError(
                        f"source_files contains duplicate filenames: {sf}. "
                        f"Each source must be unique."
                    )
                seen.add(sf)
            self.source_files = sf_normalized
        # -- mode --
        if not isinstance(self.mode, str):
            raise TypeError(f"mode must be str, got {type(self.mode).__name__}: {self.mode!r}")
        self.mode = self.mode.strip().lower()
        if self.mode not in ("train", "eval", "test", "inference"):
            raise ValueError(f"mode must be train/eval/test/inference, got {self.mode!r}")
        # -- text_max_length --
        if isinstance(self.text_max_length, bool) or not isinstance(self.text_max_length, int):
            raise TypeError(f"text_max_length must be positive int, got {type(self.text_max_length).__name__}: {self.text_max_length!r}")
        if self.text_max_length <= 0:
            raise ValueError(f"text_max_length must be > 0, got {self.text_max_length}")
        # -- tabular_columns --
        if not isinstance(self.tabular_columns, tuple) or len(self.tabular_columns) == 0:
            raise TypeError(f"tabular_columns must be non-empty tuple of str, got {self.tabular_columns!r}")
        for i, col in enumerate(self.tabular_columns):
            if not isinstance(col, str) or not col.strip():
                raise ValueError(f"tabular_columns[{i}] must be non-empty str, got {col!r}")
        # -- column name fields --
        for field_name in ("category_column", "image_path_column", "text_column", "target_column", "asin_column"):
            val = getattr(self, field_name)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"{field_name} must be non-empty str, got {val!r}")
        # -- boolean fields --
        for field_name in ("image_strict", "strict_tabular", "debug_trace", "enable_timing"):
            val = getattr(self, field_name)
            if not isinstance(val, bool):
                raise TypeError(f"{field_name} must be bool, got {type(val).__name__}: {val!r}")


# =============================================================================
# Image path resolution
# =============================================================================

def resolve_image_path(
    row, asin: str, image_path_column: str, has_image_path_column: bool = True
) -> Tuple[str, bool, str]:
    """
    Resolve image path from row.

    Returns:
        (resolved_path_str, used_fallback, reason)

    Path routing:
      - If image_path column is missing entirely -> ASIN fallback (safe).
      - If raw path is missing/empty/NaN         -> ASIN fallback (safe).
      - If raw path is absolute inside IMAGE_DATASET_DIR -> use as-is.
      - If raw path is relative and passes traversal check -> resolve.
      - If raw path is traversal/unsafe/absolute outside base -> FATAL.
    """
    # If column doesn't exist, go straight to ASIN fallback
    if not has_image_path_column:
        if not asin or not str(asin).strip():
            raise ValueError(
                f"[DATASET ERROR] Cannot construct fallback image path: "
                f"ASIN is empty/missing. image_path_column='{image_path_column}'"
            )
        fallback = str(resolve_image_file(asin))
        return fallback, True, "image_path_column_missing_asin_fallback"

    raw = row.get(image_path_column) if isinstance(row, dict) else getattr(row, image_path_column, None)

    # -- Handle missing/NaN: ASIN fallback is safe --
    raw_is_missing = (
        raw is None
        or (isinstance(raw, str) and not raw.strip())
        or (isinstance(raw, float) and math.isnan(raw))
    )
    if not raw_is_missing:
        # Also check pandas NA
        try:
            import pandas as pd
            if pd.isna(raw):
                raw_is_missing = True
        except (ImportError, TypeError, ValueError):
            pass
    if raw_is_missing:
        if not asin or not str(asin).strip():
            raise ValueError(
                f"[DATASET ERROR] Cannot construct fallback image path: "
                f"ASIN is empty/missing. image_path_column='{image_path_column}'"
            )
        fallback = str(resolve_image_file(asin))
        return fallback, True, "asin_fallback"

    # -- Convert to string and validate --
    raw_str = str(raw).strip() if raw is not None else ""
    if not raw_str:
        # Empty string after conversion: ASIN fallback
        fallback = str(resolve_image_file(asin))
        return fallback, True, "asin_fallback"

    # -- Non-empty path present: MUST be safe or fail loudly --
    p = Path(raw_str)
    if p.is_absolute():
        # Absolute paths: verify inside IMAGE_DATASET_DIR
        resolved = p.resolve()
        img_base = IMAGE_DATASET_DIR.resolve()
        if resolved == img_base or img_base in resolved.parents:
            return str(resolved), False, "absolute_path"
        # Absolute outside IMAGE_DATASET_DIR: FATAL
        raise ValueError(
            f"[DATASET ERROR] Stage: image_path_resolution\n"
            f"  ASIN            : {asin}\n"
            f"  Column          : {image_path_column}\n"
            f"  Received path   : {raw_str}\n"
            f"  Resolved to     : {resolved}\n"
            f"  Expected base   : {img_base}\n"
            f"  Error           : Absolute image path outside IMAGE_DATASET_DIR.\n"
            f"  Resolution      : Use relative paths or move images into IMAGE_DATASET_DIR."
        )

    # Relative path -> resolve via resolve_image_file (traversal-safe)
    # If traversal is detected, this MUST be fatal, not a silent fallback.
    try:
        resolved = resolve_image_file(raw_str)
        return str(resolved), False, "relative_resolved"
    except ValueError as exc:
        # Traversal or malformed path: FATAL with full context
        raise ValueError(
            f"[DATASET ERROR] Stage: image_path_resolution\n"
            f"  ASIN            : {asin}\n"
            f"  Column          : {image_path_column}\n"
            f"  Received path   : {raw_str}\n"
            f"  Error           : Unsafe image path blocked by traversal guard.\n"
            f"  Detail          : {exc}\n"
            f"  Resolution      : Fix the image_path value in the source CSV. "
            f"Use simple filenames like 'B001.jpg', not '../escape.jpg'."
        ) from exc


# =============================================================================
# Text missing-value detection (Fix #4)
# =============================================================================

def _is_missing_text(value: Any) -> bool:
    """
    Returns True if value represents missing/invalid text that
    sanitize_text() would convert to FALLBACK_TEXT.

    Detects: None, float NaN, pandas NA, empty string, whitespace-only.
    Does NOT treat the literal string "nan" as missing.
    """
    if value is None:
        return True
    if isinstance(value, float):
        try:
            if math.isnan(value):
                return True
        except (TypeError, ValueError):
            pass
    # pandas NA check (safe without importing pandas)
    try:
        import pandas as pd
        if pd.isna(value):
            return True
    except (ImportError, TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.strip() == ""
    return False


# =============================================================================
# MultimodalProductDataset
# =============================================================================

class MultimodalProductDataset(_torch_data.Dataset):
    """
    Multimodal sample integrity authority.

    Constructs synchronized (image, text, tabular, rating) samples from a
    preprocessed CSV. Each __getitem__ call returns a stable dictionary with
    full trace metadata. Stateless, CPU-only, DataLoader-worker safe.
    """

    def __init__(self, config: Optional[DatasetConfig] = None) -> None:
        import torch
        import pandas as pd

        if config is None:
            config = DatasetConfig()
        self.config = config

        # -- Route CSV source --------------------------------------------------
        # Priority: source_files > dataset_name > csv_filename
        self._source_files: List[str] = []  # tracks source per-row for metadata

        if config.source_files is not None:
            # Multi-source: load each, concatenate, validate ASIN uniqueness
            frames: List[pd.DataFrame] = []
            for fname in config.source_files:
                csv_path = get_dataset_csv(fname)
                logger.info(f"Loading source CSV: {csv_path}")
                frame = pd.read_csv(csv_path)
                frame["_source_file"] = fname
                frame["_source_index"] = len(frames)
                frames.append(frame)
            # Current in-memory concat is intentional for current scale (6k-100k rows).
            # Future large-scale option: lazy indexing / ConcatDataset / parquet shards.
            # Do not write physical combined CSVs — multi-source stays in-memory only.
            self.df = pd.concat(frames, ignore_index=True)
            self._source_files = list(config.source_files)
            logger.info(
                f"Multi-source loaded | sources={len(frames)} | "
                f"total_rows={len(self.df)}"
            )
            # Cross-source ASIN uniqueness
            asin_col = self.df[config.asin_column]
            dup_mask = asin_col.duplicated(keep=False)
            dup_count = dup_mask.sum()
            if dup_count > 0:
                examples = asin_col[dup_mask].unique()[:5].tolist()
                raise ValueError(
                    f"[DATASET ERROR] {dup_count} duplicate ASINs across "
                    f"{len(frames)} source files (source_files routing).\n"
                    f"  Duplicate examples : {examples}\n"
                    f"  Source files       : {list(config.source_files)}\n"
                    f"  Multi-source datasets require globally unique ASINs.\n"
                    f"  Resolution: deduplicate CSVs, use non-overlapping sources, "
                    f"or use dataset_name='sample' for current smoke training.\n"
                    f"  Split/dedup policy is intentionally deferred."
                )
        elif config.dataset_name is not None:
            # Registry lookup: check groups first, then single-file
            from data_pipeline.dataset_registry import (
                resolve_dataset, resolve_dataset_group, REGISTERED_DATASETS,
            )
            if config.dataset_name in REGISTERED_DATASETS:
                # Group resolution -> multi-source
                group_files = resolve_dataset_group(config.dataset_name)
                logger.info(
                    f"Group '{config.dataset_name}' resolved to {len(group_files)} sources: {group_files}"
                )
                frames_g: List[pd.DataFrame] = []
                for gf in group_files:
                    gcsv = get_dataset_csv(gf)
                    gframe = pd.read_csv(gcsv)
                    gframe["_source_file"] = gf
                    gframe["_source_index"] = len(frames_g)
                    frames_g.append(gframe)
                self.df = pd.concat(frames_g, ignore_index=True)
                self._source_files = list(group_files)
                # Cross-source ASIN uniqueness (same check as source_files)
                asin_col_g = self.df[config.asin_column]
                dup_mask_g = asin_col_g.duplicated(keep=False)
                dup_count_g = dup_mask_g.sum()
                if dup_count_g > 0:
                    examples_g = asin_col_g[dup_mask_g].unique()[:5].tolist()
                    raise ValueError(
                        f"[DATASET ERROR] {dup_count_g} duplicate ASINs across group "
                        f"'{config.dataset_name}' ({len(frames_g)} source files).\n"
                        f"  Duplicate examples : {examples_g}\n"
                        f"  Group sources      : {list(group_files)}\n"
                        f"  Resolution: Use dataset_name='sample' for current smoke "
                        f"training, or create a non-overlapping registry group.\n"
                        f"  Split/dedup policy is intentionally deferred."
                    )
            else:
                # Single-file logical lookup
                desc = resolve_dataset(dataset_name=config.dataset_name)
                csv_path = Path(desc.csv_path)
                logger.info(f"Registry resolved: {config.dataset_name} -> {csv_path}")
                self.df = pd.read_csv(csv_path)
                self._source_files = [desc.filename]
        else:
            # Direct CSV override (original behavior)
            csv_path = get_dataset_csv(config.csv_filename)
            logger.info(f"Loading CSV: {csv_path}")
            self.df = pd.read_csv(csv_path)
            self._source_files = [config.csv_filename]

        self._num_rows = len(self.df)
        logger.info(f"CSV loaded | rows={self._num_rows} | columns={list(self.df.columns)}")

        # -- Schema validation -------------------------------------------------
        self.validate_schema()

        # -- Build transform (stateless callable, one per instance) ------------
        self._transform = get_transforms(mode=config.mode)

        # -- Tokenizer: lazy-loaded on first sample (worker-local) --------------
        self._tokenizer = None

        logger.info(
            f"MultimodalProductDataset ready | mode={config.mode} | "
            f"rows={self._num_rows} | tabular={config.tabular_columns}"
        )

    # =========================================================================
    # Schema validation
    # =========================================================================

    def validate_schema(self) -> None:
        """Validate CSV schema before any sample is built."""
        cfg = self.config
        cols = set(self.df.columns)

        # Required columns
        required = {cfg.asin_column, cfg.text_column, cfg.target_column}
        required.update(cfg.tabular_columns)
        missing = required - cols
        if missing:
            raise ValueError(
                f"[SCHEMA ERROR] Missing required columns: {sorted(missing)}. "
                f"Available: {sorted(cols)}. "
                f"Resolution: check CSV or DatasetConfig column names."
            )

        # Non-empty
        if self._num_rows == 0:
            raise ValueError("[SCHEMA ERROR] CSV has 0 rows.")

        # ASIN validity: reject None, NaN, empty, whitespace-only
        asin_col = self.df[cfg.asin_column]
        invalid_mask = asin_col.isna() | asin_col.astype(str).str.strip().eq("")
        # Also catch literal "nan" from float NaN coercion
        invalid_mask = invalid_mask | (asin_col.astype(str).str.strip().str.lower() == "nan")
        invalid_count = int(invalid_mask.sum())
        if invalid_count > 0:
            bad_rows = invalid_mask[invalid_mask].index[:10].tolist()
            raise ValueError(
                f"[SCHEMA ERROR] {invalid_count} invalid ASIN values found. "
                f"Rows: {bad_rows}. "
                f"ASIN is the sample identity key and cannot be missing, empty, or NaN. "
                f"Resolution: fix the CSV before building the dataset."
            )

        # ASIN uniqueness (within this CSV; cross-split leakage belongs to
        # future split orchestration, not dataset.py)
        dup_count = asin_col.duplicated().sum()
        if dup_count > 0:
            examples = asin_col[asin_col.duplicated(keep=False)].head(5).tolist()
            raise ValueError(
                f"[SCHEMA ERROR] {dup_count} duplicate ASIN values found. "
                f"Examples: {examples}. "
                f"Duplicates corrupt image identity and split integrity."
            )

        # Rating is numeric and finite
        import pandas as pd
        rating = self.df[cfg.target_column]
        if not pd.api.types.is_numeric_dtype(rating):
            raise ValueError(
                f"[SCHEMA ERROR] '{cfg.target_column}' is not numeric "
                f"(dtype={rating.dtype}). Rating must be numeric."
            )
        bad_mask = rating.isna() | rating.apply(lambda x: not math.isfinite(x) if isinstance(x, (int, float)) else True)
        bad_count = bad_mask.sum()
        if bad_count > 0:
            bad_idx = bad_mask[bad_mask].index[:5].tolist()
            raise ValueError(
                f"[SCHEMA ERROR] {bad_count} non-finite rating values. "
                f"First bad rows: {bad_idx}. Rating must be numeric and finite."
            )

        # Text column exists and is not entirely empty
        text_col = self.df[cfg.text_column]
        non_null = text_col.dropna()
        non_empty = non_null[non_null.astype(str).str.strip() != ""]
        if len(non_empty) == 0:
            raise ValueError(
                f"[SCHEMA ERROR] '{cfg.text_column}' is entirely empty/null."
            )

        logger.info("Schema validation PASSED.")

    # =========================================================================
    # Length
    # =========================================================================

    def __len__(self) -> int:
        return self._num_rows

    # =========================================================================
    # Lazy tokenizer (Fix #5 -- worker-local, loaded on first sample)
    # =========================================================================

    def _get_tokenizer(self):
        """
        Lazy-load tokenizer on first use.

        Each DataLoader worker gets its own dataset copy via fork/spawn,
        so each worker lazily loads its own tokenizer instance. No
        threading.Lock needed -- PyTorch workers are process-isolated.
        """
        if self._tokenizer is None:
            self._tokenizer = load_tokenizer()
        return self._tokenizer

    # =========================================================================
    # Trace event factory (Fix #8 -- identity in every event)
    # =========================================================================

    @staticmethod
    def _trace_event(
        stage: str, status: str, sample_id: str, row_index: int, asin: str,
        duration_ms: Optional[float] = None, message: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ev: Dict[str, Any] = {
            "stage": stage, "status": status,
            "sample_id": sample_id, "row_index": row_index, "asin": asin,
            "duration_ms": duration_ms,
        }
        if message:
            ev["message"] = message
        if details:
            # Guard: reject heavy objects that would leak memory at scale
            _SAFE = (str, int, float, bool, type(None))
            for k, v in details.items():
                if isinstance(v, (list, tuple)):
                    if len(v) > 20:
                        raise TypeError(
                            f"[TRACE ERROR] Trace details must remain lightweight. "
                            f"Stage: {stage} | Bad key: {k!r} | "
                            f"Received: sequence of length {len(v)}"
                        )
                elif isinstance(v, dict):
                    if len(v) > 20:
                        raise TypeError(
                            f"[TRACE ERROR] Trace details must remain lightweight. "
                            f"Stage: {stage} | Bad key: {k!r} | "
                            f"Received: dict of length {len(v)}"
                        )
                elif not isinstance(v, _SAFE):
                    type_name = type(v).__name__
                    raise TypeError(
                        f"[TRACE ERROR] Trace details must remain lightweight. "
                        f"Stage: {stage} | Bad key: {k!r} | "
                        f"Received: {type_name} (only scalar/str allowed)"
                    )
            ev["details"] = details
        return ev

    # =========================================================================
    # Image loader with trace (Fix #3 -- reliable fallback metadata)
    # =========================================================================

    def _load_image_with_trace(
        self, img_path: str, strict: bool, sample_id: str,
        row_index: int, asin: str, enable_timing: bool,
    ) -> Tuple[Any, float, str, Optional[str]]:
        """
        Load image with reliable fallback detection.

        Returns:
            (pil_img, duration_ms, status, fallback_reason_or_None)
        """
        path_obj = Path(img_path)
        path_exists = path_obj.exists() and path_obj.is_file()
        t0 = _now_ms() if enable_timing else None

        if not path_exists and not strict:
            # Known missing -- load via fallback, record clearly
            pil_img = safe_load_image(img_path, strict=False)
            dur = (_now_ms() - t0) if t0 else None
            return pil_img, dur, "fallback_used", f"image_file_missing: {img_path}"

        if path_exists and not strict:
            # File exists -- try strict first to detect corruption
            try:
                pil_img = safe_load_image(img_path, strict=True)
                dur = (_now_ms() - t0) if t0 else None
                return pil_img, dur, "ok", None
            except Exception:
                # Corrupt file -- fallback with clear reason
                pil_img = safe_load_image(img_path, strict=False)
                dur = (_now_ms() - t0) if t0 else None
                return pil_img, dur, "fallback_used", f"image_decode_failed: {img_path}"

        # strict=True: wrap with dataset-level error context
        try:
            pil_img = safe_load_image(img_path, strict=True)
        except Exception as exc:
            raise RuntimeError(
                _dataset_error(
                    stage="image_load", sample_id=sample_id,
                    row_index=row_index, asin=asin,
                    expected="valid decodable image file",
                    received=f"path={img_path}, error={exc}",
                    origin="dataset.py -> data_pipeline.transforms.safe_load_image",
                    resolution="verify file exists, is readable, and is a valid image",
                    cause=exc,
                )
            ) from exc
        dur = (_now_ms() - t0) if t0 else None
        return pil_img, dur, "ok", None

    # =========================================================================
    # Core sample builder (Fix #7 -- __getitem__ delegates here)
    # =========================================================================

    def _build_sample(
        self, index: int,
        debug_trace: Optional[bool] = None,
        enable_timing: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Build one synchronized multimodal sample.

        Args:
            index        : Row index into the DataFrame.
            debug_trace  : Override config.debug_trace if not None.
            enable_timing: Override config.enable_timing if not None.
        """
        import torch
        cfg = self.config
        do_trace = debug_trace if debug_trace is not None else cfg.debug_trace
        do_timing = enable_timing if enable_timing is not None else cfg.enable_timing
        trace: List[Dict[str, Any]] = []
        fallback_reasons: List[str] = []
        missing_modalities: List[str] = []
        total_start = _now_ms() if do_timing else None

        # -- Index validation --------------------------------------------------
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError(f"Index must be int, got {type(index).__name__}: {index!r}")
        if index < 0 or index >= self._num_rows:
            raise IndexError(
                f"Index {index} out of range [0, {self._num_rows}). "
                f"Dataset has {self._num_rows} samples."
            )

        # -- Row fetch ---------------------------------------------------------
        t0 = _now_ms() if do_timing else None
        row = self.df.iloc[index]
        asin = str(row[cfg.asin_column])
        sample_id = f"{index}:{asin}"
        t_row = (_now_ms() - t0) if t0 else None
        trace.append(self._trace_event("row_fetch", "ok", sample_id, index, asin, t_row))

        # == IMAGE =============================================================
        t0 = _now_ms() if do_timing else None
        img_path, img_fallback, resolve_reason = resolve_image_path(
            row, asin, cfg.image_path_column,
            has_image_path_column=(cfg.image_path_column in self.df.columns),
        )
        t_resolve = (_now_ms() - t0) if t0 else None
        trace.append(self._trace_event(
            "image_path_resolution",
            "fallback_used" if img_fallback else "ok",
            sample_id, index, asin, t_resolve,
            message=resolve_reason,
            details={"path": img_path},
        ))
        if img_fallback:
            fallback_reasons.append(f"image_path missing, fallback to {img_path}")

        # Load with reliable fallback detection
        pil_img, t_load, load_status, load_reason = self._load_image_with_trace(
            img_path, cfg.image_strict, sample_id, index, asin, do_timing,
        )
        trace.append(self._trace_event(
            "image_load", load_status, sample_id, index, asin, t_load,
            message=load_reason or "",
        ))
        if load_reason:
            fallback_reasons.append(load_reason)
            if "image" not in missing_modalities:
                missing_modalities.append("image")

        t0 = _now_ms() if do_timing else None
        try:
            image_tensor = self._transform(pil_img)
        except Exception as exc:
            raise RuntimeError(
                _dataset_error(
                    stage="image_transform", sample_id=sample_id,
                    row_index=index, asin=asin,
                    expected="PIL image -> Tensor(3,224,224)",
                    received=f"transform failed: {exc}",
                    origin="dataset.py -> data_pipeline.transforms",
                    resolution="verify image mode conversion and transform config",
                    cause=exc,
                )
            ) from exc
        t_transform = (_now_ms() - t0) if t0 else None
        trace.append(self._trace_event("image_transform", "ok", sample_id, index, asin, t_transform))

        # == TEXT ===============================================================
        t0 = _now_ms() if do_timing else None
        raw_text = row.get(cfg.text_column) if isinstance(row, dict) else getattr(row, cfg.text_column, None)
        raw_text_str = str(raw_text) if raw_text is not None else ""
        sanitized = sanitize_text(raw_text)
        text_fallback = (sanitized == FALLBACK_TEXT and _is_missing_text(raw_text))
        t_sanitize = (_now_ms() - t0) if t0 else None
        if text_fallback:
            fallback_reasons.append("text was empty/missing/NaN, used FALLBACK_TEXT")
            missing_modalities.append("text")
        trace.append(self._trace_event(
            "text_sanitize", "fallback_used" if text_fallback else "ok",
            sample_id, index, asin, t_sanitize,
        ))

        t0 = _now_ms() if do_timing else None
        tokenizer = self._get_tokenizer()
        try:
            tokens = tokenize_text(sanitized, tokenizer=tokenizer, max_length=cfg.text_max_length)
            validate_tokenized_output(tokens, expected_batch_size=1, max_length=cfg.text_max_length)
        except Exception as exc:
            raise RuntimeError(
                _dataset_error(
                    stage="tokenization", sample_id=sample_id,
                    row_index=index, asin=asin,
                    expected=f"tokenized dict with input_ids shape (1, {cfg.text_max_length})",
                    received=f"tokenization failed: {exc}",
                    origin="dataset.py -> data_pipeline.tokenization",
                    resolution="verify text sanitization and tokenizer config",
                    cause=exc,
                )
            ) from exc
        t_tok = (_now_ms() - t0) if t0 else None
        token_count = int((tokens["attention_mask"][0] == 1).sum())
        trace.append(self._trace_event(
            "tokenization", "ok", sample_id, index, asin, t_tok,
            details={"token_count": token_count},
        ))

        # == TABULAR ============================================================
        t0 = _now_ms() if do_timing else None
        tab_values = []
        tab_sanitized = False
        for col in cfg.tabular_columns:
            val = row[col]
            is_bad = False
            reason = ""
            if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
                is_bad = True
                reason = f"tabular '{col}' was NaN/Inf/None"
            else:
                try:
                    fval = float(val)
                    if not math.isfinite(fval):
                        is_bad = True
                        reason = f"tabular '{col}' non-finite ({val})"
                    else:
                        tab_values.append(fval)
                except (TypeError, ValueError):
                    is_bad = True
                    reason = f"tabular '{col}' not numeric ({val!r})"
            if is_bad:
                if cfg.strict_tabular:
                    raise ValueError(
                        _dataset_error(
                            stage="tabular_extract", sample_id=sample_id,
                            row_index=index, asin=asin,
                            expected=f"finite numeric value for '{col}'",
                            received=f"{val!r}",
                            resolution="fix tabular data in CSV or set strict_tabular=False",
                        )
                    )
                tab_values.append(0.0)
                tab_sanitized = True
                fallback_reasons.append(f"{reason}, set to 0.0")
        tabular_tensor = torch.tensor(tab_values, dtype=torch.float32)
        t_tab = (_now_ms() - t0) if t0 else None
        trace.append(self._trace_event(
            "tabular_extract", "sanitized" if tab_sanitized else "ok",
            sample_id, index, asin, t_tab,
        ))

        # == RATING =============================================================
        t0 = _now_ms() if do_timing else None
        raw_rating = row[cfg.target_column]
        try:
            rating_val = float(raw_rating)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                _dataset_error(
                    stage="target_extract", sample_id=sample_id,
                    row_index=index, asin=asin,
                    expected="finite numeric rating",
                    received=f"{raw_rating!r} (type={type(raw_rating).__name__})",
                    resolution="fix rating column in CSV",
                    cause=exc,
                )
            ) from exc
        if not math.isfinite(rating_val):
            raise ValueError(
                _dataset_error(
                    stage="target_extract", sample_id=sample_id,
                    row_index=index, asin=asin,
                    expected="finite numeric rating",
                    received=f"{rating_val} (NaN/Inf)",
                    resolution="fix rating column in CSV",
                )
            )
        rating_tensor = torch.tensor(rating_val, dtype=torch.float32)
        t_rating = (_now_ms() - t0) if t0 else None
        trace.append(self._trace_event("target_extract", "ok", sample_id, index, asin, t_rating))

        # == ASSEMBLE SAMPLE ====================================================
        total_ms = (_now_ms() - total_start) if total_start else None

        metadata: Dict[str, Any] = {
            "mode": cfg.mode,
            "fallback_used": len(fallback_reasons) > 0,
            "fallback_reasons": fallback_reasons,
            "missing_modalities": missing_modalities,
            "token_count": token_count,
            "text_length": len(sanitized),
            "tabular_columns": list(cfg.tabular_columns),
            "image_load_ms": t_load,
            "image_transform_ms": t_transform,
            "tokenization_ms": t_tok,
            "tabular_ms": t_tab,
            "total_sample_ms": total_ms,
            "image_exists": Path(img_path).exists(),
            "image_resolution_reason": resolve_reason,
        }
        # Multi-source metadata (lightweight strings/ints only)
        if "_source_file" in row.index if hasattr(row, 'index') else "_source_file" in row:
            metadata["source_file"] = str(row["_source_file"])
            metadata["source_index"] = int(row["_source_index"])
        if do_trace:
            metadata["trace"] = trace

        return {
            "sample_id": sample_id,
            "row_index": index,
            "asin": asin,
            "image": image_tensor,
            "image_path": img_path,
            "raw_text": raw_text_str,
            "sanitized_text": sanitized,
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "tabular": tabular_tensor,
            "rating": rating_tensor,
            "metadata": metadata,
        }

    # =========================================================================
    # __getitem__ (delegates to _build_sample)
    # =========================================================================

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self._build_sample(index)

    # =========================================================================
    # Trace inspector (Fix #7 -- no config mutation)
    # =========================================================================

    def get_sample_trace(self, index: int) -> Dict[str, Any]:
        """Build one sample with full trace, without mutating config."""
        sample = self._build_sample(index, debug_trace=True, enable_timing=True)
        return {
            "sample_id": sample["sample_id"],
            "row_index": sample["row_index"],
            "asin": sample["asin"],
            "metadata": sample["metadata"],
        }

    # =========================================================================
    # Dry-run validator
    # =========================================================================

    def dry_run_validate_sample(self, index: int = 0) -> Dict[str, Any]:
        """
        Build one sample and return a compact audit object.
        Use as pre-training trust check in Colab.
        """
        sample = self[index]
        return {
            "ok": True,
            "sample_id": sample["sample_id"],
            "asin": sample["asin"],
            "row_index": sample["row_index"],
            "shapes": {
                "image": list(sample["image"].shape),
                "input_ids": list(sample["input_ids"].shape),
                "attention_mask": list(sample["attention_mask"].shape),
                "tabular": list(sample["tabular"].shape),
                "rating": list(sample["rating"].shape),
            },
            "rating_value": sample["rating"].item(),
            "text_length": len(sample["sanitized_text"]),
            "token_count": sample["metadata"]["token_count"],
            "fallback_used": sample["metadata"]["fallback_used"],
            "fallback_reasons": sample["metadata"]["fallback_reasons"],
            "missing_modalities": sample["metadata"]["missing_modalities"],
            "image_path": sample["image_path"],
            "total_sample_ms": sample["metadata"]["total_sample_ms"],
            "trace": sample["metadata"].get("trace", []),
        }


# =============================================================================
# Factory
# =============================================================================

def build_dataset(config: Optional[DatasetConfig] = None) -> MultimodalProductDataset:
    """Factory entry point for train.py and notebooks."""
    return MultimodalProductDataset(config)


# =============================================================================
# Smoke Test
# =============================================================================

if __name__ == "__main__":
    import tempfile
    import os
    import shutil

    # NOTE: No logging.basicConfig() here -- library modules must not
    # configure global logging. Only train.py / CLI should do that.

    print("=" * 60)
    print("  data_pipeline/dataset.py -- smoke test")
    print("=" * 60)

    passed = 0
    total = 0

    def chk(label, ok):
        global passed, total
        total += 1
        if ok:
            passed += 1
        print(f"    [{'PASS' if ok else 'FAIL'}] {label}")

    # State for cleanup
    _tmp_dir = None
    _orig_get = None
    _ds_mod_ref = None

    try:
        import torch
        import pandas as pd
        import data_pipeline.dataset as _ds_mod
        _ds_mod_ref = _ds_mod

        # -- 1. DatasetConfig validation -----------------------------------
        print("\n  1. DatasetConfig...")
        cfg = DatasetConfig()
        chk("defaults OK", cfg.mode == "train" and cfg.text_max_length == 64)

        try:
            DatasetConfig(csv_filename="")
            chk("empty csv rejected", False)
        except ValueError:
            chk("empty csv rejected", True)

        try:
            DatasetConfig(mode="invalid")
            chk("invalid mode rejected", False)
        except ValueError:
            chk("invalid mode rejected", True)

        try:
            DatasetConfig(mode=123)
            chk("non-str mode rejected", False)
        except TypeError:
            chk("non-str mode rejected", True)

        try:
            DatasetConfig(text_max_length=True)
            chk("bool max_length rejected", False)
        except TypeError:
            chk("bool max_length rejected", True)

        try:
            DatasetConfig(tabular_columns=("price", ""))
            chk("empty tabular col rejected", False)
        except ValueError:
            chk("empty tabular col rejected", True)

        try:
            DatasetConfig(image_strict="yes")
            chk("non-bool strict rejected", False)
        except TypeError:
            chk("non-bool strict rejected", True)

        chk("strict_tabular default False", DatasetConfig().strict_tabular is False)
        chk("dataset_name default None", DatasetConfig().dataset_name is None)
        chk("source_files default None", DatasetConfig().source_files is None)

        try:
            DatasetConfig(dataset_name="")
            chk("empty dataset_name rejected", False)
        except ValueError:
            chk("empty dataset_name rejected", True)

        try:
            DatasetConfig(source_files="bad")
            chk("string source_files rejected", False)
        except TypeError:
            chk("string source_files rejected", True)

        try:
            DatasetConfig(source_files=[])
            chk("empty source_files rejected", False)
        except ValueError:
            chk("empty source_files rejected", True)

        chk("list source_files ok", DatasetConfig(source_files=["a.csv"]).source_files is not None)
        chk("tuple source_files ok", DatasetConfig(source_files=("a.csv",)).source_files is not None)

        # Normalization tests
        chk("csv_filename stripped", DatasetConfig(csv_filename=" sample_100.csv ").csv_filename == "sample_100.csv")
        chk("dataset_name stripped", DatasetConfig(dataset_name=" sample_100 ").dataset_name == "sample_100")

        # Generator consumption test
        gen_cfg = DatasetConfig(source_files=(x for x in ["a.csv", "b.csv"]))
        chk("generator -> tuple", isinstance(gen_cfg.source_files, tuple))
        chk("generator length 2", len(gen_cfg.source_files) == 2)
        chk("source_files stripped", DatasetConfig(source_files=[" a.csv "]).source_files == ("a.csv",))

        # Duplicate rejection
        try:
            DatasetConfig(source_files=["a.csv", "a.csv"])
            chk("duplicate source_files rejected", False)
        except ValueError:
            chk("duplicate source_files rejected", True)

        # -- 2. Build temp CSV for testing ---------------------------------
        print("\n  2. Temporary CSV dataset...")
        _tmp_dir = tempfile.mkdtemp()
        tmp_csv = os.path.join(_tmp_dir, "test_smoke.csv")
        test_df = pd.DataFrame({
            "asin": ["B001", "B002", "B003"],
            "text": ["Nice shoes", "Great hat", "Cool jacket"],
            "price": [29.99, 15.50, 89.00],
            "rating_number": [100, 50, 200],
            "category": ["shoes", "hats", "jackets"],
            "rating": [4.5, 3.0, 5.0],
            "image_path": ["", "", ""],
        })
        test_df.to_csv(tmp_csv, index=False)
        chk("temp CSV created", os.path.exists(tmp_csv))

        # -- 3. Schema validation ------------------------------------------
        print("\n  3. Schema validation...")
        bad_csv = os.path.join(_tmp_dir, "bad_schema.csv")
        pd.DataFrame({"wrong_col": [1]}).to_csv(bad_csv, index=False)

        _orig_get = _ds_mod.get_dataset_csv
        _orig_get_local = globals().get("get_dataset_csv")
        def _mock_get(fn):
            lookup = {
                "test_smoke.csv": tmp_csv,
                "bad_schema.csv": bad_csv,
            }
            if fn in lookup:
                return Path(lookup[fn])
            return _orig_get(fn)
        _ds_mod.get_dataset_csv = _mock_get
        globals()["get_dataset_csv"] = _mock_get

        try:
            MultimodalProductDataset(DatasetConfig(csv_filename="bad_schema.csv"))
            chk("missing columns rejected", False)
        except ValueError as e:
            chk("missing columns rejected", "SCHEMA ERROR" in str(e))

        # -- 3b. ASIN validity check ---------------------------------------
        print("\n  3b. ASIN validity...")
        asin_csv = os.path.join(_tmp_dir, "bad_asin.csv")
        pd.DataFrame({
            "asin": ["B001", "", "B003"],
            "text": ["a", "b", "c"],
            "price": [1, 2, 3], "rating_number": [1, 2, 3],
            "rating": [3, 4, 5],
        }).to_csv(asin_csv, index=False)
        def _mock_get2(fn):
            if fn == "bad_asin.csv":
                return Path(asin_csv)
            return _mock_get(fn)
        _ds_mod.get_dataset_csv = _mock_get2
        globals()["get_dataset_csv"] = _mock_get2

        try:
            MultimodalProductDataset(DatasetConfig(csv_filename="bad_asin.csv"))
            chk("empty ASIN rejected", False)
        except ValueError as e:
            chk("empty ASIN rejected", "invalid ASIN" in str(e) or "SCHEMA ERROR" in str(e))

        _ds_mod.get_dataset_csv = _mock_get  # restore to mock_get
        globals()["get_dataset_csv"] = _mock_get

        # -- 4. Build dataset from temp CSV --------------------------------
        print("\n  4. Dataset construction...")
        ds = MultimodalProductDataset(DatasetConfig(
            csv_filename="test_smoke.csv",
            mode="eval",
            image_strict=False,
        ))
        chk("dataset built", ds is not None)
        chk("length correct", len(ds) == 3)
        chk("subclass Dataset", isinstance(ds, _torch_data.Dataset))
        chk("tokenizer lazy", ds._tokenizer is None)

        # -- 5. Sample construction ----------------------------------------
        print("\n  5. Sample construction...")
        sample = ds[0]
        chk("sample_id present", "sample_id" in sample)
        chk("image shape", tuple(sample["image"].shape) == (3, 224, 224))
        chk("input_ids shape", sample["input_ids"].shape[0] == 1 and sample["input_ids"].shape[1] == 64)
        chk("attention_mask shape", sample["attention_mask"].shape == sample["input_ids"].shape)
        chk("tabular shape", tuple(sample["tabular"].shape) == (2,))
        chk("rating scalar", sample["rating"].ndim == 0)
        chk("rating finite", torch.isfinite(sample["rating"]))
        chk("metadata present", "metadata" in sample)
        chk("trace present", "trace" in sample["metadata"])
        chk("asin correct", sample["asin"] == "B001")
        chk("tokenizer loaded after sample", ds._tokenizer is not None)
        chk("image_exists in metadata", "image_exists" in sample["metadata"])
        chk("image_resolution_reason", "image_resolution_reason" in sample["metadata"])
        chk("image_exists is bool", isinstance(sample["metadata"]["image_exists"], bool))
        chk("resolution_reason is str", isinstance(sample["metadata"]["image_resolution_reason"], str))

        # -- 6. Index validation -------------------------------------------
        print("\n  6. Index validation...")
        try:
            ds[True]
            chk("bool index rejected", False)
        except TypeError:
            chk("bool index rejected", True)

        try:
            ds[-1]
            chk("negative index rejected", False)
        except IndexError:
            chk("negative index rejected", True)

        try:
            ds[999]
            chk("out of range rejected", False)
        except IndexError:
            chk("out of range rejected", True)

        # -- 7. Dry run validator ------------------------------------------
        print("\n  7. Dry run...")
        audit = ds.dry_run_validate_sample(0)
        chk("dry run ok", audit["ok"] is True)
        chk("dry run shapes", audit["shapes"]["image"] == [3, 224, 224])

        # -- 8. Trace inspector (no config mutation) -----------------------
        print("\n  8. Trace inspector...")
        old_dt = ds.config.debug_trace
        old_et = ds.config.enable_timing
        trace_out = ds.get_sample_trace(1)
        chk("trace has stages", len(trace_out["metadata"].get("trace", [])) > 0)
        chk("config not mutated (debug_trace)", ds.config.debug_trace == old_dt)
        chk("config not mutated (enable_timing)", ds.config.enable_timing == old_et)

        # -- 9. Trace identity context -------------------------------------
        print("\n  9. Trace identity...")
        for ev in trace_out["metadata"]["trace"]:
            if "sample_id" not in ev or "row_index" not in ev or "asin" not in ev:
                chk("trace events have identity", False)
                break
        else:
            chk("trace events have identity", True)

        # -- 10. Trace detail guard ----------------------------------------
        print("\n  10. Trace guard...")
        try:
            import numpy as np
            MultimodalProductDataset._trace_event(
                "test", "ok", "0:X", 0, "X", details={"bad": np.zeros(5)},
            )
            chk("numpy array rejected in trace", False)
        except TypeError:
            chk("numpy array rejected in trace", True)
        except ImportError:
            chk("numpy array rejected in trace (numpy unavailable, skip)", True)

        # -- 11. _is_missing_text ------------------------------------------
        print("\n  11. Text fallback detection...")
        chk("None is missing", _is_missing_text(None))
        chk("NaN is missing", _is_missing_text(float("nan")))
        chk("empty is missing", _is_missing_text(""))
        chk("whitespace is missing", _is_missing_text("   "))
        chk("real text not missing", not _is_missing_text("hello"))
        chk("literal nan not missing", not _is_missing_text("nan"))

        # -- 12. Pickle safety (multiprocessing) ---------------------------
        print("\n  12. Pickle safety...")
        import pickle
        from torch.utils.data import Subset, DataLoader, TensorDataset
        try:
            pickle.dumps(ds)
            chk("dataset picklable", True)
        except Exception as e:
            chk("dataset picklable", False)
        try:
            pickle.dumps(Subset(ds, [0]))
            chk("subset picklable", True)
        except Exception as e:
            chk("subset picklable", False)

        # -- 13. DataLoader worker test ------------------------------------
        print("\n  13. DataLoader worker retrieval...")
        try:
            _tiny_dl = DataLoader(
                Subset(ds, list(range(min(2, len(ds))))),
                batch_size=1, num_workers=0,
            )
            _b = next(iter(_tiny_dl))
            chk("loader yields batch", isinstance(_b, dict))
        except Exception as e:
            chk("loader yields batch", False)

        # -- Summary -------------------------------------------------------
        print(f"\n{'='*60}")
        status = "PASS" if passed == total else "FAIL"
        print(f"  [{status}]  {passed}/{total} checks passed")
        print("=" * 60)
        if passed < total:
            sys.exit(1)

    except Exception as e:
        print(f"[FAIL] SMOKE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        # Restore monkeypatch even on failure
        if _orig_get is not None and _ds_mod_ref is not None:
            try:
                _ds_mod_ref.get_dataset_csv = _orig_get
            except Exception:
                pass
        if _orig_get is not None:
            try:
                globals()["get_dataset_csv"] = _orig_get
            except Exception:
                pass
        # Clean temp files even on failure
        if _tmp_dir and os.path.isdir(_tmp_dir):
            try:
                shutil.rmtree(_tmp_dir)
            except Exception:
                pass


