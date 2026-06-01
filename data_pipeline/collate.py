# =============================================================================
# data_pipeline/collate.py
# Multimodal Batch Orchestration Authority — Multimodal AI Pipeline
# =============================================================================
#
# Ownership (this file ONLY):
#   - Validates incoming sample contract from dataset.py
#   - Preserves sample identity and ordering
#   - Detects duplicate sample IDs, row indices, ASINs
#   - Converts sample-major data into modality-major tensors
#   - Stacks images, text, tabular, ratings into batch tensors
#   - Validates shape/dtype/finite consistency per modality
#   - Builds lightweight batch manifest, fingerprint, transfer readiness
#   - Aggregates fallback and trace metadata (bounded)
#   - Returns CPU-only, pin-ready, contiguous batch for DataLoader
#
# What this file does NOT own:
#   +-----------------------------+---------------------------+
#   | Responsibility              | Correct File              |
#   +-----------------------------+---------------------------+
#   | Image loading/transforms    | data_pipeline/transforms  |
#   | Text tokenization           | data_pipeline/tokenization|
#   | Sample construction         | data_pipeline/dataset     |
#   | GPU transfer / .cuda()      | train.py                  |
#   | Pin memory                  | DataLoader (pin_memory=T) |
#   | Model forward pass          | models/*                  |
#   | Loss / optimizer / metrics  | train.py                  |
#   | Batch scheduling/prefetch   | DataLoader                |
#   +-----------------------------+---------------------------+
#
# =============================================================================

from __future__ import annotations

import sys
import time
import math
import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Sequence

import torch

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Constants
# =============================================================================

REQUIRED_SAMPLE_KEYS = frozenset({
    "sample_id", "row_index", "asin",
    "image", "image_path",
    "raw_text", "sanitized_text",
    "input_ids", "attention_mask",
    "tabular", "rating",
    "metadata",
})

_LIGHTWEIGHT_TYPES = (str, int, float, bool, type(None))


# =============================================================================
# 2. Timing helper
# =============================================================================

def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _shape_str(tensor: torch.Tensor) -> str:
    """Flatten tensor shape to a lightweight string like '3x224x224'."""
    return "x".join(str(d) for d in tensor.shape)


# =============================================================================
# 3. Error helper
# =============================================================================

def _collate_error(
    stage: str,
    message: str,
    batch_pos: Optional[int] = None,
    sample_id: str = "unknown",
    expected: str = "",
    received: str = "",
    resolution: str = "",
    cause: Optional[Exception] = None,
) -> str:
    parts = [
        f"[COLLATE ERROR]",
        f"  Stage     : {stage}",
    ]
    if batch_pos is not None:
        parts.append(f"  Batch pos : {batch_pos}")
    parts.append(f"  Sample ID : {sample_id}")
    if message:
        parts.append(f"  Message   : {message}")
    if expected:
        parts.append(f"  Expected  : {expected}")
    if received:
        parts.append(f"  Received  : {received}")
    if resolution:
        parts.append(f"  Resolution: {resolution}")
    if cause:
        parts.append(f"  Cause     : {type(cause).__name__}: {cause}")
    return "\n".join(parts)


# =============================================================================
# 4. CollateConfig
# =============================================================================

@dataclass
class CollateConfig:
    """Configuration for BatchCollator."""

    strict_validation: bool = True
    include_metadata: bool = True
    include_trace: bool = True
    include_manifest: bool = True
    include_fingerprint: bool = True
    include_transfer_readiness: bool = True
    enable_timing: bool = True
    validate_contiguous: bool = True
    validate_dtypes: bool = True
    validate_finite: bool = True
    lightweight_mode: bool = False
    max_metadata_items: int = 64
    max_metadata_depth: int = 4
    max_trace_events: int = 128
    expected_image_channels: int = 3
    expected_image_size: Tuple[int, int] = (224, 224)

    def __post_init__(self):
        # Bool fields
        _BOOL_FIELDS = (
            "strict_validation", "include_metadata", "include_trace",
            "include_manifest", "include_fingerprint",
            "include_transfer_readiness", "enable_timing",
            "validate_contiguous", "validate_dtypes", "validate_finite",
            "lightweight_mode",
        )
        for name in _BOOL_FIELDS:
            val = getattr(self, name)
            if not isinstance(val, bool):
                raise TypeError(
                    f"CollateConfig.{name} must be bool, "
                    f"got {type(val).__name__}: {val!r}"
                )
        # Positive-int fields (reject bool subclass)
        for name in ("max_metadata_items", "max_metadata_depth", "max_trace_events", "expected_image_channels"):
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, int) or val <= 0:
                raise TypeError(
                    f"CollateConfig.{name} must be positive int, "
                    f"got {type(val).__name__}: {val!r}"
                )
        # expected_image_size
        s = self.expected_image_size
        if (not isinstance(s, tuple) or len(s) != 2
                or not all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in s)):
            raise TypeError(
                f"CollateConfig.expected_image_size must be tuple of 2 positive ints, "
                f"got {s!r}"
            )


# =============================================================================
# 5. Lightweight metadata guard
# =============================================================================

def _validate_lightweight_metadata(
    obj: Any, context: str, max_items: int,
    max_depth: int = 4, _depth: int = 0,
) -> None:
    """Reject heavy or deeply nested objects in metadata."""
    if _depth > max_depth:
        raise TypeError(
            f"[COLLATE METADATA ERROR] {context} exceeds max depth {max_depth}. "
            f"Metadata must remain shallow and lightweight."
        )
    if obj is None:
        return
    if isinstance(obj, _LIGHTWEIGHT_TYPES):
        return
    if isinstance(obj, dict):
        if len(obj) > max_items:
            raise TypeError(
                f"[COLLATE METADATA ERROR] {context}: dict has {len(obj)} items "
                f"(max {max_items}). Reduce metadata size."
            )
        for k, v in obj.items():
            _validate_lightweight_metadata(
                v, f"{context}.{k}", max_items, max_depth, _depth + 1
            )
        return
    if isinstance(obj, (list, tuple)):
        if len(obj) > max_items:
            raise TypeError(
                f"[COLLATE METADATA ERROR] {context}: sequence has {len(obj)} items "
                f"(max {max_items}). Reduce metadata size."
            )
        for i, v in enumerate(obj[:max_items]):
            _validate_lightweight_metadata(
                v, f"{context}[{i}]", max_items, max_depth, _depth + 1
            )
        return
    # Reject tensors, arrays, PIL images, etc.
    type_name = type(obj).__name__
    raise TypeError(
        f"[COLLATE METADATA ERROR] {context}: contains non-lightweight type "
        f"'{type_name}'. Only scalar/str/list/dict allowed in batch metadata."
    )


# =============================================================================
# 6. Sample validation helpers
# =============================================================================

def _validate_batch_input(samples: Any) -> None:
    """Validate that incoming batch is a non-empty sequence of dicts."""
    if samples is None:
        raise ValueError(_collate_error(
            "batch_input", "Batch is None.",
            expected="non-empty sequence of sample dicts",
            received="None",
            resolution="verify DataLoader is providing samples",
        ))
    if not isinstance(samples, (list, tuple)):
        raise TypeError(_collate_error(
            "batch_input", "Batch is not a sequence.",
            expected="list or tuple of sample dicts",
            received=type(samples).__name__,
            resolution="verify DataLoader collate_fn usage",
        ))
    if len(samples) == 0:
        raise ValueError(_collate_error(
            "batch_input", "Batch is empty.",
            expected="at least 1 sample",
            received="0 samples",
            resolution="verify DataLoader batch_size > 0",
        ))
    for i, s in enumerate(samples):
        if not isinstance(s, dict):
            raise TypeError(_collate_error(
                "batch_input", f"Item at batch position {i} is not a dict.",
                batch_pos=i,
                expected="dict (sample from dataset.py)",
                received=type(s).__name__,
            ))


def _validate_sample_structure(
    sample: Dict[str, Any], batch_pos: int, cfg: CollateConfig,
) -> None:
    """Validate required keys, field types, and incoming metadata safety."""
    sid = sample.get("sample_id", "unknown")

    # -- Required keys (always enforced, no toggle) --
    missing = REQUIRED_SAMPLE_KEYS - set(sample.keys())
    if missing:
        raise ValueError(_collate_error(
            "sample_structure", f"Missing required keys: {sorted(missing)}",
            batch_pos=batch_pos, sample_id=sid,
            expected=f"keys {sorted(REQUIRED_SAMPLE_KEYS)}",
            received=f"keys {sorted(sample.keys())}",
            resolution="verify dataset.py sample contract",
        ))

    # -- Lightweight string field validation --
    for fname in ("image_path", "raw_text"):
        val = sample.get(fname)
        if not isinstance(val, str):
            raise TypeError(_collate_error(
                "sample_field_validation",
                f"{fname} must be str, got {type(val).__name__}.",
                batch_pos=batch_pos, sample_id=sid,
                expected="str", received=repr(val),
                resolution="verify dataset.py sample contract",
            ))
    # image_path must be non-empty (dataset.py owns fallback path generation)
    if not sample["image_path"].strip():
        raise ValueError(_collate_error(
            "sample_field_validation",
            "image_path must be non-empty str.",
            batch_pos=batch_pos, sample_id=sid,
            expected="non-empty str",
            received=repr(sample["image_path"]),
            resolution="verify dataset.py image path routing",
        ))
    # sanitized_text must be non-empty str
    st = sample.get("sanitized_text")
    if not isinstance(st, str) or not st.strip():
        raise ValueError(_collate_error(
            "sample_field_validation",
            "sanitized_text must be non-empty str.",
            batch_pos=batch_pos, sample_id=sid,
            expected="non-empty str",
            received=repr(st),
            resolution="verify dataset.py text sanitization contract",
        ))

    # -- Metadata must always be dict (not None, not other types) --
    meta = sample.get("metadata")
    if not isinstance(meta, dict):
        raise TypeError(_collate_error(
            "sample_metadata",
            f"sample metadata must be dict, got {type(meta).__name__}.",
            batch_pos=batch_pos, sample_id=sid,
            expected="dict metadata from dataset.py",
            received=type(meta).__name__,
            resolution="verify dataset.py sample contract",
        ))
    _validate_lightweight_metadata(
        meta, f"sample[{batch_pos}].metadata",
        cfg.max_metadata_items, cfg.max_metadata_depth,
    )


def _validate_sample_identity(
    sample: Dict[str, Any], batch_pos: int
) -> Tuple[str, int, str]:
    """Validate and extract identity fields. Returns (sample_id, row_index, asin)."""
    sid = sample.get("sample_id", "")
    if not isinstance(sid, str) or not sid.strip():
        raise ValueError(_collate_error(
            "sample_identity", "sample_id is empty or not a string.",
            batch_pos=batch_pos, sample_id=repr(sid),
            expected="non-empty str", received=repr(sid),
        ))

    ridx = sample.get("row_index")
    if isinstance(ridx, bool) or not isinstance(ridx, int):
        raise TypeError(_collate_error(
            "sample_identity", f"row_index must be int, got {type(ridx).__name__}.",
            batch_pos=batch_pos, sample_id=sid,
        ))

    asin = sample.get("asin", "")
    if not isinstance(asin, str) or not asin.strip():
        raise ValueError(_collate_error(
            "sample_identity", "asin is empty or not a string.",
            batch_pos=batch_pos, sample_id=sid,
            expected="non-empty str", received=repr(asin),
        ))
    return sid, ridx, asin


def _check_duplicate_identities(
    ids: List[str], indices: List[int], asins: List[str]
) -> None:
    """Fail on duplicate sample_id, row_index, or ASIN within one batch."""
    # sample_id
    seen = {}
    for i, sid in enumerate(ids):
        if sid in seen:
            raise ValueError(_collate_error(
                "identity_uniqueness",
                f"Duplicate sample_id '{sid}' at batch positions {seen[sid]} and {i}.",
                batch_pos=i, sample_id=sid,
                resolution="verify DataLoader sampler does not duplicate indices",
            ))
        seen[sid] = i

    # row_index
    seen_idx: Dict[int, int] = {}
    for i, ridx in enumerate(indices):
        if ridx in seen_idx:
            raise ValueError(_collate_error(
                "identity_uniqueness",
                f"Duplicate row_index {ridx} at batch positions {seen_idx[ridx]} and {i}.",
                batch_pos=i, sample_id=ids[i],
            ))
        seen_idx[ridx] = i

    # asin
    seen_asin: Dict[str, int] = {}
    for i, a in enumerate(asins):
        if a in seen_asin:
            raise ValueError(_collate_error(
                "identity_uniqueness",
                f"Duplicate ASIN '{a}' at batch positions {seen_asin[a]} and {i}.",
                batch_pos=i, sample_id=ids[i],
            ))
        seen_asin[a] = i


# =============================================================================
# 7. Tensor validation helpers
# =============================================================================

def _validate_tensor_contract(
    name: str, tensor: Any, batch_pos: int, sample_id: str,
    expected_rank: Optional[int] = None,
    expected_dtype_family: Optional[str] = None,
    validate_finite: bool = False,
    validate_cpu: bool = True,
) -> None:
    """Validate a single tensor from a sample."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(_collate_error(
            f"{name}_validation",
            f"'{name}' is not a tensor.",
            batch_pos=batch_pos, sample_id=sample_id,
            expected="torch.Tensor", received=type(tensor).__name__,
        ))
    if validate_cpu and tensor.is_cuda:
        raise ValueError(_collate_error(
            f"{name}_validation",
            f"'{name}' is on GPU. collate.py requires CPU tensors.",
            batch_pos=batch_pos, sample_id=sample_id,
            expected="CPU tensor", received=f"device={tensor.device}",
            resolution="dataset.py must return CPU-only tensors",
        ))
    if expected_rank is not None and tensor.ndim != expected_rank:
        raise ValueError(_collate_error(
            f"{name}_validation",
            f"'{name}' has wrong rank.",
            batch_pos=batch_pos, sample_id=sample_id,
            expected=f"rank {expected_rank}", received=f"rank {tensor.ndim}, shape {list(tensor.shape)}",
        ))
    if expected_dtype_family == "float" and not tensor.dtype.is_floating_point:
        raise TypeError(_collate_error(
            f"{name}_validation",
            f"'{name}' dtype is not floating-point.",
            batch_pos=batch_pos, sample_id=sample_id,
            expected="floating-point dtype", received=str(tensor.dtype),
        ))
    if expected_dtype_family == "integer_non_bool":
        if tensor.dtype == torch.bool or tensor.dtype.is_floating_point:
            raise TypeError(_collate_error(
                f"{name}_validation",
                f"'{name}' dtype must be integer (non-bool).",
                batch_pos=batch_pos, sample_id=sample_id,
                expected="integer dtype (not bool)",
                received=str(tensor.dtype),
            ))
    elif expected_dtype_family == "integer_or_bool":
        if tensor.dtype.is_floating_point:
            raise TypeError(_collate_error(
                f"{name}_validation",
                f"'{name}' dtype must be integer or bool.",
                batch_pos=batch_pos, sample_id=sample_id,
                expected="integer or bool dtype",
                received=str(tensor.dtype),
            ))
    elif expected_dtype_family == "integer":
        if tensor.dtype.is_floating_point:
            raise TypeError(_collate_error(
                f"{name}_validation",
                f"'{name}' dtype should be integer-like.",
                batch_pos=batch_pos, sample_id=sample_id,
                expected="integer dtype", received=str(tensor.dtype),
            ))
    if validate_finite and tensor.dtype.is_floating_point:
        if not torch.isfinite(tensor).all():
            raise ValueError(_collate_error(
                f"{name}_validation",
                f"'{name}' contains NaN or Inf.",
                batch_pos=batch_pos, sample_id=sample_id,
                expected="all finite values",
                received=f"has {(~torch.isfinite(tensor)).sum().item()} non-finite values",
                resolution="verify dataset.py sample integrity",
            ))


def _normalize_text_tensor(
    tensor: torch.Tensor, name: str, batch_pos: int, sample_id: str
) -> torch.Tensor:
    """Normalize text tensor from [1,L] or [L] to [L]. Rejects other ranks."""
    if tensor.ndim == 1:
        return tensor
    if tensor.ndim == 2 and tensor.shape[0] == 1:
        return tensor.squeeze(0)
    raise ValueError(_collate_error(
        "text_normalize",
        f"'{name}' has unsupported shape for text tensor.",
        batch_pos=batch_pos, sample_id=sample_id,
        expected="shape [L] or [1, L]",
        received=f"shape {list(tensor.shape)}",
        resolution="verify dataset.py token output contract",
    ))


# =============================================================================
# 8. Stacking helpers
# =============================================================================

def _stack_images(
    samples: Sequence[Dict[str, Any]],
    cfg: CollateConfig,
) -> torch.Tensor:
    """Stack image tensors into [B, C, H, W]."""
    C = cfg.expected_image_channels
    H, W = cfg.expected_image_size
    tensors = []
    for i, s in enumerate(samples):
        sid = s.get("sample_id", "unknown")
        img = s["image"]
        _validate_tensor_contract(
            "image", img, i, sid,
            expected_rank=3,
            expected_dtype_family="float",
            validate_finite=cfg.validate_finite,
        )
        if cfg.strict_validation:
            if img.shape[0] != C:
                raise ValueError(_collate_error(
                    "image_stack",
                    f"Channel mismatch.",
                    batch_pos=i, sample_id=sid,
                    expected=f"{C} channels", received=f"{img.shape[0]} channels",
                ))
            if img.shape[1] != H or img.shape[2] != W:
                raise ValueError(_collate_error(
                    "image_stack",
                    f"Spatial size mismatch.",
                    batch_pos=i, sample_id=sid,
                    expected=f"({H}, {W})", received=f"({img.shape[1]}, {img.shape[2]})",
                ))
        tensors.append(img)
    stacked = torch.stack(tensors, dim=0)
    if cfg.validate_contiguous and not stacked.is_contiguous():
        stacked = stacked.contiguous()
    return stacked


def _stack_text(
    samples: Sequence[Dict[str, Any]],
    cfg: CollateConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack text tensors into [B, L]. Returns (input_ids, attention_mask)."""
    ids_list = []
    mask_list = []
    expected_len = None

    for i, s in enumerate(samples):
        sid = s.get("sample_id", "unknown")
        raw_ids = s["input_ids"]
        raw_mask = s["attention_mask"]

        _validate_tensor_contract("input_ids", raw_ids, i, sid)
        _validate_tensor_contract("attention_mask", raw_mask, i, sid)

        ids = _normalize_text_tensor(raw_ids, "input_ids", i, sid)
        mask = _normalize_text_tensor(raw_mask, "attention_mask", i, sid)

        if ids.shape != mask.shape:
            raise ValueError(_collate_error(
                "text_stack",
                "input_ids and attention_mask shape mismatch within sample.",
                batch_pos=i, sample_id=sid,
                expected="matching shapes",
                received=f"input_ids={list(ids.shape)}, mask={list(mask.shape)}",
            ))

        L = ids.shape[0]
        if expected_len is None:
            expected_len = L
        elif L != expected_len:
            raise ValueError(_collate_error(
                "text_stack",
                "Token length mismatch across samples.",
                batch_pos=i, sample_id=sid,
                expected=f"length {expected_len}",
                received=f"length {L}",
                resolution="all samples must have same max_length from dataset.py",
            ))

        if cfg.validate_dtypes:
            _validate_tensor_contract(
                "input_ids", ids, i, sid,
                expected_rank=1, expected_dtype_family="integer_non_bool",
            )
            _validate_tensor_contract(
                "attention_mask", mask, i, sid,
                expected_rank=1, expected_dtype_family="integer_or_bool",
            )

        # -- Text value validation --
        if cfg.strict_validation:
            # input_ids: no negative token IDs
            if ids.dtype != torch.bool and (ids < 0).any():
                raise ValueError(_collate_error(
                    "text_stack",
                    "input_ids contains negative token IDs.",
                    batch_pos=i, sample_id=sid,
                    expected="all token IDs >= 0",
                    received=f"min token id = {ids.min().item()}",
                    resolution="verify tokenization.py output contract",
                ))
            # attention_mask: must be binary {0, 1}
            if mask.dtype == torch.bool:
                # Bool masks are valid; will be cast to long before stacking
                pass
            else:
                unique_vals = mask.unique()
                if not all(v in (0, 1) for v in unique_vals.tolist()):
                    raise ValueError(_collate_error(
                        "text_stack",
                        "attention_mask contains non-binary values.",
                        batch_pos=i, sample_id=sid,
                        expected="only values {0, 1}",
                        received=f"unique values = {unique_vals.tolist()}",
                        resolution="verify tokenization.py output contract",
                    ))

        # Normalize bool mask to long for consistent stacking
        if mask.dtype == torch.bool:
            mask = mask.to(torch.long)

        # Current contract expects fixed token length from tokenization.py
        # because padding="max_length" is used. Future dynamic batching
        # should add an explicit adaptive-padding mode here, not silently.

        ids_list.append(ids)
        mask_list.append(mask)

    stacked_ids = torch.stack(ids_list, dim=0)
    stacked_mask = torch.stack(mask_list, dim=0)
    if cfg.validate_contiguous:
        if not stacked_ids.is_contiguous():
            stacked_ids = stacked_ids.contiguous()
        if not stacked_mask.is_contiguous():
            stacked_mask = stacked_mask.contiguous()
    return stacked_ids, stacked_mask


def _stack_tabular(
    samples: Sequence[Dict[str, Any]],
    cfg: CollateConfig,
) -> torch.Tensor:
    """Stack tabular tensors into [B, F]."""
    tensors = []
    expected_width = None

    for i, s in enumerate(samples):
        sid = s.get("sample_id", "unknown")
        tab = s["tabular"]
        _validate_tensor_contract(
            "tabular", tab, i, sid,
            expected_rank=1,
            expected_dtype_family="float",
            validate_finite=cfg.validate_finite,
        )
        F = tab.shape[0]
        if expected_width is None:
            expected_width = F
        elif F != expected_width:
            raise ValueError(_collate_error(
                "tabular_stack",
                "Tabular width mismatch across samples.",
                batch_pos=i, sample_id=sid,
                expected=f"width {expected_width}",
                received=f"width {F}",
            ))
        tensors.append(tab)

    stacked = torch.stack(tensors, dim=0)
    if cfg.validate_contiguous and not stacked.is_contiguous():
        stacked = stacked.contiguous()
    return stacked


def _stack_ratings(
    samples: Sequence[Dict[str, Any]],
    cfg: CollateConfig,
) -> torch.Tensor:
    """Stack scalar ratings into [B]."""
    values = []
    for i, s in enumerate(samples):
        sid = s.get("sample_id", "unknown")
        r = s["rating"]
        if isinstance(r, torch.Tensor):
            if r.is_cuda:
                raise ValueError(_collate_error(
                    "rating_validation", "Rating tensor is on GPU.",
                    batch_pos=i, sample_id=sid,
                ))
            if r.ndim == 0:
                val = r.item()
            elif r.ndim == 1 and r.shape[0] == 1:
                val = r.item()
            else:
                raise ValueError(_collate_error(
                    "rating_validation",
                    "Rating tensor has invalid shape.",
                    batch_pos=i, sample_id=sid,
                    expected="scalar or shape [1]",
                    received=f"shape {list(r.shape)}",
                ))
        elif isinstance(r, (int, float)):
            val = float(r)
        else:
            raise TypeError(_collate_error(
                "rating_validation",
                f"Rating has unexpected type.",
                batch_pos=i, sample_id=sid,
                expected="scalar tensor or numeric",
                received=type(r).__name__,
            ))
        if not math.isfinite(val):
            raise ValueError(_collate_error(
                "rating_validation",
                "Rating is NaN or Inf.",
                batch_pos=i, sample_id=sid,
                expected="finite value", received=str(val),
            ))
        values.append(val)
    ratings = torch.tensor(values, dtype=torch.float32)
    if cfg.validate_contiguous and not ratings.is_contiguous():
        ratings = ratings.contiguous()
    return ratings


# =============================================================================
# 9. Metadata builders
# =============================================================================

def _build_fingerprint(sample_ids: List[str]) -> str:
    """Deterministic batch fingerprint from ordered sample IDs.

    Uses SHA-256 for cross-process, cross-machine stability.
    Do not use Python hash(); it is salted per process and not reproducible.
    """
    joined = "||".join(sample_ids)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def _build_transfer_readiness(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Compute real transfer-readiness checks for train.py."""
    _TENSOR_KEYS = ("images", "input_ids", "attention_mask", "tabular", "ratings")
    cpu_only = True
    contiguous = True
    dtype_ok = True
    shape_ok = True
    finite_ok = True
    B = None

    for key in _TENSOR_KEYS:
        t = batch.get(key)
        if not isinstance(t, torch.Tensor):
            shape_ok = False
            continue
        if t.is_cuda:
            cpu_only = False
        if not t.is_contiguous():
            contiguous = False
        # batch dim consistency
        if B is None:
            B = t.shape[0]
        elif t.shape[0] != B:
            shape_ok = False
        # dtype contract
        if key in ("images", "tabular", "ratings"):
            if not t.dtype.is_floating_point:
                dtype_ok = False
        elif key in ("input_ids",):
            if t.dtype.is_floating_point or t.dtype == torch.bool:
                dtype_ok = False
        elif key in ("attention_mask",):
            if t.dtype.is_floating_point:
                dtype_ok = False
        # finite check on float tensors
        if t.dtype.is_floating_point:
            if not torch.isfinite(t).all():
                finite_ok = False

    # input_ids / attention_mask shape match
    ids_t = batch.get("input_ids")
    mask_t = batch.get("attention_mask")
    if isinstance(ids_t, torch.Tensor) and isinstance(mask_t, torch.Tensor):
        if ids_t.shape != mask_t.shape:
            shape_ok = False

    return {
        "cpu_only": cpu_only,
        "pin_ready": cpu_only,
        "contiguous": contiguous,
        "dtype_contract_ok": dtype_ok,
        "shape_contract_ok": shape_ok,
        "finite_check_ok": finite_ok,
        "non_blocking_ready": cpu_only and contiguous,
    }


def _build_manifest(
    batch: Dict[str, Any],
    sample_ids: List[str],
    row_indices: List[int],
    asins: List[str],
    fallback_count: int,
    missing_mod_count: int,
    collate_ms: Optional[float],
) -> Dict[str, Any]:
    """Build lightweight batch manifest."""
    shapes = {}
    dtypes = {}
    for key in ("images", "input_ids", "attention_mask", "tabular", "ratings"):
        t = batch.get(key)
        if isinstance(t, torch.Tensor):
            shapes[key] = _shape_str(t)
            dtypes[key] = str(t.dtype)

    B = len(sample_ids)
    ids_tensor = batch.get("input_ids")
    max_token_length = ids_tensor.shape[-1] if isinstance(ids_tensor, torch.Tensor) and ids_tensor.ndim >= 1 else 0

    return {
        "batch_size": B,
        "sample_ids": sample_ids,
        "row_indices": row_indices,
        "asins": asins,
        "modalities": ["image", "text", "tabular", "rating"],
        "shapes": shapes,
        "dtypes": dtypes,
        "fallback_count": fallback_count,
        "missing_modality_count": missing_mod_count,
        "max_token_length": max_token_length,
        "collate_ms": collate_ms,
    }


def _aggregate_fallback(
    samples: Sequence[Dict[str, Any]], max_items: int
) -> Dict[str, Any]:
    """Aggregate fallback metadata across samples (bounded)."""
    fallback_count = 0
    fallback_ids: List[str] = []
    missing_image = 0
    missing_text = 0
    missing_tabular = 0
    reasons: List[str] = []

    for s in samples:
        meta = s.get("metadata")
        if not isinstance(meta, dict):
            continue
        if meta.get("fallback_used"):
            fallback_count += 1
            fallback_ids.append(s.get("sample_id", "unknown"))
        for mod in meta.get("missing_modalities", []):
            if mod == "image":
                missing_image += 1
            elif mod == "text":
                missing_text += 1
            elif mod == "tabular":
                missing_tabular += 1
        for r in meta.get("fallback_reasons", []):
            if len(reasons) < max_items:
                reasons.append(r)

    return {
        "fallback_count": fallback_count,
        "fallback_sample_ids": fallback_ids[:max_items],
        "missing_modalities": {
            "image": missing_image,
            "text": missing_text,
            "tabular": missing_tabular,
        },
        "fallback_reasons": reasons,
    }


def _aggregate_trace(
    samples: Sequence[Dict[str, Any]], max_events: int
) -> Dict[str, Any]:
    """Build compact trace summary with a global event budget."""
    total_traces = 0
    stages_seen: set = set()
    error_count = 0
    fallback_count = 0
    events_scanned = 0

    for s in samples:
        if events_scanned >= max_events:
            break
        meta = s.get("metadata")
        if not isinstance(meta, dict):
            continue
        trace = meta.get("trace")
        if not isinstance(trace, list):
            continue
        total_traces += 1
        for ev in trace:
            if events_scanned >= max_events:
                break
            if isinstance(ev, dict):
                stages_seen.add(ev.get("stage", "unknown"))
                status = ev.get("status", "")
                if status == "error":
                    error_count += 1
                elif status == "fallback_used":
                    fallback_count += 1
            events_scanned += 1

    return {
        "sample_trace_count": total_traces,
        "events_scanned": events_scanned,
        "max_events": max_events,
        "truncated": events_scanned >= max_events,
        "stages_seen": sorted(stages_seen),
        "error_status_count": error_count,
        "fallback_status_count": fallback_count,
    }


# =============================================================================
# 10. BatchCollator
# =============================================================================

class BatchCollator:
    """
    Multimodal batch orchestration authority.

    Receives synchronized samples from dataset.py, validates identity
    and tensor contracts, stacks modalities into batch tensors, and
    returns CPU-only, pin-ready, contiguous batches for DataLoader.

    Usage::

        collator = BatchCollator()
        loader = DataLoader(dataset, collate_fn=collator, ...)

    Or via factory::

        collate_fn = build_collate_fn()
    """

    def __init__(self, config: Optional[CollateConfig] = None) -> None:
        if config is None:
            config = CollateConfig()
        self.config = config
        logger.info(
            f"BatchCollator ready | strict={config.strict_validation} "
            f"| lightweight={config.lightweight_mode}"
        )

    # -----------------------------------------------------------------
    # Collate trace helper
    # -----------------------------------------------------------------

    @staticmethod
    def _trace_event(
        stage: str, status: str,
        duration_ms: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ev: Dict[str, Any] = {"stage": stage, "status": status}
        if duration_ms is not None:
            ev["duration_ms"] = duration_ms
        if details:
            ev["details"] = details
        return ev

    # -----------------------------------------------------------------
    # __call__  (DataLoader invokes this)
    # -----------------------------------------------------------------

    def __call__(self, samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        cfg = self.config
        collate_trace: List[Dict[str, Any]] = []
        total_start = _now_ms() if cfg.enable_timing else None

        # -- 1. Validate batch input --------------------------------------
        t0 = _now_ms() if cfg.enable_timing else None
        _validate_batch_input(samples)
        t_val = (_now_ms() - t0) if t0 else None
        collate_trace.append(self._trace_event("batch_input_validation", "ok", t_val))

        # -- 2. Validate structure + identity per sample ------------------
        t0 = _now_ms() if cfg.enable_timing else None
        sample_ids: List[str] = []
        row_indices: List[int] = []
        asins: List[str] = []
        image_paths: List[str] = []

        for i, s in enumerate(samples):
            _validate_sample_structure(s, i, cfg)
            sid, ridx, asin = _validate_sample_identity(s, i)
            sample_ids.append(sid)
            row_indices.append(ridx)
            asins.append(asin)
            image_paths.append(s.get("image_path", ""))

        if cfg.strict_validation:
            _check_duplicate_identities(sample_ids, row_indices, asins)

        t_id = (_now_ms() - t0) if t0 else None
        collate_trace.append(self._trace_event(
            "identity_validation", "ok", t_id,
            details={"batch_size": len(samples)},
        ))

        # -- 3. Stack images ----------------------------------------------
        t0 = _now_ms() if cfg.enable_timing else None
        images = _stack_images(samples, cfg)
        t_img = (_now_ms() - t0) if t0 else None
        collate_trace.append(self._trace_event(
            "image_stack", "ok", t_img,
            details={"shape": _shape_str(images)},
        ))

        # -- 4. Stack text ------------------------------------------------
        t0 = _now_ms() if cfg.enable_timing else None
        input_ids, attention_mask = _stack_text(samples, cfg)
        t_txt = (_now_ms() - t0) if t0 else None
        collate_trace.append(self._trace_event(
            "text_stack", "ok", t_txt,
            details={"shape": _shape_str(input_ids)},
        ))

        # -- 5. Stack tabular ---------------------------------------------
        t0 = _now_ms() if cfg.enable_timing else None
        tabular = _stack_tabular(samples, cfg)
        t_tab = (_now_ms() - t0) if t0 else None
        collate_trace.append(self._trace_event(
            "tabular_stack", "ok", t_tab,
            details={"shape": _shape_str(tabular)},
        ))

        # -- 6. Stack ratings ---------------------------------------------
        t0 = _now_ms() if cfg.enable_timing else None
        ratings = _stack_ratings(samples, cfg)
        t_rat = (_now_ms() - t0) if t0 else None
        collate_trace.append(self._trace_event(
            "target_stack", "ok", t_rat,
            details={"shape": _shape_str(ratings)},
        ))

        # -- 7. Validate final batch contract -----------------------------
        t0 = _now_ms() if cfg.enable_timing else None
        B = len(samples)
        _BATCH_TENSORS = {
            "images": images, "input_ids": input_ids,
            "attention_mask": attention_mask,
            "tabular": tabular, "ratings": ratings,
        }
        for tname, t in _BATCH_TENSORS.items():
            if t.shape[0] != B:
                raise RuntimeError(_collate_error(
                    "batch_contract_validation",
                    f"{tname} batch dimension mismatch.",
                    expected=f"B={B}",
                    received=f"{tname}.shape={list(t.shape)}",
                    resolution=f"check _stack_{tname} implementation",
                ))
        if input_ids.shape != attention_mask.shape:
            raise RuntimeError(_collate_error(
                "batch_contract_validation",
                "input_ids and attention_mask shape mismatch.",
                expected=f"matching shapes",
                received=f"input_ids={list(input_ids.shape)}, mask={list(attention_mask.shape)}",
            ))
        t_contract = (_now_ms() - t0) if t0 else None
        collate_trace.append(self._trace_event(
            "batch_contract_validation", "ok", t_contract,
        ))

        # -- 8. Assemble batch dict ---------------------------------------
        batch: Dict[str, Any] = {
            "images": images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "tabular": tabular,
            "ratings": ratings,
            "sample_ids": sample_ids,
            "row_indices": row_indices,
            "asins": asins,
            "image_paths": image_paths,
        }

        # -- 9. Build metadata --------------------------------------------
        if cfg.include_metadata:
            total_ms = (_now_ms() - total_start) if total_start else None

            metadata: Dict[str, Any] = {}

            if cfg.include_fingerprint:
                metadata["batch_fingerprint"] = _build_fingerprint(sample_ids)

            if cfg.include_transfer_readiness:
                metadata["transfer_readiness"] = _build_transfer_readiness(batch)

            if not cfg.lightweight_mode:
                fallback_summary = _aggregate_fallback(samples, cfg.max_metadata_items)

                if cfg.include_manifest:
                    metadata["batch_manifest"] = _build_manifest(
                        batch, sample_ids, row_indices, asins,
                        fallback_count=fallback_summary["fallback_count"],
                        missing_mod_count=sum(
                            fallback_summary["missing_modalities"].values()
                        ),
                        collate_ms=total_ms,
                    )

                metadata["fallback_summary"] = fallback_summary

                if cfg.include_trace:
                    metadata["trace_summary"] = _aggregate_trace(
                        samples, cfg.max_trace_events
                    )
                    metadata["collate_trace"] = collate_trace
            else:
                # Lightweight: compact manifest only
                if cfg.include_manifest:
                    metadata["batch_manifest"] = {
                        "batch_size": B,
                        "collate_ms": total_ms,
                    }

            # Validate metadata is lightweight
            _validate_lightweight_metadata(
                metadata, "batch.metadata",
                cfg.max_metadata_items, cfg.max_metadata_depth,
            )
            batch["metadata"] = metadata

        return batch


# =============================================================================
# 11. Factory
# =============================================================================

def build_collate_fn(config: Optional[CollateConfig] = None) -> BatchCollator:
    """Factory entry point for train.py and notebooks."""
    return BatchCollator(config)


# =============================================================================
# 12. Smoke Tests
# =============================================================================

if __name__ == "__main__":
    import copy
    import re

    print("=" * 60)
    print("  data_pipeline/collate.py -- smoke test")
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

    def _fs(idx, asin, rating=4.0, tlen=64, tw=2, ish=(3, 224, 224), fb=False):
        """Fake sample matching dataset.py contract."""
        return {
            "sample_id": f"{idx}:{asin}", "row_index": idx, "asin": asin,
            "image": torch.randn(*ish), "image_path": f"/fake/{asin}.jpg",
            "raw_text": f"t {asin}", "sanitized_text": f"t {asin}",
            "input_ids": torch.randint(0, 1000, (1, tlen), dtype=torch.long),
            "attention_mask": torch.ones(1, tlen, dtype=torch.long),
            "tabular": torch.randn(tw), "rating": torch.tensor(rating),
            "metadata": {
                "fallback_used": fb,
                "fallback_reasons": ["fb"] if fb else [],
                "missing_modalities": ["image"] if fb else [],
                "trace": [{"stage": "load", "status": "ok"}],
            },
        }

    try:
        # 1. Config
        print("\n  1. Config...")
        chk("defaults", CollateConfig().strict_validation is True)
        chk("depth default", CollateConfig().max_metadata_depth == 4)
        try:
            CollateConfig(strict_validation="y"); chk("bool", False)
        except TypeError:
            chk("bool", True)
        try:
            CollateConfig(max_metadata_depth=True); chk("depth bool", False)
        except TypeError:
            chk("depth bool", True)

        # 2. Good batch
        print("\n  2. Good batch...")
        c = BatchCollator()
        sa = [_fs(0, "B1", 4.5), _fs(1, "B2", 3.0, fb=True), _fs(2, "B3", 5.0)]
        b = c(sa)
        chk("img", list(b["images"].shape) == [3, 3, 224, 224])
        chk("ids", list(b["input_ids"].shape) == [3, 64])
        chk("mask", list(b["attention_mask"].shape) == [3, 64])
        chk("tab", list(b["tabular"].shape) == [3, 2])
        chk("rat", list(b["ratings"].shape) == [3])
        chk("order", b["sample_ids"] == ["0:B1", "1:B2", "2:B3"])
        tr = b["metadata"]["transfer_readiness"]
        chk("tr cpu", tr["cpu_only"] is True)
        chk("tr dtype", tr["dtype_contract_ok"] is True)
        chk("tr shape", tr["shape_contract_ok"] is True)
        chk("tr finite", tr["finite_check_ok"] is True)
        chk("fb count", b["metadata"]["fallback_summary"]["fallback_count"] == 1)
        chk("contig", all(b[k].is_contiguous() for k in
            ("images", "input_ids", "attention_mask", "tabular", "ratings")))

        # 3. Missing key
        print("\n  3. Missing key...")
        s = _fs(0, "X1"); del s["image"]
        try: c([s]); chk("rej", False)
        except ValueError: chk("rej", True)

        # 4. Shape mismatch
        print("\n  4. Shape mismatch...")
        try: c([_fs(0, "A1"), _fs(1, "A2", tlen=32)]); chk("token", False)
        except ValueError: chk("token", True)
        try: c([_fs(0, "C1"), _fs(1, "C2", ish=(3,128,128))]); chk("img", False)
        except ValueError: chk("img", True)

        # 5. Duplicates
        print("\n  5. Duplicates...")
        try: c([_fs(0, "E1"), _fs(0, "E2")]); chk("idx", False)
        except ValueError: chk("idx", True)
        try: c([_fs(0, "F1"), _fs(1, "F1")]); chk("asin", False)
        except ValueError: chk("asin", True)

        # 6. Edge inputs
        print("\n  6. Edge inputs...")
        try: c([]); chk("empty", False)
        except ValueError: chk("empty", True)
        try: c(None); chk("None", False)
        except (ValueError, TypeError): chk("None", True)

        # 7. B=1
        print("\n  7. B=1...")
        chk("ok", list(c([_fs(0, "H1")])["images"].shape) == [1, 3, 224, 224])

        # 8. [L] shape
        print("\n  8. [L] shape...")
        fl = _fs(0, "I1")
        fl["input_ids"] = fl["input_ids"].squeeze(0)
        fl["attention_mask"] = fl["attention_mask"].squeeze(0)
        chk("ok", list(c([fl])["input_ids"].shape) == [1, 64])

        # 9. Fingerprint
        print("\n  9. Fingerprint...")
        chk("stable", _build_fingerprint(["a","b"]) == _build_fingerprint(["a","b"]))
        chk("order", _build_fingerprint(["a","b"]) != _build_fingerprint(["b","a"]))

        # 10. NaN/Inf
        print("\n  10. NaN/Inf...")
        s = _fs(0, "J1"); s["image"][0,0,0] = float("nan")
        try: c([s]); chk("nan img", False)
        except ValueError: chk("nan img", True)
        s = _fs(0, "K1"); s["tabular"][0] = float("inf")
        try: c([s]); chk("inf tab", False)
        except ValueError: chk("inf tab", True)

        # 11. Text value validation
        print("\n  11. Text values...")
        s = _fs(0, "P1"); s["input_ids"] = torch.tensor([[-1]+[0]*63], dtype=torch.long)
        try: c([s]); chk("neg ids", False)
        except ValueError: chk("neg ids", True)
        s = _fs(0, "Q1"); s["input_ids"] = torch.ones(1, 64, dtype=torch.bool)
        try: c([s]); chk("bool ids", False)
        except TypeError: chk("bool ids", True)
        s = _fs(0, "R1"); s["input_ids"] = torch.ones(1, 64, dtype=torch.float32)
        try: c([s]); chk("float ids", False)
        except TypeError: chk("float ids", True)
        s = _fs(0, "S1"); s["attention_mask"] = torch.tensor([[0,1,2]+[1]*61], dtype=torch.long)
        try: c([s]); chk("non-binary mask", False)
        except ValueError: chk("non-binary mask", True)

        # 12. Metadata safety
        print("\n  12. Metadata safety...")
        s = _fs(0, "T1"); s["metadata"]["bad"] = torch.zeros(5)
        try: c([s]); chk("tensor rej", False)
        except TypeError: chk("tensor rej", True)
        dc = CollateConfig(max_metadata_depth=2)
        s = _fs(0, "U1"); s["metadata"]["a"] = {"b": {"c": {"d": "x"}}}
        try: BatchCollator(dc)([s]); chk("deep rej", False)
        except TypeError: chk("deep rej", True)

        # 13. Immutability
        print("\n  13. Immutability...")
        s = _fs(0, "V1")
        ok = set(s.keys())
        osh = tuple(s["input_ids"].shape)
        om = copy.deepcopy(s["metadata"])
        c([s])
        chk("keys", set(s.keys()) == ok)
        chk("shape", tuple(s["input_ids"].shape) == osh)
        chk("meta", s["metadata"] == om)

        # 14. Source safety
        print("\n  14. Source safety...")
        src = open(__file__, encoding="utf-8").read()
        prod = src.split('if __name__')[0]
        chk("no assert", len(re.findall(r'^\s*assert\s', prod, re.MULTILINE)) == 0)
        chk("no Path", "from pathlib import Path" not in prod)
        chk("no models", "from models" not in prod)
        chk("no fail_on_missing", "fail_on_missing_keys" not in prod)

        # 15. metadata=None fails
        print("\n  15. metadata=None...")
        s = _fs(0, "MN1"); s["metadata"] = None
        try: c([s]); chk("rejected", False)
        except TypeError: chk("rejected", True)

        # 16. Missing keys always fail (no toggle)
        print("\n  16. Missing keys always fail...")
        chk("no config field", "fail_on_missing_keys" not in CollateConfig.__dataclass_fields__)
        s = _fs(0, "MK1"); del s["image"]
        try: c([s]); chk("always fails", False)
        except ValueError: chk("always fails", True)

        # 17. Field type validation
        print("\n  17. Field types...")
        s = _fs(0, "FT1"); s["image_path"] = 123
        try: c([s]); chk("bad image_path", False)
        except TypeError: chk("bad image_path", True)
        s = _fs(0, "FT2"); s["image_path"] = ""
        try: c([s]); chk("empty image_path", False)
        except ValueError: chk("empty image_path", True)
        s = _fs(0, "FT3"); s["raw_text"] = 42
        try: c([s]); chk("bad raw_text", False)
        except TypeError: chk("bad raw_text", True)
        s = _fs(0, "FT4"); s["sanitized_text"] = ""
        try: c([s]); chk("empty sanitized", False)
        except ValueError: chk("empty sanitized", True)
        s = _fs(0, "FT5"); s["sanitized_text"] = 999
        try: c([s]); chk("bad sanitized type", False)
        except ValueError: chk("bad sanitized type", True)

        # 18. Global trace budget
        print("\n  18. Trace budget...")
        tc = CollateConfig(max_trace_events=2)
        sa = []
        for i in range(5):
            fs = _fs(i, f"TR{i}")
            fs["metadata"]["trace"] = [
                {"stage": f"s{i}a", "status": "ok"},
                {"stage": f"s{i}b", "status": "ok"},
            ]
            sa.append(fs)
        tb = BatchCollator(tc)(sa)
        ts = tb["metadata"]["trace_summary"]
        chk("scanned<=2", ts["events_scanned"] <= 2)
        chk("max_events", ts["max_events"] == 2)
        chk("truncated", ts["truncated"] is True)

        # 19. Error formatting consistency
        print("\n  19. Error formatting...")
        chk("_collate_error used", "COLLATE ERROR" in prod)

        # Summary
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

