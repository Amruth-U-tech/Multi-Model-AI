# =============================================================================
# data_pipeline/tokenization.py
# Centralized Text Orchestration Layer -- Multimodal AI Pipeline
# =============================================================================
#
# Purpose:
#   THE SINGLE TOKENIZATION AUTHORITY for the entire multimodal system.
#   All text preprocessing, sanitization, and tokenization flows through
#   this module -- models never own tokenizer state.
#
# Ownership:
#   raw text cleaning    -> sanitize_text()
#   tokenizer loading    -> load_tokenizer()
#   single-text tokens   -> tokenize_text()
#   batch tokenization   -> tokenize_batch()
#   output validation    -> validate_tokenized_output()
#
# What this file does NOT own:
#   - Transformer forward pass     -> text_encoder.py
#   - Embedding projection         -> text_encoder.py
#   - Modality fusion              -> fusion.py
#   - Dataset loading / CSV I/O    -> dataset.py (future)
#   - Batching / collation         -> collate.py (future)
#   - Training orchestration       -> train.py (future)
#
# Design constraints:
#   - Zero torch import (tokenization is CPU string processing)
#   - No global tokenizer loaded at import time (multiprocessing safe)
#   - All validation fails early with explicit error messages
#   - Fully idempotent -- safe to re-import in notebooks
#   - Colab-friendly with cell-style section separators
#
# Usage:
#   from data_pipeline.tokenization import sanitize_text, load_tokenizer
#   from data_pipeline.tokenization import tokenize_batch
#
# Compatible with:
#   - Google Colab + mounted Drive
#   - Local Windows / Linux / macOS execution
#   - torch.utils.data.DataLoader multiprocessing workers
#   - Jupyter / IPython notebooks
#   - CI/CD pipelines
# =============================================================================


# %%
# =============================================================================
# CELL 1 -- Imports (Minimal, Import-Light)
# =============================================================================
#
# IMPORTANT: No torch import here. Tokenization is CPU string processing.
# torch is imported ONLY inside validate_tokenized_output() where tensor
# checks are needed. This keeps the module lightweight and avoids pulling
# heavy CUDA initialization into DataLoader worker processes.

import sys
import math
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# %%
# =============================================================================
# CELL 2 -- Path Infrastructure (from configs.paths)
# =============================================================================
#
# Uses the centralized path authority for HuggingFace cache routing.
# Falls back gracefully if configs.paths is not importable (e.g., when
# running tokenization.py in complete isolation for unit testing).

try:
    from configs.paths import CACHE_DIR, PROJECT_ROOT
    _HF_CACHE_DIR: Optional[Path] = CACHE_DIR / "huggingface"
    _HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
except ImportError:
    _HF_CACHE_DIR = None
    logger.debug(
        "configs.paths not available -- HuggingFace cache will use default location. "
        "This is expected when running tokenization.py in isolation."
    )


# %%
# =============================================================================
# CELL 3 -- Constants
# =============================================================================

# -- Default backbone model ----------------------------------------------------
# MiniLM-L6-v2: lightweight, high-quality sentence embeddings.
# Change here to swap tokenizer backbone project-wide.
DEFAULT_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

# -- Default tokenization geometry ---------------------------------------------
DEFAULT_MAX_LENGTH: int = 64    # Tuned for Amazon product titles (10-25 tokens avg)

# -- Fallback text for missing/invalid inputs ----------------------------------
# Used when sanitize_text() receives None, NaN, empty, or whitespace-only input.
# Must be a non-empty string that tokenizers can process without error.
FALLBACK_TEXT: str = "[NO_TEXT_AVAILABLE]"


# %%
# =============================================================================
# CELL 4 -- Text Sanitization
# =============================================================================

def sanitize_text(text: Union[str, float, int, None]) -> str:
    """
    Converts any input to a clean, tokenizer-safe string.

    Handles real-world data pipeline edge cases:
      - None values                        (missing CSV cells)
      - float NaN (pandas missing-value)   (missing CSV cells)
      - Non-string types (int, float)      (numeric columns read as text)
      - Empty strings after stripping      (blank descriptions)
      - Whitespace-only strings            (formatting artifacts)

    Returns FALLBACK_TEXT for anything that reduces to empty -- never returns
    an empty string, which causes tokenizers to produce degenerate embeddings.

    Args:
        text : Raw value from a dataset row. Any type accepted.

    Returns:
        Non-empty string guaranteed safe for tokenization.

    Examples:
        >>> sanitize_text(None)
        '[NO_TEXT_AVAILABLE]'
        >>> sanitize_text(float('nan'))
        '[NO_TEXT_AVAILABLE]'
        >>> sanitize_text(42)
        '42'
        >>> sanitize_text("  product title  ")
        'product title'
    """
    # None check
    if text is None:
        return FALLBACK_TEXT

    # float NaN check: NaN != NaN is the only reliable identity test
    try:
        if isinstance(text, float) and math.isnan(text):
            return FALLBACK_TEXT
    except (TypeError, ValueError):
        pass

    # Convert to string and strip whitespace
    cleaned = str(text).strip()
    return cleaned if cleaned else FALLBACK_TEXT


# %%
# =============================================================================
# CELL 5 -- Tokenizer Loading
# =============================================================================

def load_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
    cache_dir: Optional[Path] = None,
):
    """
    Loads and returns the HuggingFace AutoTokenizer for the given model.

    This is the SINGLE tokenizer loading authority. All code that needs a
    tokenizer should call this function -- never AutoTokenizer.from_pretrained()
    directly. This ensures:
      - Consistent cache routing across Colab and local
      - Centralized error handling for missing/corrupted models
      - One place to update when swapping tokenizer backbone

    Tokenizer loading is LAZY -- not called at module import time. This is
    critical for DataLoader worker safety: each worker that needs a tokenizer
    calls load_tokenizer() once in Dataset.__init__(), not at import.

    Args:
        model_name : HuggingFace model identifier.
                     Default: sentence-transformers/all-MiniLM-L6-v2
        cache_dir  : Optional directory for HuggingFace cache.
                     Default: CACHE_DIR/huggingface from configs.paths,
                     or HuggingFace default if configs.paths is unavailable.

    Returns:
        AutoTokenizer instance, ready for tokenization.

    Raises:
        RuntimeError : If the tokenizer cannot be loaded (network failure,
                       invalid model name, corrupted cache).
    """
    from transformers import AutoTokenizer

    resolved_cache = cache_dir or _HF_CACHE_DIR
    cache_info = str(resolved_cache) if resolved_cache else "HuggingFace default"

    logger.info(f"Loading tokenizer: '{model_name}' | cache={cache_info}")

    try:
        tok = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(resolved_cache) if resolved_cache else None,
        )
    except Exception as e:
        raise RuntimeError(
            f"TOKENIZER LOAD FAILED.\n"
            f"  Model    : {model_name}\n"
            f"  Cache dir: {cache_info}\n"
            f"  Error    : {e}\n"
            f"\n"
            f"  Resolution:\n"
            f"    - Check internet connection (first download requires network)\n"
            f"    - Verify model name is correct on huggingface.co\n"
            f"    - Clear cache: delete {cache_info} and retry\n"
            f"    - If offline, pre-download: AutoTokenizer.from_pretrained('{model_name}')"
        ) from e

    logger.info(f"Tokenizer loaded | vocab_size={tok.vocab_size}")
    return tok


# %%
# =============================================================================
# CELL 6 -- Single-Text Tokenization
# =============================================================================

def tokenize_text(
    text: str,
    tokenizer=None,
    max_length: int = DEFAULT_MAX_LENGTH,
    return_tensors: str = "pt",
) -> Dict[str, Any]:
    """
    Tokenizes a SINGLE sanitized text string.

    Intended for use inside Dataset.__getitem__() where each worker processes
    one sample at a time. DataLoader's collate_fn stacks the results.

    Args:
        text           : A single sanitized string (run through sanitize_text() first).
        tokenizer      : Pre-loaded AutoTokenizer. If None, loads default lazily.
        max_length     : Maximum token sequence length. Must be > 0.
        return_tensors : Tensor format ("pt" for PyTorch).

    Returns:
        Dict with "input_ids" and "attention_mask" -- both shaped (1, max_length).

    Raises:
        ValueError : If max_length <= 0 or text is empty.
        TypeError  : If text is not a string.
    """
    # -- Input validation ------------------------------------------------------
    if not isinstance(text, str):
        raise TypeError(
            f"tokenize_text() expected str, got {type(text).__name__}. "
            f"Run sanitize_text() first to convert raw data."
        )
    if not text.strip():
        raise ValueError(
            "tokenize_text() received empty/whitespace string. "
            "Run sanitize_text() first -- it guarantees non-empty output."
        )
    if max_length <= 0:
        raise ValueError(
            f"max_length must be > 0, got {max_length}. "
            f"Typical values: 32 (short), 64 (medium), 128 (long)."
        )

    # -- Lazy tokenizer load ---------------------------------------------------
    if tokenizer is None:
        tokenizer = load_tokenizer()

    encoded = tokenizer(
        text,
        padding            = "max_length",
        truncation         = True,
        max_length         = max_length,
        return_tensors     = return_tensors,
        return_attention_mask = True,
    )

    result = dict(encoded)
    validate_tokenized_output(result, expected_batch_size=1, max_length=max_length)
    return result


# %%
# =============================================================================
# CELL 7 -- Batch Tokenization
# =============================================================================

def tokenize_batch(
    texts: List[str],
    tokenizer=None,
    max_length: int = DEFAULT_MAX_LENGTH,
    return_tensors: str = "pt",
) -> Dict[str, Any]:
    """
    Tokenizes a LIST of sanitized text strings into padded, truncated tensors.

    Primary use cases:
      - Inference: tokenize a batch of product texts before encoder forward()
      - Smoke tests: verify tokenization output shape and contract
      - Notebooks: quick batch experiments

    For training, prefer tokenize_text() inside Dataset.__getitem__() so
    DataLoader workers can parallelize tokenization across CPU cores.

    Args:
        texts          : List of sanitized strings. Must be non-empty.
                         Each element must be a non-empty string.
        tokenizer      : Pre-loaded AutoTokenizer. If None, loads default lazily.
        max_length     : Maximum token sequence length. Must be > 0.
        return_tensors : Tensor format ("pt" for PyTorch).

    Returns:
        Dict with "input_ids" and "attention_mask" -- both (B, max_length)
        where B = len(texts).

    Raises:
        ValueError : If texts list is empty or max_length <= 0.
        TypeError  : If any element in texts is not a string.
    """
    # -- Input validation ------------------------------------------------------
    if not texts:
        raise ValueError(
            "tokenize_batch() received an empty text list. "
            "Provide at least one sanitized string."
        )
    if max_length <= 0:
        raise ValueError(
            f"max_length must be > 0, got {max_length}. "
            f"Typical values: 32 (short), 64 (medium), 128 (long)."
        )

    # -- Type check each element -----------------------------------------------
    for i, t in enumerate(texts):
        if not isinstance(t, str):
            raise TypeError(
                f"tokenize_batch() element [{i}] is {type(t).__name__}, expected str. "
                f"Run sanitize_text() on all elements first."
            )

    # -- Lazy tokenizer load ---------------------------------------------------
    if tokenizer is None:
        tokenizer = load_tokenizer()

    # -- Tokenize (fail explicitly, no silent fallback) -------------------------
    try:
        encoded = tokenizer(
            texts,
            padding            = "max_length",
            truncation         = True,
            max_length         = max_length,
            return_tensors     = return_tensors,
            return_attention_mask = True,
        )
    except Exception as exc:
        tok_name = type(tokenizer).__name__ if tokenizer else "None"
        raise RuntimeError(
            f"TOKENIZER EXECUTION FAILED.\n"
            f"  Batch size      : {len(texts)}\n"
            f"  max_length      : {max_length}\n"
            f"  Tokenizer class : {tok_name}\n"
            f"  Original error  : {exc}\n"
            f"\n"
            f"  Resolution:\n"
            f"    - Verify all texts are sanitized (sanitize_text() first)\n"
            f"    - Check tokenizer cache integrity\n"
            f"    - Reload tokenizer with load_tokenizer()"
        ) from exc

    result = dict(encoded)
    validate_tokenized_output(result, expected_batch_size=len(texts), max_length=max_length)
    return result


# %%
# =============================================================================
# CELL 8 -- Output Validation
# =============================================================================

def validate_tokenized_output(
    encoded: Dict[str, Any],
    expected_batch_size: Optional[int] = None,
    max_length: Optional[int] = None,
) -> None:
    """
    Validates the output of tokenize_text() or tokenize_batch().

    Call this after tokenization to catch contract violations early --
    before malformed tensors propagate silently into the encoder.

    Checks:
      - "input_ids" and "attention_mask" keys exist
      - Both are 2D tensors (batch, seq_len)
      - Both have matching shapes
      - Both have integer dtype (not float)
      - No empty tensors (batch_size > 0, seq_len > 0)
      - If expected_batch_size provided, verifies batch dimension
      - If max_length provided, verifies sequence dimension

    Args:
        encoded             : Dict from tokenize_text() or tokenize_batch().
        expected_batch_size : If set, verifies dim 0 matches this value.
        max_length          : If set, verifies dim 1 matches this value.

    Raises:
        ValueError  : If any validation check fails.
        KeyError    : If required keys are missing.
    """
    import torch

    # -- Key existence ---------------------------------------------------------
    for key in ("input_ids", "attention_mask"):
        if key not in encoded:
            raise KeyError(
                f"Tokenized output missing '{key}'. "
                f"Got keys: {list(encoded.keys())}. "
                f"Verify tokenizer is producing correct output format."
            )

    ids = encoded["input_ids"]
    mask = encoded["attention_mask"]

    # -- Tensor type check -----------------------------------------------------
    if not isinstance(ids, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise ValueError(
            f"Expected torch.Tensor, got input_ids={type(ids).__name__}, "
            f"attention_mask={type(mask).__name__}. "
            f"Use return_tensors='pt' in tokenization."
        )

    # -- Rank check (must be 2D) -----------------------------------------------
    if ids.ndim != 2:
        raise ValueError(
            f"input_ids must be 2D (batch, seq_len), got {ids.ndim}D shape {list(ids.shape)}."
        )
    if mask.ndim != 2:
        raise ValueError(
            f"attention_mask must be 2D (batch, seq_len), got {mask.ndim}D shape {list(mask.shape)}."
        )

    # -- Shape consistency -----------------------------------------------------
    if ids.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch: input_ids={list(ids.shape)}, "
            f"attention_mask={list(mask.shape)}. Must be identical."
        )

    # -- Non-empty check -------------------------------------------------------
    if ids.shape[0] == 0:
        raise ValueError("Tokenized output has batch_size=0 (empty batch).")
    if ids.shape[1] == 0:
        raise ValueError("Tokenized output has seq_len=0 (empty sequences).")

    # -- Dtype check (both must be integer, not float) --------------------------
    if ids.dtype.is_floating_point:
        raise ValueError(
            f"input_ids has float dtype ({ids.dtype}), expected integer "
            f"(torch.int64 or torch.int32). Do not cast -- fix tokenizer output."
        )
    if mask.dtype.is_floating_point:
        raise ValueError(
            f"attention_mask has float dtype ({mask.dtype}), expected integer "
            f"(torch.int64 or torch.int32). Do not cast -- fix tokenizer output."
        )

    # -- Batch size check ------------------------------------------------------
    if expected_batch_size is not None and ids.shape[0] != expected_batch_size:
        raise ValueError(
            f"Batch size mismatch: expected {expected_batch_size}, "
            f"got {ids.shape[0]}."
        )

    # -- Max length check ------------------------------------------------------
    if max_length is not None and ids.shape[1] != max_length:
        raise ValueError(
            f"Sequence length mismatch: expected {max_length}, "
            f"got {ids.shape[1]}. Check max_length parameter."
        )

    logger.debug(
        f"Tokenized output valid | shape={list(ids.shape)} | dtype={ids.dtype}"
    )


# %%
# =============================================================================
# CELL 9 -- Smoke Test (python data_pipeline/tokenization.py)
# =============================================================================

if __name__ == "__main__":

    logging.basicConfig(
        level   = logging.DEBUG,
        format  = "[%(asctime)s] [%(levelname)s] %(name)s -- %(message)s",
        datefmt = "%H:%M:%S",
    )

    print("=" * 60)
    print("  data_pipeline/tokenization.py -- smoke test")
    print("=" * 60)

    try:
        # -- sanitize_text edge cases ------------------------------------------
        print("\n  Testing sanitize_text()...")
        assert sanitize_text(None)          == FALLBACK_TEXT, "None failed"
        assert sanitize_text("")            == FALLBACK_TEXT, "empty failed"
        assert sanitize_text("   ")         == FALLBACK_TEXT, "whitespace failed"
        assert sanitize_text(float("nan"))  == FALLBACK_TEXT, "NaN failed"
        assert sanitize_text(42)            == "42",          "int failed"
        assert sanitize_text(3.14)          == "3.14",        "float failed"
        assert sanitize_text("hello")       == "hello",       "normal failed"
        assert sanitize_text("  spaced  ")  == "spaced",      "strip failed"
        print("    -> All sanitize_text() checks PASS")

        # -- load_tokenizer ----------------------------------------------------
        print("\n  Testing load_tokenizer()...")
        tok = load_tokenizer()
        print(f"    -> Tokenizer loaded | vocab_size={tok.vocab_size}")

        # -- tokenize_text (single) --------------------------------------------
        print("\n  Testing tokenize_text()...")
        single = tokenize_text("Test product title", tokenizer=tok, max_length=64)
        assert "input_ids" in single
        assert "attention_mask" in single
        print(f"    -> input_ids shape: {list(single['input_ids'].shape)}")

        # -- tokenize_batch (multiple) -----------------------------------------
        print("\n  Testing tokenize_batch()...")
        batch_texts = [
            sanitize_text("Spanx Core In-Power Line Super High Shaping Sheers"),
            sanitize_text("KingSize Men's Big & Tall Lightweight Jersey Cargo"),
            sanitize_text("Boho Tassel Earrings for Women Girls Multicolor"),
            sanitize_text(""),  # becomes FALLBACK_TEXT
        ]
        batch = tokenize_batch(batch_texts, tokenizer=tok, max_length=64)
        print(f"    -> input_ids shape    : {list(batch['input_ids'].shape)}")
        print(f"    -> attention_mask shape: {list(batch['attention_mask'].shape)}")

        # -- validate_tokenized_output -----------------------------------------
        print("\n  Testing validate_tokenized_output()...")
        validate_tokenized_output(batch, expected_batch_size=4, max_length=64)
        print("    -> Validation PASS")

        # -- Error guards: empty batch -----------------------------------------
        print("\n  Testing error guards...")
        try:
            tokenize_batch([], tokenizer=tok)
            print("    -> ERROR: empty batch should raise ValueError")
        except ValueError:
            print("    -> Empty batch guard PASS")

        # -- Error guards: invalid max_length ----------------------------------
        try:
            tokenize_batch(["test"], tokenizer=tok, max_length=0)
            print("    -> ERROR: max_length=0 should raise ValueError")
        except ValueError:
            print("    -> Invalid max_length guard PASS")

        # -- Error guards: non-string in batch ---------------------------------
        try:
            tokenize_batch([123], tokenizer=tok)
            print("    -> ERROR: non-string should raise TypeError")
        except TypeError:
            print("    -> Type guard PASS")

        # -- Error guards: invalid tokenizer name ------------------------------
        print("  Testing invalid tokenizer name...")
        try:
            load_tokenizer("nonexistent/model-that-does-not-exist-xyz")
            print("    -> ERROR: invalid model should raise RuntimeError")
        except RuntimeError:
            print("    -> Invalid model guard PASS")

        # -- Idempotency -------------------------------------------------------
        print("\n  Testing idempotency (reload tokenizer)...")
        tok2 = load_tokenizer()
        batch2 = tokenize_batch(batch_texts, tokenizer=tok2, max_length=64)
        import torch
        assert torch.equal(batch["input_ids"], batch2["input_ids"]), "Idempotency failed"
        print("    -> Idempotent reload PASS")

        print("\n" + "=" * 60)
        print("  [PASS]  Smoke test PASSED -- tokenization.py is production-grade.")
        print("=" * 60)

    except Exception as e:
        logger.exception(f"[FAIL] SMOKE TEST FAILED: {e}")
        sys.exit(1)
