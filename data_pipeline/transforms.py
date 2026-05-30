# =============================================================================
# data_pipeline/transforms.py
# Centralized Image Preprocessing Authority -- Multimodal AI Pipeline
# =============================================================================
#
# Responsibilities (this file ONLY):
#   - Image loading with full edge-case tolerance
#   - PIL mode conversion (L, RGBA, P, LA, CMYK -> RGB)
#   - Train augmentation pipeline (stochastic, online)
#   - Eval preprocessing pipeline (deterministic)
#   - Pre-transform PIL validation
#   - Post-transform tensor validation (NaN, Inf, dtype, shape)
#
# Responsibilities that live ELSEWHERE (do NOT add here):
#   +-----------------------------+---------------------------+
#   | Responsibility              | Correct File              |
#   +-----------------------------+---------------------------+
#   | text tokenization           | data_pipeline/tokenization|
#   | batch collation             | data_pipeline/collate.py  |
#   | GPU device transfer         | train.py / inference.py   |
#   | backbone feature extraction | models/image_encoder.py   |
#   | modality dropout / fusion   | models/fusion.py          |
#   | DataLoader orchestration    | dataset.py / train.py     |
#   | async prefetch / queues     | train.py                  |
#   +-----------------------------+---------------------------+
#
# Design Philosophy:
#   STATELESS   -- no internal caches, queues, counters, or schedulers
#   PURE        -- f(image) -> tensor, no side effects
#   DEVICE-AGNOSTIC -- no .cuda(), no GPU init at import time
#   WORKER-SAFE -- zero global mutable state, safe for num_workers > 0
#   QUEUE-SAFE  -- no batch ordering assumptions or prefetch coupling
#   IMPORT-LIGHT -- no torch at module level, no model dependencies
#
# Execution Flow:
#   image_path -> safe_load_image() -> validate_image_input()
#     -> build_*_transforms() -> validate_tensor_output()
#       -> collate.py -> GPU transfer -> image_encoder.forward()
#
# Compatible with:
#   - torch.utils.data.DataLoader (num_workers, pin_memory, prefetch)
#   - CUDA / CPU execution
#   - FP16 mixed precision
#   - Tesla T4 / Colab
#   - Future Kornia / torchvision.v2 GPU transforms
#   - Future async prefetch queues
# =============================================================================


# %%
# =============================================================================
# CELL 1 -- Imports (Minimal, No torch at Module Level)
# =============================================================================

import os
import sys
import math
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple, Union

# ---------------------------------------------------------------
# Project Routing -- configs.paths is the ONLY path authority
# ---------------------------------------------------------------
# Bootstrap: derive project root from this file's location so that
# `from configs.paths import PROJECT_ROOT` succeeds even when running
# this file directly (python data_pipeline/transforms.py).
# configs.paths then takes over as the canonical routing authority.
_THIS_DIR = Path(__file__).resolve().parent          # data_pipeline/
_PROJECT_DIR = _THIS_DIR.parent                      # multi-model-ai/
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

try:
    from configs.paths import PROJECT_ROOT  # noqa: F401
except ImportError as _routing_err:
    raise RuntimeError(
        "ROUTING FAILURE: Cannot import configs.paths.PROJECT_ROOT. "
        "Ensure configs/paths.py exists and the project root is on sys.path. "
        f"Current sys.path: {sys.path[:5]}..."
    ) from _routing_err

# PIL -- required at module level for image loading
# PIL is lightweight and multiprocessing-safe
from PIL import Image, ImageFile

# Allow PIL to load truncated images safely rather than crashing
# workers mid-pipeline. The integrity check in safe_load_image()
# catches truly corrupted files before they propagate.
ImageFile.LOAD_TRUNCATED_IMAGES = True


# %%
# =============================================================================
# CELL 2 -- Logging (Module-Scoped, No basicConfig)
# =============================================================================

logger = logging.getLogger(__name__)


# %%
# =============================================================================
# CELL 3 -- Constants (Preprocessing Geometry + Normalization)
# =============================================================================
# These constants are the SINGLE SOURCE OF TRUTH for all image
# preprocessing geometry across the entire project. No other file
# should define IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE, or RESIZE_SIZE.

# -- ImageNet normalization -- must match ConvNeXt pretraining -----------------
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD:  Tuple[float, float, float] = (0.229, 0.224, 0.225)

# -- Preprocessing geometry ----------------------------------------------------
# Resize short edge to RESIZE_SIZE then CenterCrop to INPUT_SIZE.
# Preserves product aspect ratio; avoids geometric distortion.
INPUT_SIZE:  Tuple[int, int] = (224, 224)
RESIZE_SIZE: int             = 256

# -- Fallback image color ------------------------------------------------------
# After ImageNet normalization, (127,127,127) maps to approx (-0.02,-0.07,-0.14)
# which is near the dataset mean. White (255,255,255) normalizes to approx
# (2.6, 2.4, 2.2), far outside the natural distribution, polluting retrieval
# indices and latent clustering. Gray is the mathematically stable fallback.
FALLBACK_COLOR: Tuple[int, int, int] = (127, 127, 127)


# %%
# =============================================================================
# CELL 4 -- TransformConfig (Structured Configuration)
# =============================================================================

@dataclass
class TransformConfig:
    """
    Single source of truth for all image preprocessing hyperparameters.

    Follows the identical dataclass pattern as TextEncoderConfig and
    FusionConfig for consistent multimodal orchestration.

    Size convention:
        input_size uses (H, W) ordering -- matches torchvision CenterCrop.
        PIL internally uses (W, H) -- safe_load_image handles this.
        For square sizes (224, 224) the ordering is irrelevant, but
        __post_init__ validates explicitly so non-square sizes never
        silently flip dimensions.

    Usage:
        config = TransformConfig(input_size=(336, 336), resize_size=384)
        tfm    = build_train_transforms(config)
    """
    # -- Spatial geometry (H, W) for torchvision --------------------------------
    input_size:  Tuple[int, int] = INPUT_SIZE
    resize_size: int             = RESIZE_SIZE

    # -- Normalization contract ------------------------------------------------
    # Must match the backbone's pretraining normalization.
    # ConvNeXt / ResNet / ViT all use ImageNet stats.
    mean: Tuple[float, float, float] = IMAGENET_MEAN
    std:  Tuple[float, float, float] = IMAGENET_STD

    def __post_init__(self) -> None:
        """Validate config before any transform is built."""
        # -- input_size --------------------------------------------------------
        if not isinstance(self.input_size, tuple) or len(self.input_size) != 2:
            raise TypeError(
                f"input_size must be a tuple of 2 ints, got {self.input_size!r}"
            )
        for i, v in enumerate(self.input_size):
            if not isinstance(v, int) or v <= 0:
                raise ValueError(
                    f"input_size[{i}] must be a positive int, got {v!r}"
                )

        # -- resize_size -------------------------------------------------------
        if not isinstance(self.resize_size, int) or self.resize_size <= 0:
            raise ValueError(
                f"resize_size must be a positive int, got {self.resize_size!r}"
            )
        if self.resize_size < max(self.input_size):
            raise ValueError(
                f"resize_size ({self.resize_size}) < max(input_size) "
                f"({max(self.input_size)}). Resize must be >= crop size "
                f"to avoid upscaling after crop."
            )

        # -- mean --------------------------------------------------------------
        if not isinstance(self.mean, (tuple, list)) or len(self.mean) != 3:
            raise TypeError(
                f"mean must be a tuple/list of 3 floats, got {self.mean!r}"
            )
        for i, v in enumerate(self.mean):
            if not isinstance(v, (int, float)):
                raise TypeError(f"mean[{i}] must be numeric, got {type(v).__name__}")
            if math.isnan(v) or math.isinf(v):
                raise ValueError(f"mean[{i}] is NaN/Inf: {v}")

        # -- std ---------------------------------------------------------------
        if not isinstance(self.std, (tuple, list)) or len(self.std) != 3:
            raise TypeError(
                f"std must be a tuple/list of 3 floats, got {self.std!r}"
            )
        for i, v in enumerate(self.std):
            if not isinstance(v, (int, float)):
                raise TypeError(f"std[{i}] must be numeric, got {type(v).__name__}")
            if math.isnan(v) or math.isinf(v):
                raise ValueError(f"std[{i}] is NaN/Inf: {v}")
            if v <= 0:
                raise ValueError(
                    f"std[{i}] must be > 0 (division by zero in normalization), got {v}"
                )


# %%
# =============================================================================
# CELL 5 -- PIL Validation & Conversion
# =============================================================================

def safe_image_to_rgb(img: Image.Image) -> Image.Image:
    """
    Converts any PIL image mode to standard RGB.

    Handles: L (grayscale), P (palette), LA, PA, RGBA, CMYK, I, F.

    Why this exists:
      ConvNeXt expects exactly 3 channels. Grayscale product photos,
      transparent PNGs, and palette-indexed GIFs from ecommerce
      datasets would cause channel-mismatch crashes without conversion.

    Args:
        img : PIL.Image.Image in any mode.

    Returns:
        PIL.Image.Image in RGB mode (3 channels).

    Raises:
        TypeError  : If img is not a PIL.Image.Image instance.
        ValueError : If conversion fails (e.g., corrupted pixel data).
    """
    if not isinstance(img, Image.Image):
        raise TypeError(
            f"safe_image_to_rgb() expected PIL.Image.Image, "
            f"got {type(img).__name__}."
        )
    if img.mode == "RGB":
        return img

    original_mode = img.mode
    try:
        converted = img.convert("RGB")
    except Exception as exc:
        raise ValueError(
            f"Failed to convert image from mode '{original_mode}' to RGB. "
            f"Original error: {exc}"
        ) from exc

    logger.debug(f"Converted image mode: {original_mode} -> RGB")
    return converted


def validate_image_input(img: Any) -> None:
    """
    Pre-transform validation gate for PIL images.

    Call this BEFORE passing an image to any transform pipeline.
    Catches degenerate inputs that would produce garbage tensors
    or silently corrupt downstream embeddings.

    Checks:
      - Instance type is PIL.Image.Image
      - Image has non-zero, positive dimensions
      - Image mode is not unknown/empty

    Args:
        img : Object to validate.

    Raises:
        TypeError  : If not a PIL.Image.Image.
        ValueError : If dimensions are invalid or mode is empty.
    """
    if not isinstance(img, Image.Image):
        raise TypeError(
            f"validate_image_input() expected PIL.Image.Image, "
            f"got {type(img).__name__}. "
            f"Use safe_load_image() to load images from disk."
        )

    w, h = img.size
    if w <= 0 or h <= 0:
        raise ValueError(
            f"Image has invalid dimensions: width={w}, height={h}. "
            f"Both must be > 0."
        )

    if not img.mode:
        raise ValueError(
            "Image has empty mode string. File may be corrupted."
        )


# %%
# =============================================================================
# CELL 6 -- Tensor Output Validation
# =============================================================================

def validate_tensor_output(
    tensor,
    expected_channels: int = 3,
    expected_size: Optional[Tuple[int, int]] = None,
) -> None:
    """
    Post-transform validation gate for image tensors.

    Call this AFTER transform pipeline produces a tensor and BEFORE
    the tensor enters collation or the encoder. Catches NaN injection,
    Inf corruption, dtype mistakes, and shape violations.

    Supports both single-image (3, H, W) and batched (B, 3, H, W) tensors.
    Device-agnostic: works on CPU and CUDA tensors without transfer.

    Args:
        tensor            : The tensor to validate.
        expected_channels : Expected channel count (default 3 for RGB).
        expected_size     : Optional (H, W) to verify spatial dimensions.

    Raises:
        TypeError  : If not a torch.Tensor.
        ValueError : If rank, channels, dtype, shape, NaN, or Inf checks fail.
    """
    import torch

    # -- Type check ------------------------------------------------------------
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"validate_tensor_output() expected torch.Tensor, "
            f"got {type(tensor).__name__}."
        )

    # -- Rank check (3D single or 4D batch) ------------------------------------
    if tensor.ndim == 3:
        C, H, W = tensor.shape
        B = None
    elif tensor.ndim == 4:
        B, C, H, W = tensor.shape
        if B == 0:
            raise ValueError(
                "Image tensor has batch_size=0 (empty batch)."
            )
    else:
        raise ValueError(
            f"Image tensor must be 3D (C,H,W) or 4D (B,C,H,W), "
            f"got {tensor.ndim}D with shape {list(tensor.shape)}."
        )

    # -- Channel check ---------------------------------------------------------
    if C != expected_channels:
        raise ValueError(
            f"Image tensor has {C} channels, expected {expected_channels}. "
            f"Shape: {list(tensor.shape)}. "
            f"Verify safe_image_to_rgb() ran before transforms."
        )

    # -- Spatial dimension check -----------------------------------------------
    if H <= 0 or W <= 0:
        raise ValueError(
            f"Image tensor has invalid spatial dims: H={H}, W={W}. "
            f"Both must be > 0."
        )

    if expected_size is not None:
        eH, eW = expected_size
        if (H, W) != (eH, eW):
            raise ValueError(
                f"Image tensor spatial dims ({H}, {W}) do not match "
                f"expected ({eH}, {eW}). Check resize/crop configuration."
            )

    # -- Dtype check (must be floating point) ----------------------------------
    if not tensor.is_floating_point():
        raise ValueError(
            f"Image tensor has non-float dtype ({tensor.dtype}), "
            f"expected torch.float32. Do not cast -- fix transform pipeline. "
            f"Ensure transforms.ToTensor() is included."
        )

    # -- NaN check (military-grade) --------------------------------------------
    if torch.isnan(tensor).any():
        nan_count = torch.isnan(tensor).sum().item()
        total = tensor.numel()
        raise ValueError(
            f"Image tensor contains {nan_count}/{total} NaN values. "
            f"Shape: {list(tensor.shape)}, dtype: {tensor.dtype}, "
            f"device: {tensor.device}. "
            f"Check normalization std (division by zero?) or corrupted input."
        )

    # -- Inf check (military-grade) --------------------------------------------
    if torch.isinf(tensor).any():
        inf_count = torch.isinf(tensor).sum().item()
        total = tensor.numel()
        raise ValueError(
            f"Image tensor contains {inf_count}/{total} Inf values. "
            f"Shape: {list(tensor.shape)}, dtype: {tensor.dtype}, "
            f"device: {tensor.device}. "
            f"Check normalization or augmentation overflow."
        )

    logger.debug(
        f"Tensor valid | shape={list(tensor.shape)} | "
        f"dtype={tensor.dtype} | device={tensor.device}"
    )


# %%
# =============================================================================
# CELL 6b -- PIL Size Validation
# =============================================================================

def validate_pil_size(size: Any, name: str = "fallback_size") -> Tuple[int, int]:
    """
    Validates that a value is a valid PIL image size: (width, height).

    PIL uses (width, height) ordering, NOT (height, width).
    This helper catches invalid caller input before it reaches
    Image.new(), producing clear error messages instead of
    cryptic PIL internals.

    Args:
        size : Value to validate.
        name : Parameter name for error messages.

    Returns:
        The validated (width, height) tuple, unchanged.

    Raises:
        TypeError  : If size is not a tuple of exactly 2 ints,
                     or if members are booleans.
        ValueError : If either dimension is zero or negative.
    """
    if not isinstance(size, tuple):
        raise TypeError(
            f"{name} must be a tuple of 2 positive ints (width, height), "
            f"got {type(size).__name__}: {size!r}"
        )
    if len(size) != 2:
        raise TypeError(
            f"{name} must be a tuple of exactly 2 ints (width, height), "
            f"got tuple of length {len(size)}: {size!r}"
        )
    for i, v in enumerate(size):
        label = "width" if i == 0 else "height"
        if isinstance(v, bool) or not isinstance(v, int):
            raise TypeError(
                f"{name}[{i}] ({label}) must be a positive int, "
                f"got {type(v).__name__}: {v!r}. "
                f"PIL expects (width, height) as integers."
            )
        if v <= 0:
            raise ValueError(
                f"{name}[{i}] ({label}) must be > 0, got {v}. "
                f"PIL expects (width, height) as positive integers."
            )
    return size


# %%
# =============================================================================
# CELL 7 -- Safe Image Loader
# =============================================================================

def safe_load_image(
    image_path: Union[str, Path, None],
    fallback_size: Tuple[int, int] = INPUT_SIZE,
    strict: bool = False,
) -> Image.Image:
    """
    Safely loads a single PIL image with comprehensive edge-case handling.

    Behavior modes:
      strict=False (default, training):
        On any failure, logs warning and returns neutral gray fallback.
        DataLoader pipeline continues uninterrupted.
      strict=True (production validation):
        Raises explicit exceptions for every failure case.
        Use for data audits and pre-training integrity checks.

    Handles:
      - None / NaN / non-string paths           (Edge Case 1)
      - Empty / whitespace-only paths            (Edge Case 1b)
      - Missing files on disk                    (Edge Case 2)
      - Corrupted / unreadable files             (Edge Case 3)
      - Truncated files (LOAD_TRUNCATED_IMAGES)  (Edge Case 3b)
      - Grayscale (L mode) images                (Edge Case 4)
      - RGBA / palette images                    (Edge Case 5)

    Fallback color (127, 127, 127):
      After ImageNet normalization this maps to approx (-0.02, -0.07, -0.14),
      near the dataset mean. Mathematically stable for latent clustering.

    Args:
        image_path    : Path to image file. Accepts str, Path, or None.
        fallback_size : PIL size as (width, height) for fallback image.
                        Default (224, 224). Only used when strict=False.
        strict        : If True, raise on any failure instead of fallback.

    Returns:
        PIL.Image.Image in RGB mode.
        In non-strict mode: never None, never raises.

    Raises (strict=True only):
        TypeError         : Invalid path type.
        ValueError        : Empty/NaN path or failed RGB conversion.
        FileNotFoundError : File not on disk.
        RuntimeError      : PIL open/verify failure.
    """
    # -- Validate fallback_size before it can reach Image.new() ----------------
    fallback_size = validate_pil_size(fallback_size, name="fallback_size")

    def _fallback(reason: str) -> Image.Image:
        logger.warning(
            f"Image fallback | reason={reason} | path={image_path}"
        )
        return Image.new("RGB", fallback_size, color=FALLBACK_COLOR)

    # -- Edge Case 1: None / NaN / non-string/Path -----------------------------
    if image_path is None:
        if strict:
            raise ValueError("safe_load_image/path_check: path is None.")
        return _fallback("path is None")

    if isinstance(image_path, float) and math.isnan(image_path):
        if strict:
            raise ValueError("safe_load_image/path_check: path is NaN.")
        return _fallback("path is NaN")

    if not isinstance(image_path, (str, Path)):
        if strict:
            raise TypeError(
                f"safe_load_image/path_check: expected str/Path, "
                f"got {type(image_path).__name__}."
            )
        return _fallback(f"path is {type(image_path).__name__}, expected str/Path")

    path_str = str(image_path).strip()

    # -- Edge Case 1b: Empty / whitespace-only ---------------------------------
    if not path_str:
        if strict:
            raise ValueError(
                "safe_load_image/path_check: path is empty or whitespace-only."
            )
        return _fallback("path is empty or whitespace-only")

    # -- Edge Case 2: File not on disk -----------------------------------------
    if not os.path.exists(path_str):
        if strict:
            raise FileNotFoundError(
                f"safe_load_image/file_check: file not found: '{path_str}'"
            )
        return _fallback("file not found on disk")

    # -- Edge Case 3: Corrupted / truncated file -------------------------------
    try:
        img = Image.open(path_str)
        img.verify()                        # integrity check (no full decode)
        img = Image.open(path_str)          # must re-open after verify()
    except Exception as exc:
        if strict:
            raise RuntimeError(
                f"safe_load_image/open_verify: PIL failed for '{path_str}': {exc}"
            ) from exc
        return _fallback(f"PIL open/verify failed: {exc}")

    # -- Edge Cases 4 + 5: Grayscale, RGBA, palette -> force RGB ---------------
    try:
        img = safe_image_to_rgb(img)
    except (TypeError, ValueError) as exc:
        if strict:
            raise ValueError(
                f"safe_load_image/rgb_convert: failed for '{path_str}': {exc}"
            ) from exc
        return _fallback(f"RGB conversion failed: {exc}")

    return img


# %%
# =============================================================================
# CELL 8 -- Validated Transform Wrapper
# =============================================================================
# This wrapper ensures every image passing through a transform pipeline
# is validated BEFORE and AFTER transformation. No caller can skip it.
#
# Ownership boundaries (CPU/GPU hybrid future-proofing):
#   transforms.py -- sample-level, stateless, CPU-only PIL->tensor
#   dataset.py    -- sample identity, __getitem__ orchestration
#   collate.py    -- batch assembly from individual samples
#   train.py      -- GPU transfer (non_blocking=True), pin_memory, prefetch
#   fusion.py     -- modality fusion only, never touches preprocessing
#
# Future GPU augmentation (Kornia, torchvision.v2) is a SEPARATE stage
# owned by train.py or a dedicated GPU transform module. This file must
# NEVER call .cuda(), .to(device), or initialize GPU state.

class ValidatedImageTransform:
    """
    Stateless wrapper that validates PIL input and tensor output
    around a torchvision transform pipeline.

    This is the ONLY callable returned by build_*_transforms() and
    get_transforms(). It guarantees that every image tensor entering
    the pipeline is contract-safe before reaching collation or the encoder.

    Properties:
      - Stateless: no counters, caches, queues, or epoch tracking
      - Worker-safe: safe for DataLoader num_workers > 0
      - Queue-safe: no batch ordering assumptions
      - Device-agnostic: no GPU init, no .cuda() calls
      - Modality-isolated: only processes image data
    """

    __slots__ = ("transform", "config", "mode")

    def __init__(self, transform, config: TransformConfig, mode: str) -> None:
        self.transform = transform
        self.config = config
        self.mode = mode

    def __call__(self, img) -> "torch.Tensor":
        # 1. Validate caller passed a usable PIL image
        validate_image_input(img)
        # 2. Force RGB -- handles direct L/RGBA/P/CMYK inputs
        #    that bypass safe_load_image() (notebooks, tests, dataset.py)
        img = safe_image_to_rgb(img)
        # 3. Re-validate after conversion (confirms valid mode + dims)
        validate_image_input(img)
        # 4. Apply transform pipeline (resize, crop, augment, normalize)
        tensor = self.transform(img)
        # 5. Validate output tensor (shape, dtype, NaN, Inf)
        validate_tensor_output(
            tensor,
            expected_channels=3,
            expected_size=self.config.input_size,
        )
        return tensor

    def __repr__(self) -> str:
        return (
            f"ValidatedImageTransform(mode='{self.mode}', "
            f"input_size={self.config.input_size}, "
            f"resize_size={self.config.resize_size})"
        )


# %%
# =============================================================================
# CELL 9 -- Transform Pipeline Builders
# =============================================================================

def build_train_transforms(
    config: Optional[TransformConfig] = None,
) -> ValidatedImageTransform:
    """
    Stochastic augmentation pipeline for training.

    Augmentations chosen to:
      - Preserve product identity, shape, and category semantics
      - Improve generalization across lighting and orientation variants

    Explicitly excluded:
      - RandomResizedCrop : destroys product boundary completeness
      - Perspective/Shear : distorts product geometry
      - Vertical flip     : unnatural for ecommerce images
      - Strong color drop : destroys texture and material signal

    Online augmentation contract:
      Each call produces a DIFFERENT stochastic view of the same image.
      The sample index is NEVER altered -- modality alignment is preserved.

    Args:
        config : TransformConfig. If None, uses module-level defaults.

    Returns:
        ValidatedImageTransform wrapping the augmentation pipeline.
    """
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    if config is None:
        config = TransformConfig()

    pipeline = transforms.Compose([
        transforms.Resize(
            config.resize_size,
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(config.input_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.mean, std=config.std),
    ])
    return ValidatedImageTransform(pipeline, config, mode="train")


def build_eval_transforms(
    config: Optional[TransformConfig] = None,
) -> ValidatedImageTransform:
    """
    Deterministic transform pipeline for evaluation, inference, and
    embedding extraction. No random operations -- guarantees reproducible
    latent vectors across runs.

    Args:
        config : TransformConfig. If None, uses module-level defaults.

    Returns:
        ValidatedImageTransform wrapping the deterministic pipeline.
    """
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    if config is None:
        config = TransformConfig()

    pipeline = transforms.Compose([
        transforms.Resize(
            config.resize_size,
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(config.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.mean, std=config.std),
    ])
    return ValidatedImageTransform(pipeline, config, mode="eval")


# %%
# =============================================================================
# CELL 10 -- Transform Router
# =============================================================================

_VALID_MODES = frozenset({"train", "eval", "inference", "test"})

def get_transforms(
    config: Optional[TransformConfig] = None,
    mode: str = "eval",
) -> ValidatedImageTransform:
    """
    Config-driven convenience router for Dataset classes and inference scripts.

    Args:
        config : TransformConfig instance. Falls back to module-level
                 defaults if None -- preserves backward compatibility.
        mode   : "train"                    -> augmented pipeline
                 "eval"/"inference"/"test"  -> deterministic pipeline

    Returns:
        ValidatedImageTransform (callable, stateless, worker-safe).

    Raises:
        TypeError  : If mode is not a string.
        ValueError : If mode is empty or unrecognized.
    """
    if not isinstance(mode, str):
        raise TypeError(
            f"get_transforms(): mode must be str, got {type(mode).__name__} "
            f"(value: {mode!r}). Accepted: {sorted(_VALID_MODES)}."
        )
    mode = mode.strip().lower()
    if not mode:
        raise ValueError(
            f"get_transforms(): mode is empty after strip(). "
            f"Accepted: {sorted(_VALID_MODES)}."
        )
    if mode == "train":
        return build_train_transforms(config)
    elif mode in ("eval", "inference", "test"):
        return build_eval_transforms(config)
    else:
        raise ValueError(
            f"get_transforms(): unrecognized mode '{mode}'. "
            f"Accepted: {sorted(_VALID_MODES)}."
        )


# %%
# =============================================================================
# CELL 11 -- Smoke Test (python data_pipeline/transforms.py)
# =============================================================================

if __name__ == "__main__":

    logging.basicConfig(
        level   = logging.DEBUG,
        format  = "[%(asctime)s] [%(levelname)s] %(name)s -- %(message)s",
        datefmt = "%H:%M:%S",
    )

    logger.info("=" * 60)
    logger.info("  data_pipeline/transforms.py -- smoke test")
    logger.info("=" * 60)

    passed = 0
    total  = 0

    def check(label, ok):
        global passed, total
        total += 1
        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        print(f"    [{status}] {label}")
        return ok

    try:
        import torch

        # ==============================================================
        # 1. TransformConfig defaults + validation
        # ==============================================================
        print("\n  1. TransformConfig...")
        cfg = TransformConfig()
        check("default input_size", cfg.input_size == (224, 224))
        check("default resize_size", cfg.resize_size == 256)
        check("default mean", cfg.mean == IMAGENET_MEAN)
        check("default std", cfg.std == IMAGENET_STD)

        # Invalid configs must fail at construction
        try:
            TransformConfig(input_size=(0, 224))
            check("input_size zero rejected", False)
        except ValueError:
            check("input_size zero rejected", True)

        try:
            TransformConfig(resize_size=100, input_size=(224, 224))
            check("resize < crop rejected", False)
        except ValueError:
            check("resize < crop rejected", True)

        try:
            TransformConfig(std=(0.0, 0.224, 0.225))
            check("zero std rejected", False)
        except ValueError:
            check("zero std rejected", True)

        try:
            TransformConfig(mean=(float("nan"), 0.456, 0.406))
            check("NaN mean rejected", False)
        except ValueError:
            check("NaN mean rejected", True)

        try:
            TransformConfig(input_size="bad")
            check("wrong input_size type rejected", False)
        except TypeError:
            check("wrong input_size type rejected", True)

        # ==============================================================
        # 2. safe_load_image (fallback mode)
        # ==============================================================
        print("\n  2. safe_load_image(strict=False)...")
        check("None path fallback", safe_load_image(None).size == INPUT_SIZE)
        check("empty path fallback", safe_load_image("").size == INPUT_SIZE)
        check("whitespace fallback", safe_load_image("   ").size == INPUT_SIZE)
        check("NaN path fallback", safe_load_image(float("nan")).size == INPUT_SIZE)
        check("int path fallback", safe_load_image(42).size == INPUT_SIZE)
        check("missing file fallback", safe_load_image("/nonexistent/img.jpg").size == INPUT_SIZE)

        # ==============================================================
        # 3. safe_load_image (strict mode)
        # ==============================================================
        print("\n  3. safe_load_image(strict=True)...")
        try:
            safe_load_image(None, strict=True)
            check("strict None raises", False)
        except ValueError:
            check("strict None raises", True)

        try:
            safe_load_image("", strict=True)
            check("strict empty raises", False)
        except ValueError:
            check("strict empty raises", True)

        try:
            safe_load_image("/nonexistent/img.jpg", strict=True)
            check("strict missing raises", False)
        except FileNotFoundError:
            check("strict missing raises", True)

        try:
            safe_load_image(42, strict=True)
            check("strict int raises TypeError", False)
        except TypeError:
            check("strict int raises TypeError", True)

        # ==============================================================
        # 4. safe_image_to_rgb conversions
        # ==============================================================
        print("\n  4. safe_image_to_rgb()...")
        check("grayscale->RGB", safe_image_to_rgb(Image.new("L", (100, 100))).mode == "RGB")
        check("RGBA->RGB", safe_image_to_rgb(Image.new("RGBA", (100, 100))).mode == "RGB")
        check("palette->RGB", safe_image_to_rgb(Image.new("P", (100, 100))).mode == "RGB")
        check("RGB no-op", safe_image_to_rgb(Image.new("RGB", (100, 100))).mode == "RGB")

        # ==============================================================
        # 5. validate_image_input
        # ==============================================================
        print("\n  5. validate_image_input()...")
        validate_image_input(Image.new("RGB", (224, 224)))
        check("valid PIL passes", True)

        try:
            validate_image_input("not_an_image")
            check("string rejected", False)
        except TypeError:
            check("string rejected", True)

        try:
            validate_image_input(torch.zeros(3, 224, 224))
            check("tensor rejected", False)
        except TypeError:
            check("tensor rejected", True)

        # ==============================================================
        # 6. ValidatedImageTransform + auto-validation
        # ==============================================================
        print("\n  6. ValidatedImageTransform auto-validation...")
        train_tfm = build_train_transforms()
        eval_tfm  = build_eval_transforms()
        check("train returns ValidatedImageTransform", isinstance(train_tfm, ValidatedImageTransform))
        check("eval returns ValidatedImageTransform", isinstance(eval_tfm, ValidatedImageTransform))

        test_img = Image.new("RGB", (300, 400), color=(100, 150, 200))

        # Successful end-to-end
        eval_tensor = eval_tfm(test_img)
        check("eval shape (3,224,224)", eval_tensor.shape == (3, 224, 224))
        check("eval dtype float32", eval_tensor.dtype == torch.float32)

        train_tensor = train_tfm(test_img)
        check("train shape (3,224,224)", train_tensor.shape == (3, 224, 224))
        check("train dtype float32", train_tensor.dtype == torch.float32)

        # Auto-validation rejects non-PIL
        try:
            eval_tfm("not_a_pil_image")
            check("auto-rejects non-PIL input", False)
        except TypeError:
            check("auto-rejects non-PIL input", True)

        # ==============================================================
        # 7. get_transforms router + type safety
        # ==============================================================
        print("\n  7. get_transforms() type safety...")
        _ = get_transforms(mode="train")
        _ = get_transforms(mode="eval")
        _ = get_transforms(mode="inference")
        _ = get_transforms(mode="test")
        check("valid modes accepted", True)

        try:
            get_transforms(mode=123)
            check("non-str mode rejected", False)
        except TypeError:
            check("non-str mode rejected", True)

        try:
            get_transforms(mode="   ")
            check("empty mode rejected", False)
        except ValueError:
            check("empty mode rejected", True)

        try:
            get_transforms(mode="invalid_mode")
            check("invalid mode rejected", False)
        except ValueError:
            check("invalid mode rejected", True)

        # ==============================================================
        # 8. validate_tensor_output
        # ==============================================================
        print("\n  8. validate_tensor_output()...")
        validate_tensor_output(eval_tensor, expected_size=(224, 224))
        check("valid 3D passes", True)

        batched = eval_tensor.unsqueeze(0)
        validate_tensor_output(batched, expected_size=(224, 224))
        check("valid 4D passes", True)

        try:
            validate_tensor_output(torch.full((3, 224, 224), float("nan")))
            check("NaN detected", False)
        except ValueError:
            check("NaN detected", True)

        try:
            validate_tensor_output(torch.full((3, 224, 224), float("inf")))
            check("Inf detected", False)
        except ValueError:
            check("Inf detected", True)

        try:
            validate_tensor_output(torch.zeros(3, 224, 224, dtype=torch.int32))
            check("int dtype rejected", False)
        except ValueError:
            check("int dtype rejected", True)

        try:
            validate_tensor_output(torch.zeros(1, 224, 224))
            check("wrong channels rejected", False)
        except ValueError:
            check("wrong channels rejected", True)

        try:
            validate_tensor_output(torch.zeros(224, 224))
            check("2D rank rejected", False)
        except ValueError:
            check("2D rank rejected", True)

        try:
            validate_tensor_output(torch.zeros(3, 100, 100), expected_size=(224, 224))
            check("size mismatch rejected", False)
        except ValueError:
            check("size mismatch rejected", True)

        try:
            validate_tensor_output("not_a_tensor")
            check("non-tensor TypeError", False)
        except TypeError:
            check("non-tensor TypeError", True)

        # ==============================================================
        # 9. Eval determinism
        # ==============================================================
        print("\n  9. Determinism & stochasticity...")
        e1 = eval_tfm(test_img)
        e2 = eval_tfm(test_img)
        check("eval deterministic", torch.equal(e1, e2))

        # ==============================================================
        # 10. Train stochasticity
        # ==============================================================
        results = [train_tfm(test_img) for _ in range(10)]
        all_same = all(torch.equal(results[0], r) for r in results[1:])
        check("train stochastic", not all_same)

        # ==============================================================
        # Summary
        # ==============================================================
        print(f"\n{'=' * 60}")
        status = "PASS" if passed == total else "FAIL"
        print(f"  [{status}]  {passed}/{total} checks passed")
        print("=" * 60)

        if passed < total:
            sys.exit(1)

    except Exception as e:
        logger.exception(f"[FAIL] SMOKE TEST FAILED: {e}")
        sys.exit(1)

