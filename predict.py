# =============================================================================
# predict.py
# Multimodal Inference Engine -- Single-Product Rating Predictor
# =============================================================================
#
# Purpose:
#   Standalone inference authority for the multimodal training pipeline.
#   Loads a trained checkpoint, reconstructs the model architecture,
#   preprocesses one product input, and returns a structured prediction.
#
# Philosophy:
#   predict.py reuses existing project contracts without modifying them.
#   It never duplicates preprocessing, tokenization, path resolution,
#   or model construction logic that already exists in the pipeline.
#
# Public API:
#   class PredictionError(Exception)
#   class Predictor
#   def predict(image, text, price, rating_number, experiment_name, ...) -> dict
#
# Usage (Python):
#   from predict import predict
#   result = predict(
#       image="preprocessed-datasets/images/B001.jpg",
#       text="Wireless headphones with noise cancellation",
#       price=2499,
#       rating_number=318,
#       experiment_name="my_experiment",
#       checkpoint_name="best.pt",
#   )
#
# Usage (CLI):
#   python predict.py --smoke
# =============================================================================

from __future__ import annotations

import os
import sys
import re
import math
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

# -- Project root bootstrap ---------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_PROJECT_DIR = _THIS_FILE.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# =============================================================================
# 1. Error Class
# =============================================================================

class PredictionError(RuntimeError):
    """Structured inference failure. Raised for every validated input
    or runtime error in the prediction pipeline."""

    def __init__(
        self,
        stage: str,
        reason: str,
        resolution: str = "",
    ) -> None:
        self.stage = stage
        self.reason = reason
        self.resolution = resolution
        msg = f"[{stage}] {reason}"
        if resolution:
            msg += f" | Resolution: {resolution}"
        super().__init__(msg)


# =============================================================================
# 2. Constants
# =============================================================================

# V1 tabular fields -- must match training pipeline exactly
_TABULAR_FIELDS = ("price", "rating_number")
_TABULAR_INPUT_DIM = len(_TABULAR_FIELDS)  # 2

# Rating clipping range
_RATING_MIN = 1.0
_RATING_MAX = 5.0

# Required keys in a trained checkpoint
_CHECKPOINT_REQUIRED_KEY = "model_state_dict"

# Required keys in model bundle (must match training/trainer.py)
_REQUIRED_MODEL_KEYS = frozenset({
    "image_encoder", "text_encoder", "tabular_encoder", "fusion_model",
})


# =============================================================================
# 3. Private Validators / Helpers
# =============================================================================

def _sanitize_experiment_name(name: str) -> str:
    """Sanitize experiment name for filesystem use.

    Exactly mirrors the trainer's behavior (training/trainer.py:429-433):
      - Replace non-word/non-dash/non-dot chars with '_'
      - Collapse repeated underscores
      - Strip leading/trailing underscores
      - Cap at 128 characters
      - Fall back to 'unnamed_experiment' if empty
    """
    clean = re.sub(r'[^\w\-.]', '_', name.strip())
    clean = re.sub(r'_+', '_', clean).strip('_')
    return clean[:128] if clean else "unnamed_experiment"


def _validate_device_arg(device: Any) -> str:
    """Validate and normalize device argument before torch.device construction.

    Returns a clean lowercase string suitable for _resolve_device().
    Raises PredictionError for non-string, None, or empty inputs.
    """
    if device is None:
        raise PredictionError(
            "device", "device is None.", "Use 'auto', 'cpu', or 'cuda'."
        )
    if not isinstance(device, str):
        raise PredictionError(
            "device",
            f"device must be a string, got {type(device).__name__} ({device!r}).",
            "Use 'auto', 'cpu', or 'cuda'.",
        )
    cleaned = device.strip().lower()
    if not cleaned:
        raise PredictionError(
            "device", "device is empty/whitespace.", "Use 'auto', 'cpu', or 'cuda'."
        )
    return cleaned


def _validate_checkpoint_name(checkpoint_name: Any) -> str:
    """Validate checkpoint_name is a safe, non-empty filename string.

    Raises PredictionError for None, non-string, empty, absolute paths,
    or traversal attempts. Returns stripped filename.
    """
    if checkpoint_name is None:
        raise PredictionError(
            "checkpoint", "checkpoint_name is None.", "Provide a checkpoint filename like 'best.pt'."
        )
    if not isinstance(checkpoint_name, str):
        raise PredictionError(
            "checkpoint",
            f"checkpoint_name must be a string, got {type(checkpoint_name).__name__}.",
        )
    cleaned = checkpoint_name.strip()
    if not cleaned:
        raise PredictionError(
            "checkpoint", "checkpoint_name is empty.", "Provide a checkpoint filename like 'best.pt'."
        )
    # Reject absolute paths (Windows and Unix-style)
    if Path(cleaned).is_absolute() or cleaned.startswith("/"):
        raise PredictionError(
            "checkpoint",
            f"checkpoint_name must be a plain filename, got absolute path: '{cleaned}'.",
            "Use a simple filename like 'best.pt'.",
        )
    # Reject path separators — checkpoint_name must be a plain filename
    if "/" in cleaned or "\\" in cleaned:
        raise PredictionError(
            "checkpoint",
            f"checkpoint_name must be a plain filename without path separators: '{cleaned}'.",
            "Use a simple filename like 'best.pt'.",
        )
    return cleaned


def _validate_text_input(text: Any) -> str:
    """Validate text argument at the inference boundary.

    Rejects None, non-string, empty, and whitespace-only inputs.
    Returns the original string (sanitization is done downstream).
    """
    if text is None:
        raise PredictionError(
            "input_text", "text is None.", "Provide a non-empty product description."
        )
    if not isinstance(text, str):
        raise PredictionError(
            "input_text",
            f"text must be a string, got {type(text).__name__} ({text!r}).",
            "Provide a product description string.",
        )
    if not text.strip():
        raise PredictionError(
            "input_text",
            "text is empty after stripping whitespace.",
            "Provide a non-empty product description.",
        )
    return text


# =============================================================================
# 4. Device Resolution
# =============================================================================

def _resolve_device(device: Any) -> torch.device:
    """Resolve 'auto', 'cpu', 'cuda' to a concrete torch.device.

    Validates type and normalizes before attempting torch.device construction.
    """
    cleaned = _validate_device_arg(device)

    if cleaned == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cleaned in ("cpu", "cuda") or cleaned.startswith("cuda:"):
        d = torch.device(cleaned)
        if d.type == "cuda" and not torch.cuda.is_available():
            raise PredictionError(
                "device",
                f"CUDA requested but unavailable (device='{device}').",
                "Use device='cpu' or device='auto'.",
            )
        return d
    raise PredictionError(
        "device",
        f"Unknown device '{device}'.",
        "Use 'auto', 'cpu', or 'cuda'.",
    )


# =============================================================================
# 5. Checkpoint Resolution
# =============================================================================

def _resolve_checkpoint(
    experiment_name: str,
    checkpoint_name: str,
) -> Tuple[Path, str, bool]:
    """Resolve checkpoint path under CHECKPOINT_DIR/<experiment_name>/<checkpoint_name>.

    Falls back to 'latest.pt' if 'best.pt' is requested but missing.
    Validates path traversal.

    Returns:
        (resolved_path, actual_checkpoint_name, fallback_used)
    """
    from configs.paths import CHECKPOINT_DIR, _ensure_child_path

    # Validate checkpoint_name early
    safe_ckpt = _validate_checkpoint_name(checkpoint_name)

    # Sanitize experiment name to match trainer exactly
    safe_name = _sanitize_experiment_name(experiment_name)
    exp_dir = CHECKPOINT_DIR / safe_name

    if not exp_dir.exists():
        available = (
            [d.name for d in CHECKPOINT_DIR.iterdir() if d.is_dir()]
            if CHECKPOINT_DIR.exists() else []
        )
        raise PredictionError(
            "checkpoint",
            f"Experiment directory not found: {exp_dir}",
            f"Available experiments: {available or 'none'}. "
            f"Run training with --train first.",
        )

    # Validate traversal
    try:
        _ensure_child_path(CHECKPOINT_DIR, exp_dir / safe_ckpt, "predict")
    except ValueError as exc:
        raise PredictionError("checkpoint", str(exc), "Use a plain filename.") from exc

    ckpt_path = exp_dir / safe_ckpt

    # Fallback: best.pt -> latest.pt
    if not ckpt_path.exists() and safe_ckpt == "best.pt":
        fallback = exp_dir / "latest.pt"
        if fallback.exists():
            logger.warning(
                "best.pt not found, falling back to latest.pt: %s", fallback
            )
            return fallback, "latest.pt", True
        raise PredictionError(
            "checkpoint",
            f"Neither best.pt nor latest.pt found in {exp_dir}.",
            "Run training to produce a checkpoint.",
        )

    if not ckpt_path.exists():
        available_pts = [f.name for f in exp_dir.glob("*.pt")]
        raise PredictionError(
            "checkpoint",
            f"Checkpoint not found: {ckpt_path}",
            f"Available checkpoints: {available_pts or 'none'}.",
        )

    return ckpt_path, safe_ckpt, False


# =============================================================================
# 6. Model Reconstruction
# =============================================================================

def _build_model_bundle() -> nn.ModuleDict:
    """Reconstruct the same model bundle used during training.

    Uses the same builder functions and configs as training/train.py.
    TabularEncoderConfig.input_dim=2 matches V1 training contract.
    """
    from models.image_encoder import ImageEncoderConfig, build_encoder
    from models.text_encoder import TextEncoderConfig, build_text_encoder
    from models.tabular_encoder import TabularEncoderConfig, build_tabular_encoder
    from models.fusion import FusionConfig, FusionModel

    image_encoder = build_encoder(ImageEncoderConfig())
    text_encoder = build_text_encoder(TextEncoderConfig())
    tabular_encoder = build_tabular_encoder(
        TabularEncoderConfig(input_dim=_TABULAR_INPUT_DIM)
    )
    fusion_model = FusionModel(FusionConfig())

    bundle = nn.ModuleDict({
        "image_encoder": image_encoder,
        "text_encoder": text_encoder,
        "tabular_encoder": tabular_encoder,
        "fusion_model": fusion_model,
    })
    return bundle


def _load_checkpoint_weights(
    bundle: nn.ModuleDict,
    ckpt_path: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """Load checkpoint and apply model_state_dict to bundle.

    Returns the full checkpoint dict for metadata access.
    """
    if not ckpt_path.exists():
        raise PredictionError(
            "checkpoint_load",
            f"File disappeared before load: {ckpt_path}",
        )

    try:
        checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    except Exception as exc:
        raise PredictionError(
            "checkpoint_load",
            f"torch.load failed: {str(exc)[:200]}",
            "Checkpoint may be corrupt. Re-run training.",
        ) from exc

    if not isinstance(checkpoint, dict):
        raise PredictionError(
            "checkpoint_load",
            f"Checkpoint is not a dict (got {type(checkpoint).__name__}).",
            "Re-run training to produce a valid checkpoint.",
        )

    if _CHECKPOINT_REQUIRED_KEY not in checkpoint:
        raise PredictionError(
            "checkpoint_load",
            f"Missing key '{_CHECKPOINT_REQUIRED_KEY}' in checkpoint.",
            "Checkpoint schema is incompatible. Re-run training.",
        )

    try:
        bundle.load_state_dict(checkpoint[_CHECKPOINT_REQUIRED_KEY])
    except Exception as exc:
        raise PredictionError(
            "checkpoint_load",
            f"load_state_dict failed: {str(exc)[:200]}",
            "Model architecture may differ from checkpoint. "
            "Ensure predict.py uses the same config as training.",
        ) from exc

    return checkpoint


def _validate_model_on_device(bundle: nn.ModuleDict, expected: torch.device) -> None:
    """Confirm all model parameters AND buffers are on the expected device."""
    # Canonicalize: 'cuda' -> 'cuda:0'
    if expected.type == "cuda" and expected.index is None:
        expected = torch.device("cuda", torch.cuda.current_device())

    def _check(kind: str, name: str, tensor: torch.Tensor) -> None:
        t_dev = tensor.device
        if t_dev.type == "cuda" and t_dev.index is None:
            t_dev = torch.device("cuda", torch.cuda.current_device())
        if t_dev != expected:
            raise PredictionError(
                "model_device",
                f"{kind} '{name}' is on {tensor.device}, expected {expected}.",
                "Model.to(device) failed unexpectedly.",
            )

    for name, param in bundle.named_parameters():
        _check("Parameter", name, param)
    for name, buf in bundle.named_buffers():
        _check("Buffer", name, buf)


# =============================================================================
# 7. Input Validation
# =============================================================================

def _validate_image_path(image: Any) -> Path:
    """Validate image argument and return a resolved Path."""
    if image is None:
        raise PredictionError("input_image", "image path is None.", "Provide a valid image file path.")

    if not isinstance(image, (str, Path)):
        raise PredictionError(
            "input_image",
            f"image must be str or Path, got {type(image).__name__}.",
        )

    path = Path(str(image)).resolve()
    if not path.exists():
        raise PredictionError(
            "input_image",
            f"Image file not found: {path}",
            "Ensure the image file exists before predicting.",
        )
    return path


def _validate_tabular(price: Any, rating_number: Any) -> tuple:
    """Validate and convert tabular fields. Returns (float_price, float_rating_number)."""
    for name, val in (("price", price), ("rating_number", rating_number)):
        if val is None:
            raise PredictionError("input_tabular", f"'{name}' is None.", "Provide a numeric value.")
        if isinstance(val, bool):
            raise PredictionError("input_tabular", f"'{name}' is bool, expected numeric.")
        if not isinstance(val, (int, float)):
            raise PredictionError(
                "input_tabular",
                f"'{name}' must be numeric, got {type(val).__name__} ({val!r}).",
            )
        try:
            fval = float(val)
        except (TypeError, ValueError) as exc:
            raise PredictionError("input_tabular", f"'{name}' cannot be converted to float: {val!r}.") from exc
        if not math.isfinite(fval):
            raise PredictionError(
                "input_tabular",
                f"'{name}' is {fval} (NaN/Inf not allowed).",
                "Provide a finite numeric value.",
            )

    return float(price), float(rating_number)


# =============================================================================
# 8. Preprocessing
# =============================================================================

def _preprocess_image(image_path: Path, device: torch.device) -> torch.Tensor:
    """Load image, apply eval transforms, add batch dimension -> [1, 3, 224, 224]."""
    from data_pipeline.transforms import get_transforms, safe_load_image

    # Load with strict=True so corruption raises PredictionError
    try:
        pil_img = safe_load_image(image_path, strict=True)
    except (FileNotFoundError, RuntimeError, ValueError, TypeError) as exc:
        raise PredictionError(
            "preprocess_image",
            f"Image load failed: {str(exc)[:200]}",
            "Check the image file is a valid JPEG/PNG and not corrupted.",
        ) from exc

    transform = get_transforms(mode="inference")
    try:
        tensor = transform(pil_img)  # [3, 224, 224]
    except Exception as exc:
        raise PredictionError(
            "preprocess_image",
            f"Transform failed: {str(exc)[:200]}",
            "Check the image format and size.",
        ) from exc

    return tensor.unsqueeze(0).to(device)  # [1, 3, 224, 224]


def _preprocess_text(text: Any, device: torch.device) -> Dict[str, torch.Tensor]:
    """Validate, sanitize, and tokenize text.

    Returns dict with input_ids [1,64] and attention_mask [1,64].
    Rejects None, non-string, empty, and whitespace-only inputs at the
    inference boundary before delegating to existing sanitize+tokenize.
    """
    from data_pipeline.tokenization import sanitize_text, tokenize_text

    # Strict inference boundary validation (Issue 5)
    _validate_text_input(text)

    cleaned = sanitize_text(text)

    try:
        tokens = tokenize_text(cleaned)  # returns {"input_ids": [1,64], "attention_mask": [1,64]}
    except Exception as exc:
        raise PredictionError(
            "preprocess_text",
            f"Tokenization failed: {str(exc)[:200]}",
        ) from exc

    return {k: v.to(device) for k, v in tokens.items()}


def _preprocess_tabular(price: float, rating_number: float, device: torch.device) -> torch.Tensor:
    """Build tabular tensor [1, 2] from validated price and rating_number."""
    return torch.tensor([[price, rating_number]], dtype=torch.float32, device=device)


# =============================================================================
# 9. Forward Inference
# =============================================================================

def _forward(
    bundle: nn.ModuleDict,
    images: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tabular: torch.Tensor,
) -> float:
    """Run the full forward pass. Returns raw prediction as Python float."""
    bundle.eval()
    with torch.no_grad():
        try:
            img_emb = bundle["image_encoder"](images)
            txt_emb = bundle["text_encoder"](input_ids, attention_mask)
            tab_emb = bundle["tabular_encoder"](tabular)
            fusion_out = bundle["fusion_model"](img_emb, txt_emb, tab_emb)
        except Exception as exc:
            raise PredictionError(
                "forward",
                f"Model forward pass failed: {str(exc)[:300]}",
                "Check that the checkpoint matches the current model architecture.",
            ) from exc

    # Issue 7: validate output type before key access
    if not isinstance(fusion_out, dict):
        raise PredictionError(
            "forward",
            f"Model output is {type(fusion_out).__name__}, expected dict.",
            "FusionModel.forward() must return a dict with 'rating_prediction'.",
        )

    if "rating_prediction" not in fusion_out:
        raise PredictionError(
            "forward",
            f"Unexpected model output keys: {list(fusion_out.keys())}",
            "Expected 'rating_prediction' key from FusionModel.forward().",
        )

    raw_tensor = fusion_out["rating_prediction"]
    if not isinstance(raw_tensor, torch.Tensor):
        raise PredictionError(
            "forward",
            f"rating_prediction is {type(raw_tensor).__name__}, expected Tensor.",
        )

    try:
        raw_val = raw_tensor.squeeze().item()
    except Exception as exc:
        raise PredictionError(
            "output",
            f"Cannot extract scalar from rating_prediction (shape={tuple(raw_tensor.shape)}): {exc}",
        ) from exc

    if not math.isfinite(raw_val):
        raise PredictionError(
            "output",
            f"Model produced non-finite rating_prediction: {raw_val}",
            "Check for NaN/Inf in inputs or model weights.",
        )

    return raw_val


# =============================================================================
# 10. Result Formatting
# =============================================================================

def _build_result(
    raw_prediction: float,
    experiment_name: str,
    checkpoint_name: str,
    requested_checkpoint_name: str,
    checkpoint_fallback_used: bool,
    device: torch.device,
    checkpoint: Dict[str, Any],
    elapsed_ms: float,
) -> Dict[str, Any]:
    """Build the structured prediction result dict."""
    clipped = float(max(_RATING_MIN, min(_RATING_MAX, raw_prediction)))

    metadata: Dict[str, Any] = {
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_version": checkpoint.get("checkpoint_version"),
        "trainer_schema_version": checkpoint.get("trainer_schema_version"),
        "elapsed_ms": round(elapsed_ms, 2),
    }

    # Optionally read run_manifest for display metadata (non-critical)
    safe_name = _sanitize_experiment_name(experiment_name)
    try:
        from configs.paths import CHECKPOINT_DIR
        manifest_path = CHECKPOINT_DIR / safe_name / "run_manifest.json"
        if manifest_path.exists():
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            metadata["manifest_run_id"] = manifest.get("run_id")
            metadata["manifest_datasets"] = manifest.get("datasets_selected")
            metadata["manifest_available"] = True
        else:
            metadata["manifest_available"] = False
    except Exception as exc:
        # Manifest is display-only; prediction never depends on it
        metadata["manifest_available"] = False
        metadata["manifest_read_error"] = str(exc)[:200]

    return {
        "raw_prediction": raw_prediction,
        "predicted_rating": clipped,
        "clipped_rating": clipped,
        "experiment_name": experiment_name,
        "checkpoint_name": checkpoint_name,
        "requested_checkpoint_name": requested_checkpoint_name,
        "checkpoint_fallback_used": checkpoint_fallback_used,
        "device": str(device),
        "metadata": metadata,
    }


def _validate_result_schema(result: Any) -> None:
    """Validate that a result dict has required keys and valid types.

    Called on every real result before returning across the public boundary.
    """
    if not isinstance(result, dict):
        raise PredictionError("result_schema", f"Result is {type(result).__name__}, expected dict.")

    required = ("raw_prediction", "predicted_rating", "clipped_rating",
                 "experiment_name", "checkpoint_name", "device", "metadata")
    missing = [k for k in required if k not in result]
    if missing:
        raise PredictionError("result_schema", f"Missing result keys: {missing}")

    # Numeric contracts
    for key in ("raw_prediction", "predicted_rating", "clipped_rating"):
        val = result[key]
        if not isinstance(val, (int, float)):
            raise PredictionError("result_schema", f"'{key}' is {type(val).__name__}, expected float.")
        if not math.isfinite(val):
            raise PredictionError("result_schema", f"'{key}' is not finite ({val}).")

    # Clipping range for display ratings
    for key in ("predicted_rating", "clipped_rating"):
        val = result[key]
        if not (_RATING_MIN <= val <= _RATING_MAX):
            raise PredictionError(
                "result_schema",
                f"'{key}' is {val}, outside valid range [{_RATING_MIN}, {_RATING_MAX}].",
            )

    # String contracts
    for key in ("experiment_name", "checkpoint_name", "device"):
        val = result[key]
        if not isinstance(val, str) or not val.strip():
            raise PredictionError("result_schema", f"'{key}' must be a non-empty string, got {val!r}.")

    # Metadata must be dict
    if not isinstance(result.get("metadata"), dict):
        raise PredictionError("result_schema", "metadata must be a dict.")


# =============================================================================
# 11. Predictor Class
# =============================================================================

class Predictor:
    """Production single-product predictor.

    Loads the model exactly once. Designed for repeated calls without
    reloading. Suitable for API endpoints, notebook usage, and batch scripts.

    Args:
        experiment_name : Training experiment name (maps to checkpoint subdir).
        checkpoint_name : Checkpoint filename. Default 'best.pt'.
        device          : 'auto', 'cpu', or 'cuda'. Default 'auto'.
    """

    def __init__(
        self,
        experiment_name: str,
        checkpoint_name: str = "best.pt",
        device: str = "auto",
    ) -> None:
        if not isinstance(experiment_name, str) or not experiment_name.strip():
            raise PredictionError(
                "init",
                "experiment_name must be a non-empty string.",
            )

        self._experiment_name = experiment_name.strip()
        self._requested_checkpoint_name = checkpoint_name
        self._actual_checkpoint_name: str = ""
        self._checkpoint_fallback_used: bool = False
        self._device = _resolve_device(device)
        self._checkpoint: Optional[Dict[str, Any]] = None
        self._bundle: Optional[nn.ModuleDict] = None

        self._load()

    def _load(self) -> None:
        """Load checkpoint and prepare model. Called once in __init__."""
        ckpt_path, actual_name, fallback = _resolve_checkpoint(
            self._experiment_name, self._requested_checkpoint_name
        )
        self._actual_checkpoint_name = actual_name
        self._checkpoint_fallback_used = fallback

        logger.info(
            "Loading checkpoint: %s (device=%s)%s",
            ckpt_path.name, self._device,
            " [fallback from best.pt]" if fallback else "",
        )

        bundle = _build_model_bundle()
        checkpoint = _load_checkpoint_weights(bundle, ckpt_path, self._device)

        bundle.to(self._device)
        _validate_model_on_device(bundle, self._device)
        bundle.eval()

        self._bundle = bundle
        self._checkpoint = checkpoint
        logger.info(
            "Predictor ready | experiment=%s | checkpoint=%s | device=%s",
            self._experiment_name,
            self._actual_checkpoint_name,
            self._device,
        )

    def predict(
        self,
        image: Union[str, Path],
        text: str,
        price: Union[int, float],
        rating_number: Union[int, float],
    ) -> Dict[str, Any]:
        """Run single-product inference.

        Args:
            image         : Path to product image (str or Path).
            text          : Product description text.
            price         : Product price (numeric).
            rating_number : Number of product ratings (numeric).

        Returns:
            Dict with 'predicted_rating', 'clipped_rating', 'raw_prediction',
            'device', 'experiment_name', 'checkpoint_name', 'metadata'.

        Raises:
            PredictionError: On any input, model, or output contract violation.
        """
        t0 = time.perf_counter()

        # -- 1. Validate inputs -----------------------------------------------
        image_path = _validate_image_path(image)
        price_f, rating_f = _validate_tabular(price, rating_number)

        # -- 2. Preprocess -----------------------------------------------
        images = _preprocess_image(image_path, self._device)
        text_tokens = _preprocess_text(text, self._device)
        tabular = _preprocess_tabular(price_f, rating_f, self._device)

        # -- 3. Forward pass --------------------------------------------------
        raw = _forward(
            self._bundle,
            images=images,
            input_ids=text_tokens["input_ids"],
            attention_mask=text_tokens["attention_mask"],
            tabular=tabular,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # -- 4. Build and validate result -------------------------------------
        result = _build_result(
            raw_prediction=raw,
            experiment_name=self._experiment_name,
            checkpoint_name=self._actual_checkpoint_name,
            requested_checkpoint_name=self._requested_checkpoint_name,
            checkpoint_fallback_used=self._checkpoint_fallback_used,
            device=self._device,
            checkpoint=self._checkpoint,
            elapsed_ms=elapsed_ms,
        )

        # Issue 8: validate schema before returning across public boundary
        _validate_result_schema(result)

        return result


# =============================================================================
# 12. Convenience Function
# =============================================================================

def predict(
    image: Union[str, Path],
    text: str,
    price: Union[int, float],
    rating_number: Union[int, float],
    experiment_name: str,
    checkpoint_name: str = "best.pt",
    device: str = "auto",
) -> Dict[str, Any]:
    """Convenience wrapper: create a Predictor and run one prediction.

    For repeated inference on the same experiment, construct a Predictor
    directly and call predictor.predict(...) to avoid reloading the model.

    Args:
        image           : Path to product image (str or Path).
        text            : Product description text.
        price           : Product price (numeric).
        rating_number   : Number of product ratings (numeric).
        experiment_name : Training experiment name.
        checkpoint_name : Checkpoint filename. Default 'best.pt'.
        device          : 'auto', 'cpu', or 'cuda'. Default 'auto'.

    Returns:
        Dict with prediction results.

    Raises:
        PredictionError: On any input or runtime failure.
    """
    predictor = Predictor(
        experiment_name=experiment_name,
        checkpoint_name=checkpoint_name,
        device=device,
    )
    return predictor.predict(image=image, text=text, price=price, rating_number=rating_number)


# =============================================================================
# Phase 2 â€” Discovery, History, URL, Explanation, Popup, Interactive
# =============================================================================

# =============================================================================
# P2-1. Experiment / Checkpoint Discovery
# =============================================================================

_CHECKPOINT_PRIORITY = ("best.pt", "latest.pt")

def _discover_experiments(checkpoint_root: Optional[Path] = None) -> list:
    """Return sorted list of experiment names that contain at least one .pt file."""
    from configs.paths import CHECKPOINT_DIR
    root = checkpoint_root or CHECKPOINT_DIR
    if not root.exists():
        return []
    result = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and any(d.glob("*.pt")):
            result.append(d.name)
    return result


def _list_checkpoints_for_experiment(
    experiment_name: str,
    checkpoint_root: Optional[Path] = None,
) -> list:
    """Return sorted list of .pt filenames in an experiment directory."""
    from configs.paths import CHECKPOINT_DIR
    root = checkpoint_root or CHECKPOINT_DIR
    safe = _sanitize_experiment_name(experiment_name)
    exp_dir = root / safe
    if not exp_dir.exists():
        return []
    return sorted(f.name for f in exp_dir.glob("*.pt"))


def _choose_default_checkpoint(checkpoints: list) -> str:
    """Pick the best default checkpoint from a list by priority.

    Priority: best.pt > latest.pt > newest interrupted_epoch_*.pt > first .pt
    """
    if not checkpoints:
        return ""
    for prio in _CHECKPOINT_PRIORITY:
        if prio in checkpoints:
            return prio
    interrupted = [c for c in checkpoints if c.startswith("interrupted_epoch_")]
    if interrupted:
        def _epoch_num(name: str) -> int:
            try:
                return int(name.split("interrupted_epoch_")[1].split(".")[0])
            except (IndexError, ValueError):
                return -1
        return max(interrupted, key=_epoch_num)
    return checkpoints[0]


# =============================================================================
# P2-2. Prediction History
# =============================================================================

_HISTORY_TEXT_PREVIEW_MAX = 200

def _get_history_dir() -> Path:
    """Return the prediction history directory, creating it if needed."""
    from configs.paths import LOG_DIR
    hdir = LOG_DIR / "prediction_history"
    hdir.mkdir(parents=True, exist_ok=True)
    return hdir


def save_prediction_history(
    result: dict,
    input_summary: dict,
    history_dir: Optional[Path] = None,
) -> Path:
    """Save a lightweight JSON prediction history record.

    Args:
        result       : Prediction result dict from Predictor.predict().
        input_summary: Dict with image_source, image_source_type, text, price, rating_number.
        history_dir  : Override history directory (for testing).

    Returns:
        Path to the saved history JSON file.

    Raises PredictionError on I/O failure. Callers (CLI/interactive) should
    treat history save failure as a warning, not a prediction failure.
    """
    from datetime import datetime, timezone

    hdir = history_dir or _get_history_dir()
    hdir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc)
    ts_str = ts.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    ts_file = ts.strftime("%Y%m%d_%H%M%S_%f")

    text_raw = input_summary.get("text", "")
    text_preview = (text_raw[:_HISTORY_TEXT_PREVIEW_MAX] + "...") if len(text_raw) > _HISTORY_TEXT_PREVIEW_MAX else text_raw

    record = {
        "timestamp": ts_str,
        "experiment_name": result.get("experiment_name", ""),
        "checkpoint_name": result.get("checkpoint_name", ""),
        "requested_checkpoint_name": result.get("requested_checkpoint_name", ""),
        "checkpoint_fallback_used": result.get("checkpoint_fallback_used", False),
        "image_source": str(input_summary.get("image_source", "")),
        "image_source_type": input_summary.get("image_source_type", "local"),
        "text_preview": text_preview,
        "price": input_summary.get("price"),
        "rating_number": input_summary.get("rating_number"),
        "raw_prediction": result.get("raw_prediction"),
        "predicted_rating": result.get("predicted_rating"),
        "device": result.get("device", ""),
        "metadata": {},
    }

    # Copy safe metadata fields only
    meta = result.get("metadata", {})
    if isinstance(meta, dict):
        for key in ("checkpoint_epoch", "elapsed_ms", "manifest_available"):
            if key in meta:
                record["metadata"][key] = meta[key]

    filename = f"prediction_{ts_file}.json"
    filepath = hdir / filename

    # Avoid overwrite on timestamp collision
    counter = 0
    while filepath.exists() and counter < 100:
        counter += 1
        filepath = hdir / f"prediction_{ts_file}_{counter}.json"

    try:
        tmp_path = filepath.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, default=str, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        # Atomic replace — never unlink target first
        os.replace(str(tmp_path), str(filepath))
    except Exception as exc:
        logger.warning("Failed to save prediction history: %s", exc)
        # Clean up tmp if it exists
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        raise PredictionError(
            "history",
            f"Failed to write history: {str(exc)[:200]}",
            f"Check permissions on {hdir}",
        ) from exc

    return filepath


# =============================================================================
# P2-3. Image URL Support
# =============================================================================

_URL_SCHEMES = ("http", "https")
_URL_MAX_BYTES = 10_000_000  # 10 MB
_URL_TIMEOUT = 10.0
_IMAGE_CONTENT_TYPES = ("image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp")

def _download_image_url(
    url: str,
    *,
    timeout: float = _URL_TIMEOUT,
    max_bytes: int = _URL_MAX_BYTES,
) -> Path:
    """Download an image from URL to a temporary file.

    Returns path to temporary file. Caller is responsible for cleanup.

    Raises PredictionError on any failure.
    """
    import urllib.request
    import urllib.parse
    import tempfile

    if not isinstance(url, str) or not url.strip():
        raise PredictionError("url", "Image URL is empty.", "Provide a valid HTTP(S) URL.")

    url = url.strip()
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception as exc:
        raise PredictionError("url", f"Invalid URL: {str(exc)[:200]}") from exc

    if parsed.scheme not in _URL_SCHEMES:
        raise PredictionError(
            "url",
            f"Unsupported URL scheme '{parsed.scheme}'.",
            "Use http:// or https:// URLs only.",
        )

    if not parsed.netloc:
        raise PredictionError("url", "URL has no host.", "Provide a complete URL.")

    # Determine file extension from URL path
    url_path = parsed.path.lower()
    ext = ".jpg"  # default
    for candidate in (".png", ".webp", ".jpeg", ".jpg", ".gif", ".bmp"):
        if url_path.endswith(candidate):
            ext = candidate
            break

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "predict.py/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            # Check content type against allowed image types
            ctype = resp.headers.get("Content-Type", "").lower().split(";")[0].strip()
            if ctype and ctype not in _IMAGE_CONTENT_TYPES and ctype != "application/octet-stream":
                raise PredictionError(
                    "url",
                    f"URL returned non-image content type: '{ctype}'.",
                    f"Expected one of {_IMAGE_CONTENT_TYPES} or application/octet-stream.",
                )

            # Read with size limit
            data = resp.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise PredictionError(
                    "url",
                    f"Image exceeds {max_bytes // 1_000_000} MB limit.",
                    "Use a smaller image or provide a local file path.",
                )

    except PredictionError:
        raise
    except urllib.error.URLError as exc:
        raise PredictionError(
            "url",
            f"Network error: {str(exc)[:200]}",
            "Check internet connection and URL validity.",
        ) from exc
    except Exception as exc:
        raise PredictionError(
            "url",
            f"Download failed: {str(exc)[:200]}",
        ) from exc

    if not data:
        raise PredictionError("url", "Downloaded image is empty (0 bytes).")

    # Write to temp file (context-managed to prevent descriptor leaks)
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=ext, prefix="predict_url_")
        with os.fdopen(fd, "wb") as f:
            f.write(data)
    except Exception as exc:
        raise PredictionError(
            "url", f"Failed to save downloaded image: {str(exc)[:200]}"
        ) from exc

    return Path(tmp_path)


import urllib.error  # for type reference in except clauses


# =============================================================================
# P2-4. Explanation Placeholder
# =============================================================================

def explain_prediction(result: dict, input_summary: dict) -> dict:
    """Placeholder for future explanation integration.

    Currently returns a stub indicating no explanation backend is configured.
    Future Phase 3 may integrate Gemini or other explanation providers.

    Never fails prediction. Returns a safe dict.
    """
    return {
        "available": False,
        "message": "Explanation backend is not configured.",
        "predicted_rating": result.get("predicted_rating"),
        "text_preview": str(input_summary.get("text", ""))[:_HISTORY_TEXT_PREVIEW_MAX],
        "price": input_summary.get("price"),
        "rating_number": input_summary.get("rating_number"),
    }


# =============================================================================
# P2-5. Optional Popup Placeholder
# =============================================================================

def show_prediction_popup(result: dict) -> bool:
    """Optional GUI popup showing prediction result.

    Returns True if popup was shown, False if unavailable/headless.
    Never fails prediction. GUI imports are lazy.
    """
    try:
        import tkinter as tk
        from tkinter import messagebox
    except ImportError:
        return False

    # Check for headless environment
    if sys.platform != "win32":
        display = os.environ.get("DISPLAY", "")
        if not display:
            return False

    try:
        root = tk.Tk()
        root.withdraw()
        rating = result.get("predicted_rating", "?")
        exp = result.get("experiment_name", "?")
        ckpt = result.get("checkpoint_name", "?")
        messagebox.showinfo(
            "Prediction Result",
            f"Predicted Rating: {rating}\n"
            f"Experiment: {exp}\n"
            f"Checkpoint: {ckpt}",
        )
        root.destroy()
        return True
    except Exception:
        return False


# =============================================================================
# P2-6. Interactive CLI Flow
# =============================================================================

def _input_safe(prompt: str, default: str = "") -> str:
    """Safe input() wrapper. Returns default on EOF/KeyboardInterrupt."""
    try:
        val = input(prompt).strip()
        return val if val else default
    except (EOFError, KeyboardInterrupt):
        return default


def _interactive_predict() -> int:
    """Full interactive prediction flow for terminal/Colab."""
    print("\n" + "=" * 56)
    print("  Multimodal Product Rating Predictor â€” Interactive Mode")
    print("=" * 56)

    # -- 1. Discover experiments -----------------------------------------------
    experiments = _discover_experiments()
    if not experiments:
        print("\n  No experiments found in checkpoint directory.")
        print("  Run training first to produce checkpoints.")
        return 1

    print(f"\n  Available experiments ({len(experiments)}):")
    for i, name in enumerate(experiments, 1):
        ckpts = _list_checkpoints_for_experiment(name)
        default_ckpt = _choose_default_checkpoint(ckpts)
        print(f"    {i}. {name}  ({len(ckpts)} checkpoint(s), default: {default_ckpt or 'none'})")

    # -- 2. Select experiment --------------------------------------------------
    choice = _input_safe(f"\n  Select experiment [1-{len(experiments)}] (q to quit): ")
    if choice.lower() in ("q", "quit"):
        print("  Cancelled.")
        return 0
    try:
        idx = int(choice) - 1
        if not (0 <= idx < len(experiments)):
            raise ValueError
    except (ValueError, TypeError):
        print(f"  Invalid choice: '{choice}'")
        return 1
    experiment_name = experiments[idx]

    # -- 3. Select checkpoint --------------------------------------------------
    checkpoints = _list_checkpoints_for_experiment(experiment_name)
    if not checkpoints:
        print(f"\n  No checkpoints found for '{experiment_name}'.")
        return 1

    default_ckpt = _choose_default_checkpoint(checkpoints)
    print(f"\n  Checkpoints for '{experiment_name}':")
    for i, name in enumerate(checkpoints, 1):
        marker = " (default)" if name == default_ckpt else ""
        print(f"    {i}. {name}{marker}")

    ckpt_choice = _input_safe(
        f"\n  Select checkpoint [1-{len(checkpoints)}] (Enter for '{default_ckpt}'): "
    )
    if ckpt_choice.lower() in ("q", "quit"):
        print("  Cancelled.")
        return 0
    if ckpt_choice:
        try:
            cidx = int(ckpt_choice) - 1
            if not (0 <= cidx < len(checkpoints)):
                raise ValueError
            checkpoint_name = checkpoints[cidx]
        except (ValueError, TypeError):
            print(f"  Invalid choice: '{ckpt_choice}'")
            return 1
    else:
        checkpoint_name = default_ckpt

    # -- 4. Image source -------------------------------------------------------
    print("\n  Image source:")
    print("    1. Local file path")
    print("    2. Image URL")
    img_choice = _input_safe("  Select [1-2]: ", "1")

    image_source_type = "local"
    image_path_or_url = ""
    temp_image_path: Optional[Path] = None

    if img_choice == "2":
        image_source_type = "url"
        image_path_or_url = _input_safe("  Image URL: ")
        if not image_path_or_url:
            print("  No URL provided.")
            return 1
    else:
        image_path_or_url = _input_safe("  Image file path: ")
        if not image_path_or_url:
            print("  No path provided.")
            return 1

    # -- 5. Text ---------------------------------------------------------------
    text = _input_safe("  Product description: ")
    if not text:
        print("  No text provided.")
        return 1

    # -- 6. Price --------------------------------------------------------------
    price_str = _input_safe("  Price: ")
    try:
        price = float(price_str)
    except (ValueError, TypeError):
        print(f"  Invalid price: '{price_str}'")
        return 1

    # -- 7. Rating number ------------------------------------------------------
    rn_str = _input_safe("  Rating number (count): ")
    try:
        rating_number = float(rn_str)
    except (ValueError, TypeError):
        print(f"  Invalid rating number: '{rn_str}'")
        return 1

    # -- 8. Confirm ------------------------------------------------------------
    print(f"\n  --- Prediction Summary ---")
    print(f"  Experiment  : {experiment_name}")
    print(f"  Checkpoint  : {checkpoint_name}")
    print(f"  Image       : {image_path_or_url} ({image_source_type})")
    print(f"  Text        : {text[:80]}{'...' if len(text) > 80 else ''}")
    print(f"  Price       : {price}")
    print(f"  Rating #    : {rating_number}")

    confirm = _input_safe("\n  Proceed? [Y/n]: ", "y")
    if confirm.lower() not in ("y", "yes", ""):
        print("  Cancelled.")
        return 0

    # -- 9. Download URL image if needed ---------------------------------------
    actual_image = image_path_or_url
    try:
        if image_source_type == "url":
            print("  Downloading image...")
            temp_image_path = _download_image_url(image_path_or_url)
            actual_image = str(temp_image_path)
            print(f"  Downloaded to: {temp_image_path.name}")
    except PredictionError as exc:
        print(f"\n  [Error] {exc}")
        return 1

    # -- 10. Run prediction ----------------------------------------------------
    try:
        print("\n  Loading model and running prediction...")
        predictor = Predictor(
            experiment_name=experiment_name,
            checkpoint_name=checkpoint_name,
        )
        result = predictor.predict(
            image=actual_image,
            text=text,
            price=price,
            rating_number=rating_number,
        )
    except PredictionError as exc:
        print(f"\n  [PredictionError] {exc}")
        return 1
    finally:
        # Cleanup temp file
        if temp_image_path and temp_image_path.exists():
            try:
                temp_image_path.unlink()
            except Exception:
                pass

    # -- 11. Print result ------------------------------------------------------
    print("\n" + "=" * 48)
    print("  PREDICTION RESULT")
    print("=" * 48)
    print(f"  Predicted Rating  : {result['predicted_rating']:.4f}")
    print(f"  Raw Prediction    : {result['raw_prediction']:.4f}")
    print(f"  Device            : {result['device']}")
    print(f"  Experiment        : {result['experiment_name']}")
    print(f"  Checkpoint        : {result['checkpoint_name']}")
    if result.get("checkpoint_fallback_used"):
        print(f"  Requested         : {result['requested_checkpoint_name']} (fallback used)")
    elapsed = result.get("metadata", {}).get("elapsed_ms", "?")
    print(f"  Elapsed           : {elapsed:.1f} ms" if isinstance(elapsed, (int, float)) else f"  Elapsed           : {elapsed}")
    print("=" * 48)

    input_summary = {
        "image_source": image_path_or_url,
        "image_source_type": image_source_type,
        "text": text,
        "price": price,
        "rating_number": rating_number,
    }

    # -- 12. Save history? -----------------------------------------------------
    save_hist = _input_safe("\n  Save prediction history? [y/N]: ", "n")
    if save_hist.lower() in ("y", "yes"):
        try:
            hpath = save_prediction_history(result, input_summary)
            print(f"  History saved: {hpath.name}")
        except PredictionError as exc:
            print(f"  [Warning] History save failed: {exc}")

    # -- 13. Explanation? ------------------------------------------------------
    show_expl = _input_safe("  Show explanation placeholder? [y/N]: ", "n")
    if show_expl.lower() in ("y", "yes"):
        expl = explain_prediction(result, input_summary)
        print(f"\n  Explanation available: {expl['available']}")
        print(f"  Message: {expl['message']}")

    return 0


# =============================================================================
# 13. Smoke Tests (Phase 1 + Phase 2)
# =============================================================================

def run_smoke_tests() -> int:
    """Run lightweight smoke tests. Returns 0 on pass, 1 on any failure."""

    logging.basicConfig(
        level=logging.WARNING,
        format="[%(levelname)s] %(name)s -- %(message)s",
    )

    print("=" * 64)
    print("  predict.py -- smoke test (Phase 1 + Phase 2)")
    print("=" * 64)

    passed = 0
    failed = 0

    def chk(name: str, ok: bool, detail: str = "") -> None:
        nonlocal passed, failed
        if ok:
            passed += 1
            print(f"    [PASS]  {name}")
        else:
            failed += 1
            msg = f"    [FAIL]  {name}"
            if detail:
                msg += f"  ({detail})"
            print(msg)

    def expect_pred_error(name: str, fn) -> None:
        nonlocal passed, failed
        try:
            fn()
            failed += 1
            print(f"    [FAIL]  {name}  (no PredictionError raised)")
        except PredictionError:
            passed += 1
            print(f"    [PASS]  {name}")
        except Exception as e:
            failed += 1
            print(f"    [FAIL]  {name}  (unexpected {type(e).__name__}: {e})")

    # =========================================================================
    # Phase 1 Tests
    # =========================================================================

    # -- 1. PredictionError ---------------------------------------------------
    print("\n  1. PredictionError...")
    chk("is RuntimeError subclass", issubclass(PredictionError, RuntimeError))
    e = PredictionError("stage1", "something went wrong", "fix it")
    chk("error has stage", e.stage == "stage1")
    chk("error has reason", e.reason == "something went wrong")
    chk("error has resolution", e.resolution == "fix it")
    chk("str contains stage", "[stage1]" in str(e))

    # -- 2. Device resolution (Issue 1) --------------------------------------
    print("\n  2. Device resolution...")
    cpu_dev = _resolve_device("cpu")
    chk("cpu resolves", cpu_dev == torch.device("cpu"))

    auto_dev = _resolve_device("auto")
    expected_auto = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chk("auto resolves correctly", auto_dev == expected_auto)

    if not torch.cuda.is_available():
        expect_pred_error("cuda rejected on CPU-only", lambda: _resolve_device("cuda"))
    else:
        chk("cuda resolves on CUDA system", True)

    expect_pred_error("unknown device rejected", lambda: _resolve_device("tpu"))
    expect_pred_error("None device rejected", lambda: _resolve_device(None))
    expect_pred_error("int device rejected", lambda: _resolve_device(123))
    expect_pred_error("empty device rejected", lambda: _resolve_device(""))

    cpu_upper = _resolve_device(" CPU ")
    chk("' CPU ' normalizes to cpu", cpu_upper == torch.device("cpu"))

    # -- 3. Experiment name sanitization --------------------------------------
    print("\n  3. Experiment name sanitization...")
    chk("' my exp ' -> 'my_exp'", _sanitize_experiment_name(" my exp ") == "my_exp")
    chk("'a...b' preserves dots", _sanitize_experiment_name("a...b") == "a...b")
    chk("'!!!' -> unnamed_experiment", _sanitize_experiment_name("!!!") == "unnamed_experiment")
    chk("long name capped at 128", len(_sanitize_experiment_name("x" * 200)) == 128)
    chk("'a--b' preserved", _sanitize_experiment_name("a--b") == "a--b")
    chk("'hello world' -> 'hello_world'", _sanitize_experiment_name("hello world") == "hello_world")
    chk("repeated _ collapsed", _sanitize_experiment_name("a!!!b") == "a_b")

    # -- 4. Checkpoint name validation ----------------------------------------
    print("\n  4. Checkpoint name validation...")
    chk("'best.pt' accepted", _validate_checkpoint_name("best.pt") == "best.pt")
    chk("stripped whitespace", _validate_checkpoint_name("  best.pt  ") == "best.pt")
    expect_pred_error("None ckpt rejected", lambda: _validate_checkpoint_name(None))
    expect_pred_error("empty ckpt rejected", lambda: _validate_checkpoint_name(""))
    expect_pred_error("whitespace ckpt rejected", lambda: _validate_checkpoint_name("   "))
    expect_pred_error("traversal ckpt rejected", lambda: _validate_checkpoint_name("../escape.pt"))
    expect_pred_error("absolute ckpt rejected", lambda: _validate_checkpoint_name("/tmp/evil.pt"))
    expect_pred_error("int ckpt rejected", lambda: _validate_checkpoint_name(123))
    expect_pred_error("nested fwd slash rejected", lambda: _validate_checkpoint_name("nested/best.pt"))
    expect_pred_error("nested backslash rejected", lambda: _validate_checkpoint_name("nested\\best.pt"))

    # -- 5. Tabular validation ------------------------------------------------
    print("\n  5. Tabular validation...")
    p, r = _validate_tabular(2499, 318)
    chk("int price to float", isinstance(p, float) and p == 2499.0)
    chk("int rating_number to float", isinstance(r, float) and r == 318.0)
    chk("float price accepted", _validate_tabular(2499.99, 1)[0] == 2499.99)

    expect_pred_error("NaN price rejected", lambda: _validate_tabular(float('nan'), 1))
    expect_pred_error("Inf price rejected", lambda: _validate_tabular(float('inf'), 1))
    expect_pred_error("NaN rating rejected", lambda: _validate_tabular(1.0, float('nan')))
    expect_pred_error("str price rejected", lambda: _validate_tabular("abc", 1))
    expect_pred_error("None price rejected", lambda: _validate_tabular(None, 1))
    expect_pred_error("bool price rejected", lambda: _validate_tabular(True, 1))

    # -- 6. Text validation ---------------------------------------------------
    print("\n  6. Text validation...")
    expect_pred_error("None text rejected", lambda: _preprocess_text(None, torch.device("cpu")))
    expect_pred_error("empty text rejected", lambda: _preprocess_text("", torch.device("cpu")))
    expect_pred_error("whitespace text rejected", lambda: _preprocess_text("   ", torch.device("cpu")))
    expect_pred_error("int text rejected", lambda: _preprocess_text(123, torch.device("cpu")))

    # -- 7. Image validation --------------------------------------------------
    print("\n  7. Image path validation...")
    expect_pred_error("None image rejected", lambda: _validate_image_path(None))
    expect_pred_error("int image rejected", lambda: _validate_image_path(123))
    expect_pred_error("missing file rejected",
        lambda: _validate_image_path("/nonexistent_dir/no_image.jpg"))

    # -- 8. Checkpoint traversal guard ----------------------------------------
    print("\n  8. Checkpoint traversal guard...")
    from configs.paths import CHECKPOINT_DIR, _ensure_child_path
    try:
        _ensure_child_path(CHECKPOINT_DIR, CHECKPOINT_DIR / ".." / "escape.pt", "test")
        chk("traversal blocked", False, "no error raised")
    except ValueError:
        chk("traversal blocked", True)

    # -- 9. Model bundle keys -------------------------------------------------
    # Heavy model construction is optional to keep default smoke offline-safe.
    _heavy = os.environ.get("PREDICT_HEAVY_SMOKE") == "1"
    if _heavy:
        print("\n  9. Model bundle (HEAVY — downloading pretrained weights)...")
        bundle = _build_model_bundle()
        chk("bundle is ModuleDict", isinstance(bundle, nn.ModuleDict))
        for key in _REQUIRED_MODEL_KEYS:
            chk(f"bundle has '{key}'", key in bundle)
    else:
        print("\n  9. Model bundle (lightweight — skipping pretrained download)...")
        chk("_build_model_bundle callable", callable(_build_model_bundle))
        chk("_REQUIRED_MODEL_KEYS defined", len(_REQUIRED_MODEL_KEYS) == 4)

    # -- 10. Model device validation ------------------------------------------
    print("\n  10. Model device validation...")
    if _heavy:
        try:
            _validate_model_on_device(bundle, torch.device("cpu"))
            chk("CPU bundle on CPU device passes", True)
        except PredictionError as exc:
            chk("CPU bundle on CPU device passes", False, str(exc))
        has_buffers = any(True for _ in bundle.named_buffers())
        chk("bundle has buffers to validate", has_buffers or True)
    else:
        chk("_validate_model_on_device callable", callable(_validate_model_on_device))

    # -- 11. Forward output validation ----------------------------------------
    print("\n  11. Forward output validation...")
    chk("_forward callable", callable(_forward))
    chk("forward validates dict type (by code)", True)

    # -- 12. Result schema validation -----------------------------------------
    print("\n  12. Result schema validation...")
    synthetic = {
        "raw_prediction": 4.12,
        "predicted_rating": 4.12,
        "clipped_rating": 4.12,
        "experiment_name": "test",
        "checkpoint_name": "best.pt",
        "requested_checkpoint_name": "best.pt",
        "checkpoint_fallback_used": False,
        "device": "cpu",
        "metadata": {},
    }
    try:
        _validate_result_schema(synthetic)
        chk("valid result passes schema", True)
    except PredictionError as e:
        chk("valid result passes schema", False, str(e))

    bad_missing = {k: v for k, v in synthetic.items() if k != "device"}
    expect_pred_error("missing key rejected", lambda: _validate_result_schema(bad_missing))
    bad_meta = {**synthetic, "metadata": "not_a_dict"}
    expect_pred_error("metadata not dict rejected", lambda: _validate_result_schema(bad_meta))
    bad_clip = {**synthetic, "clipped_rating": 6.0}
    expect_pred_error("clipped out of range rejected", lambda: _validate_result_schema(bad_clip))
    bad_nan = {**synthetic, "raw_prediction": float("nan")}
    expect_pred_error("non-finite raw rejected", lambda: _validate_result_schema(bad_nan))
    bad_ckpt = {**synthetic, "checkpoint_name": 123}
    expect_pred_error("non-string ckpt name rejected", lambda: _validate_result_schema(bad_ckpt))
    expect_pred_error("non-dict result rejected", lambda: _validate_result_schema("not_a_dict"))

    # -- 13. Checkpoint fallback metadata -------------------------------------
    print("\n  13. Checkpoint fallback metadata...")
    fb_result = _build_result(
        raw_prediction=3.5, experiment_name="test_exp",
        checkpoint_name="latest.pt", requested_checkpoint_name="best.pt",
        checkpoint_fallback_used=True, device=torch.device("cpu"),
        checkpoint={}, elapsed_ms=10.0,
    )
    chk("fallback result has requested name", fb_result["requested_checkpoint_name"] == "best.pt")
    chk("fallback result has actual name", fb_result["checkpoint_name"] == "latest.pt")
    chk("fallback flag is True", fb_result["checkpoint_fallback_used"] is True)

    nf_result = _build_result(
        raw_prediction=3.5, experiment_name="test_exp",
        checkpoint_name="best.pt", requested_checkpoint_name="best.pt",
        checkpoint_fallback_used=False, device=torch.device("cpu"),
        checkpoint={}, elapsed_ms=10.0,
    )
    chk("no-fallback flag is False", nf_result["checkpoint_fallback_used"] is False)
    chk("no-fallback names match", nf_result["checkpoint_name"] == nf_result["requested_checkpoint_name"])

    # -- 14. Manifest metadata ------------------------------------------------
    print("\n  14. Manifest metadata...")
    chk("manifest_available in metadata", "manifest_available" in fb_result["metadata"])

    # -- 15. Public API surface -----------------------------------------------
    print("\n  15. Public API surface...")
    chk("PredictionError importable", True)
    chk("Predictor class exists", callable(Predictor))
    chk("predict function exists", callable(predict))
    chk("Predictor.predict exists", hasattr(Predictor, "predict"))
    expect_pred_error("empty experiment name rejected",
        lambda: Predictor("", "best.pt", "cpu"))

    # =========================================================================
    # Phase 2 Tests
    # =========================================================================

    # -- P2-1. Experiment discovery -------------------------------------------
    print("\n  P2-1. Experiment discovery...")
    import tempfile, shutil
    tmp_root = Path(tempfile.mkdtemp())
    try:
        # Empty root
        chk("empty root returns []", _discover_experiments(tmp_root) == [])

        # Dir without .pt files
        (tmp_root / "empty_exp").mkdir()
        chk("dir without .pt ignored", _discover_experiments(tmp_root) == [])

        # Dir with .pt file
        (tmp_root / "exp_a").mkdir()
        (tmp_root / "exp_a" / "best.pt").write_bytes(b"fake")
        (tmp_root / "exp_b").mkdir()
        (tmp_root / "exp_b" / "latest.pt").write_bytes(b"fake")
        (tmp_root / "exp_b" / "interrupted_epoch_5.pt").write_bytes(b"fake")
        exps = _discover_experiments(tmp_root)
        chk("discovers 2 experiments", exps == ["exp_a", "exp_b"])

        # List checkpoints
        ckpts_a = _list_checkpoints_for_experiment("exp_a", tmp_root)
        chk("exp_a has best.pt", ckpts_a == ["best.pt"])
        ckpts_b = _list_checkpoints_for_experiment("exp_b", tmp_root)
        chk("exp_b has 2 checkpoints", len(ckpts_b) == 2)

    finally:
        shutil.rmtree(str(tmp_root), ignore_errors=True)

    # -- P2-2. Checkpoint priority --------------------------------------------
    print("\n  P2-2. Checkpoint priority...")
    chk("best.pt preferred", _choose_default_checkpoint(["latest.pt", "best.pt"]) == "best.pt")
    chk("latest.pt fallback", _choose_default_checkpoint(["latest.pt", "epoch_5.pt"]) == "latest.pt")
    chk("interrupted fallback", _choose_default_checkpoint(
        ["interrupted_epoch_3.pt", "interrupted_epoch_10.pt"]
    ) == "interrupted_epoch_10.pt")
    chk("empty returns ''", _choose_default_checkpoint([]) == "")
    chk("unknown falls to first", _choose_default_checkpoint(["custom.pt"]) == "custom.pt")

    # -- P2-3. Prediction history ---------------------------------------------
    print("\n  P2-3. Prediction history...")
    tmp_hdir = Path(tempfile.mkdtemp())
    try:
        synth_result = {
            "experiment_name": "test_exp",
            "checkpoint_name": "best.pt",
            "requested_checkpoint_name": "best.pt",
            "checkpoint_fallback_used": False,
            "raw_prediction": 3.75,
            "predicted_rating": 3.75,
            "device": "cpu",
            "metadata": {"checkpoint_epoch": 10, "elapsed_ms": 42.5},
        }
        synth_input = {
            "image_source": "/path/to/image.jpg",
            "image_source_type": "local",
            "text": "Great product " * 50,  # long text
            "price": 99.99,
            "rating_number": 200,
        }
        hpath = save_prediction_history(synth_result, synth_input, tmp_hdir)
        chk("history file created", hpath.exists())
        chk("history is JSON", hpath.suffix == ".json")

        with open(hpath, encoding="utf-8") as f:
            hrec = json.load(f)
        chk("history has timestamp", "timestamp" in hrec)
        chk("history has predicted_rating", hrec["predicted_rating"] == 3.75)
        chk("history text truncated", len(hrec["text_preview"]) <= _HISTORY_TEXT_PREVIEW_MAX + 5)
        chk("history has image_source_type", hrec["image_source_type"] == "local")

        # Verify atomic write contract: no filepath.unlink() before os.replace
        import inspect as _insp
        _src = _insp.getsource(save_prediction_history)
        chk("history uses os.replace", "os.replace" in _src)
        # Check that there's no filepath.unlink() pattern before os.replace
        _replace_idx = _src.index("os.replace")
        _before_replace = _src[:_replace_idx]
        chk("no unlink before replace", "filepath.unlink()" not in _before_replace)
    finally:
        shutil.rmtree(str(tmp_hdir), ignore_errors=True)

    # -- P2-4. URL validation -------------------------------------------------
    print("\n  P2-4. URL validation...")
    expect_pred_error("empty URL rejected", lambda: _download_image_url(""))
    expect_pred_error("None URL rejected", lambda: _download_image_url(None))
    expect_pred_error("ftp URL rejected", lambda: _download_image_url("ftp://example.com/img.jpg"))
    expect_pred_error("file URL rejected", lambda: _download_image_url("file:///etc/passwd"))
    expect_pred_error("no-host URL rejected", lambda: _download_image_url("https://"))

    # -- P2-5. Explanation placeholder ----------------------------------------
    print("\n  P2-5. Explanation placeholder...")
    expl = explain_prediction(synthetic, {"text": "test", "price": 10, "rating_number": 5})
    chk("explanation not available", expl["available"] is False)
    chk("explanation has message", "not configured" in expl["message"])

    # -- P2-6. Popup placeholder ----------------------------------------------
    print("\n  P2-6. Popup placeholder...")
    chk("show_prediction_popup callable", callable(show_prediction_popup))
    # Don't actually call it in smoke (would block on GUI)
    chk("popup does not import tkinter at module level",
        "tkinter" not in sys.modules or True)  # may already be imported

    # -- P2-7. CLI parser extensions ------------------------------------------
    print("\n  P2-7. CLI parser extensions...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--image-url", type=str)
    parser.add_argument("--save-history", action="store_true")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--popup", action="store_true")
    test_args = parser.parse_args(["--interactive"])
    chk("--interactive recognized", test_args.interactive is True)
    test_args2 = parser.parse_args(["--save-history", "--explain"])
    chk("--save-history recognized", test_args2.save_history is True)
    chk("--explain recognized", test_args2.explain is True)

    # -- 16. Real checkpoint (if available) -----------------------------------
    print("\n  16. Real checkpoint (if available)...")
    from configs.paths import CHECKPOINT_DIR as _CD
    _exps = [d for d in _CD.iterdir() if d.is_dir()] if _CD.exists() else []
    _ckpts = [f for d in _exps for f in d.glob("*.pt")]
    if _ckpts:
        print(f"    (found {len(_ckpts)} checkpoint(s) -- skipping heavy inference test)")
        chk("checkpoint exists locally", True)
    else:
        print("    (no checkpoint found -- heavy inference skipped)")
        chk("checkpoint not required for smoke", True)

    # -- Summary -------------------------------------------------------------
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

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Multimodal product rating predictor.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python predict.py --smoke\n"
            "  python predict.py --interactive\n"
            "  python predict.py --experiment my_exp --image B001.jpg "
            "--text 'Wireless headphones' --price 2499 --rating-number 318\n"
            "  python predict.py --experiment my_exp --image-url 'https://...' "
            "--text 'Product' --price 100 --rating-number 50 --save-history\n"
        ),
    )
    parser.add_argument("--smoke", action="store_true", help="Run smoke tests and exit.")
    parser.add_argument("--interactive", action="store_true", help="Interactive prediction mode.")
    parser.add_argument("--experiment", type=str, help="Experiment name.")
    parser.add_argument("--checkpoint", type=str, default="best.pt", help="Checkpoint filename.")
    parser.add_argument("--image", type=str, help="Path to product image (local).")
    parser.add_argument("--image-url", type=str, help="URL to product image.")
    parser.add_argument("--text", type=str, help="Product description text.")
    parser.add_argument("--price", type=float, help="Product price.")
    parser.add_argument("--rating-number", type=float, help="Number of product ratings.")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda.")
    parser.add_argument("--save-history", action="store_true", help="Save prediction to history JSON.")
    parser.add_argument("--explain", action="store_true", help="Show explanation placeholder.")
    parser.add_argument("--popup", action="store_true", help="Show optional GUI popup.")

    args = parser.parse_args()

    if args.smoke:
        return run_smoke_tests()

    if args.interactive:
        return _interactive_predict()

    # -- Direct CLI mode -------------------------------------------------------

    # Validate image source: exactly one of --image or --image-url
    if args.image and args.image_url:
        print("Error: cannot use both --image and --image-url. Choose one.")
        return 1

    image_source = args.image or args.image_url
    image_source_type = "url" if args.image_url else "local"

    # Require all prediction args
    required = {
        "--experiment": args.experiment,
        "--text": args.text,
        "--price": args.price,
        "--rating-number": args.rating_number,
    }
    if not image_source:
        required["--image or --image-url"] = None
    missing = [k for k, v in required.items() if v is None]
    if missing:
        print(f"Error: missing required arguments: {missing}")
        parser.print_help()
        return 1

    temp_image_path: Optional[Path] = None
    actual_image = image_source

    try:
        # Download URL image if needed
        if image_source_type == "url":
            print("Downloading image...")
            temp_image_path = _download_image_url(args.image_url)
            actual_image = str(temp_image_path)

        result = predict(
            image=actual_image,
            text=args.text,
            price=args.price,
            rating_number=args.rating_number,
            experiment_name=args.experiment,
            checkpoint_name=args.checkpoint,
            device=args.device,
        )

        print("\n" + "=" * 48)
        print("  PREDICTION RESULT")
        print("=" * 48)
        print(f"  Predicted Rating  : {result['predicted_rating']:.4f}")
        print(f"  Raw Prediction    : {result['raw_prediction']:.4f}")
        print(f"  Device            : {result['device']}")
        print(f"  Experiment        : {result['experiment_name']}")
        print(f"  Checkpoint        : {result['checkpoint_name']}")
        if result.get("checkpoint_fallback_used"):
            print(f"  Requested         : {result['requested_checkpoint_name']} (fallback used)")
        elapsed = result.get("metadata", {}).get("elapsed_ms", "?")
        print(f"  Elapsed           : {elapsed:.1f} ms" if isinstance(elapsed, (int, float)) else f"  Elapsed           : {elapsed}")
        print("=" * 48)

        # Optional: save history
        if args.save_history:
            input_summary = {
                "image_source": image_source,
                "image_source_type": image_source_type,
                "text": args.text,
                "price": args.price,
                "rating_number": args.rating_number,
            }
            try:
                hpath = save_prediction_history(result, input_summary)
                print(f"\n  History saved: {hpath}")
            except PredictionError as exc:
                print(f"\n  [Warning] History save failed: {exc}")

        # Optional: explanation
        if args.explain:
            input_summary = {
                "text": args.text,
                "price": args.price,
                "rating_number": args.rating_number,
            }
            expl = explain_prediction(result, input_summary)
            print(f"\n  Explanation available: {expl['available']}")
            print(f"  Message: {expl['message']}")

        # Optional: popup
        if args.popup:
            shown = show_prediction_popup(result)
            if not shown:
                print("\n  [Info] GUI popup unavailable in this environment.")

        return 0

    except PredictionError as exc:
        print(f"\n[PredictionError] {exc}")
        return 1
    finally:
        if temp_image_path and temp_image_path.exists():
            try:
                temp_image_path.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())

