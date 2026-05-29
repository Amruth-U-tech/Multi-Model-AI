# =============================================================================
# image_encoder.py
# Visual Representation Encoder — Multimodal AI Pipeline
# =============================================================================
#
# Responsibilities:
#   - Preprocess and augment ecommerce product images
#   - Safely load images with full edge case tolerance
#   - Generate stable latent embeddings via ConvNeXt Tiny backbone
#   - Compress visual features through a configurable projection head
#   - Return fusion-ready latent vectors for multimodal training
#
# This file is:
#   - encoder-only         (no training logic)
#   - device-agnostic      (GPU transfer belongs in train.py / inference.py)
#   - training-independent (no loss, optimizer, or scheduler)
#   - fusion-independent   (latent vectors are modality-agnostic)
#
# Module dependency order (critical for import-time safety):
#   imports → constants → dataclass config → transforms → utilities →
#   projection head → encoder → factory → smoke test
#
# Compatible with:
#   - torch.utils.data.DataLoader pipelines
#   - CUDA / CPU execution
#   - FP16 mixed precision (enabled externally in train.py)
#   - Tesla T4 / Colab execution
#   - Future ViT / SwinTransformer backbone swaps
# =============================================================================

import os
import sys
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

# ─────────────────────────────────────────────────────────────
# Google Colab + Project Import Safety
# ─────────────────────────────────────────────────────────────
PROJECT_PATH = Path("/content/drive/MyDrive/multi-model-ai")
if str(PROJECT_PATH) not in sys.path:
    sys.path.append(str(PROJECT_PATH))

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageFile
import timm

# ── Allow PIL to load truncated images safely ─────────────────────────────────
ImageFile.LOAD_TRUNCATED_IMAGES = True

# =============================================================================
# Logging
# =============================================================================

# Module-scoped logger — NO basicConfig here.
# Logging must be configured ONLY in top-level entry points (train.py / inference.py).
# This prevents handler conflicts and duplicate output across multimodal modules.
logger = logging.getLogger(__name__)

# =============================================================================
# Global Constants
# =============================================================================
# Defined FIRST — ImageEncoderConfig dataclass defaults reference these at
# class-body evaluation time (import-time). Any constant used as a dataclass
# default field MUST exist before the @dataclass decorator is reached.

# ── ImageNet normalization — must match ConvNeXt pretraining ──────────────────
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD : Tuple[float, float, float] = (0.229, 0.224, 0.225)

# ── ConvNeXt Tiny feature dimension (timm, num_classes=0) ────────────────────
# Default reference for documentation and smoke-test fallback.
# At runtime, ImageEncoder.__init__ queries backbone.num_features dynamically
# to support backbone swaps (ViT, SwinTransformer, etc.) without code changes.
CONVNEXT_FEATURE_DIM: int = 768

# ── Preprocessing geometry ────────────────────────────────────────────────────
# Resize short edge to RESIZE_SIZE → CenterCrop to INPUT_SIZE.
# Preserves product aspect ratio; avoids geometric distortion.
INPUT_SIZE  : Tuple[int, int] = (224, 224)
RESIZE_SIZE : int             = 256

# ── Latent space defaults ─────────────────────────────────────────────────────
DEFAULT_LATENT_DIM : int   = 512
DEFAULT_HIDDEN_DIM : int   = 512
DEFAULT_DROPOUT    : float = 0.2

# =============================================================================
# ImageEncoderConfig — Structured Configuration
# =============================================================================
# Placed AFTER constants so all default values resolve at import time.
# Placed BEFORE encoder class so ImageEncoder.__init__ can type-hint it.

@dataclass
class ImageEncoderConfig:
    """
    Single source of truth for all ImageEncoder hyperparameters.

    Design rationale:
      - One config object replaces a growing list of constructor arguments
      - Inspectable, loggable, and serializable (dataclasses.asdict())
      - Future-compatible with YAML / Hydra / argparse config systems
      - Follows the same pattern as TextEncoderConfig, TabularEncoderConfig,
        FusionConfig, and TrainingConfig for consistent multimodal orchestration

    Integration:
        config  = ImageEncoderConfig(latent_dim=256)
        encoder = ImageEncoder(config)

    Future centralization (no encoder changes required):
        configs/image_encoder_config.py  ← move dataclass here
        configs/multimodal_config.py     ← umbrella config for all encoders
    """

    # ── Backbone ──────────────────────────────────────────────────────────────
    backbone_name       : str            = "convnext_tiny"
    # timm model string — change here to swap backbone without touching encoder internals
    pretrained          : bool           = True
    # Always True in production; False only for architecture unit tests

    # ── Latent Space ──────────────────────────────────────────────────────────
    latent_dim          : int            = DEFAULT_LATENT_DIM
    # Output embedding size — must match FusionConfig.image_dim downstream
    hidden_dim          : int            = DEFAULT_HIDDEN_DIM
    # Projection head bottleneck: Linear(CONVNEXT_FEATURE_DIM → hidden_dim → latent_dim)

    # ── Regularization ────────────────────────────────────────────────────────
    dropout             : float          = DEFAULT_DROPOUT
    # Projection head dropout — increase if overfitting on small datasets

    # ── Embedding Geometry ────────────────────────────────────────────────────
    normalize_embeddings: bool           = True
    # L2-normalize output to unit sphere
    # Required for cosine retrieval, contrastive loss, multimodal alignment

    # ── Training Control ──────────────────────────────────────────────────────
    freeze_backbone     : bool           = True
    # Freeze ConvNeXt stages 0-2; keep stage 3 + projection trainable
    # Reduces VRAM; prevents early overfitting; call unfreeze_backbone() later

    # ── Preprocessing Geometry ────────────────────────────────────────────────
    input_size          : Tuple[int,int] = INPUT_SIZE
    # Final spatial resolution into ConvNeXt — standard (224, 224)
    resize_size         : int            = RESIZE_SIZE
    # Short-edge resize before CenterCrop — preserves product aspect ratio

# =============================================================================
# Transform Pipelines
# =============================================================================
# Transforms are functions, not classes — defined before encoder so Dataset
# classes can import them without importing the full encoder module if needed.

def build_train_transforms(
    input_size  : Tuple[int, int] = INPUT_SIZE,
    resize_size : int             = RESIZE_SIZE,
) -> transforms.Compose:
    """
    Lightweight ecommerce-safe augmentation pipeline for training.

    Augmentations chosen to:
      - Preserve product identity, shape, and category semantics
      - Improve generalization across lighting and orientation variants

    Explicitly excluded:
      - RandomResizedCrop : destroys product boundary completeness
      - Perspective/Shear : distorts product geometry
      - Vertical flip     : unnatural for ecommerce images
      - Strong color drop : destroys texture and material signal

    Args:
        input_size  : Final (H, W) after CenterCrop. Default (224, 224).
        resize_size : Short-edge resize before crop. Default 256.

    Returns:
        torchvision.transforms.Compose
    """
    return transforms.Compose([
        # Aspect-ratio-preserving resize then deterministic center crop.
        # BICUBIC is explicit — prevents backend-dependent interpolation drift.
        transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,           # minimal — product colors are semantically meaningful
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_eval_transforms(
    input_size  : Tuple[int, int] = INPUT_SIZE,
    resize_size : int             = RESIZE_SIZE,
) -> transforms.Compose:
    """
    Deterministic transform pipeline for evaluation, inference, and
    embedding extraction. No random operations — guarantees reproducible
    latent vectors across runs.

    Args:
        input_size  : Final (H, W) after CenterCrop. Default (224, 224).
        resize_size : Short-edge resize before crop. Default 256.

    Returns:
        torchvision.transforms.Compose
    """
    return transforms.Compose([
        transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_transforms(
    config : Optional[ImageEncoderConfig] = None,
    mode   : str                          = "eval",
) -> transforms.Compose:
    """
    Config-driven convenience accessor for Dataset classes and inference scripts.

    Routing geometry through config ensures that changing config.input_size or
    config.resize_size automatically propagates to the transform pipeline —
    no scattered constant updates required across files.

    Args:
        config : ImageEncoderConfig instance. Falls back to module-level
                 defaults (INPUT_SIZE, RESIZE_SIZE) if None — preserves
                 backward compatibility for callers that don't pass a config.
        mode   : "train"                    → augmented pipeline
                 "eval"/"inference"/"test"  → deterministic pipeline

    Returns:
        torchvision.transforms.Compose

    Raises:
        ValueError : If mode string is unrecognized.

    Usage:
        # Config-driven (preferred — geometry is fully orchestrated):
        config    = ImageEncoderConfig(input_size=(336,336), resize_size=384)
        transform = get_transforms(config, mode="train")

        # Backward-compatible (uses module defaults):
        transform = get_transforms(mode="eval")
    """
    input_size  = config.input_size  if config is not None else INPUT_SIZE
    resize_size = config.resize_size if config is not None else RESIZE_SIZE

    mode = mode.strip().lower()
    if mode == "train":
        return build_train_transforms(input_size=input_size, resize_size=resize_size)
    elif mode in ("eval", "inference", "test"):
        return build_eval_transforms(input_size=input_size, resize_size=resize_size)
    else:
        raise ValueError(
            f"get_transforms(): unrecognized mode '{mode}'. "
            f"Expected one of: 'train', 'eval', 'inference', 'test'."
        )

# =============================================================================
# Safe Image Loader
# =============================================================================

def safe_load_image(
    image_path : Optional[str],
    input_size : Tuple[int, int] = INPUT_SIZE,
) -> Image.Image:
    """
    Safely loads a single PIL image with comprehensive edge case handling.

    Handles:
      - None / NaN / non-string paths           (Edge Case 1)
      - Missing files on disk                   (Edge Case 2)
      - Corrupted / unreadable files            (Edge Case 3)
      - Grayscale (L mode) images               (Edge Case 4)
      - RGBA / palette images                   (Edge Case 5)
      - Tiny thumbnails                         (Edge Case 6 — resize handles)
      - Truncated files (LOAD_TRUNCATED_IMAGES) (Edge Case 3 extension)

    On any failure:
      - Logs a warning (never raises)
      - Returns a neutral gray fallback image
      - DataLoader pipeline continues uninterrupted

    Fallback color (127, 127, 127):
      After ImageNet normalization this maps to ≈ (−0.02, −0.07, −0.14) —
      near the dataset mean. White (255,255,255) normalizes to ≈ (2.6, 2.4, 2.2),
      far outside the natural image distribution, polluting retrieval indices
      and latent clustering.

    Args:
        image_path : Absolute path to image file.
        input_size : Fallback image size. Actual resize is in transform pipeline.

    Returns:
        PIL.Image.Image in RGB mode — never None, never raises.
    """
    def _fallback(reason: str) -> Image.Image:
        logger.warning(f"Image fallback | reason={reason} | path={image_path}")
        return Image.new("RGB", input_size, color=(127, 127, 127))

    # ── Edge Case 1: Invalid path ─────────────────────────────────────────────
    if not image_path or not isinstance(image_path, str) or not image_path.strip():
        return _fallback("invalid or missing path")

    image_path = image_path.strip()

    # ── Edge Case 2: File not on disk ─────────────────────────────────────────
    if not os.path.exists(image_path):
        return _fallback("file not found on disk")

    # ── Edge Case 3: Corrupted / truncated file ───────────────────────────────
    try:
        img = Image.open(image_path)
        img.verify()            # integrity check — does not fully decode
        img = Image.open(image_path)    # must re-open after verify() — PIL requirement
    except Exception as exc:
        return _fallback(f"PIL open/verify failed: {exc}")

    # ── Edge Cases 4 + 5: Grayscale, RGBA, palette → force RGB ───────────────
    # .convert("RGB") handles: L, P, RGBA, LA, CMYK → consistent 3-channel tensor
    try:
        img = img.convert("RGB")
    except Exception as exc:
        return _fallback(f"RGB conversion failed: {exc}")

    return img

# =============================================================================
# Projection Head
# =============================================================================

class ProjectionHead(nn.Module):
    """
    Two-layer MLP that compresses backbone features into compact latent space.

    Architecture:
        Linear(in_dim → hidden_dim) → GELU → Dropout → Linear(hidden_dim → latent_dim)

    Design rationale:
      - GELU: smoother gradients than ReLU; standard for modern vision encoders
      - Dropout: regularization + prevents latent collapse on small datasets
      - No BatchNorm: stable at any batch size including batch=1 at inference
      - Two layers: sufficient capacity without overfitting small ecommerce data

    Args:
        in_dim     : Input dimension (CONVNEXT_FEATURE_DIM = 768).
        hidden_dim : Bottleneck dimension.
        latent_dim : Output embedding dimension.
        dropout    : Dropout probability.
    """

    def __init__(
        self,
        in_dim    : int   = CONVNEXT_FEATURE_DIM,
        hidden_dim: int   = DEFAULT_HIDDEN_DIM,
        latent_dim: int   = DEFAULT_LATENT_DIM,
        dropout   : float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,     hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# =============================================================================
# ImageEncoder — Main Module
# =============================================================================

class ImageEncoder(nn.Module):
    """
    Reusable visual representation encoder for multimodal learning.

    Architecture:
        Input Tensor  (B, 3, 224, 224)
             ↓
        ConvNeXt Tiny pretrained backbone
             ↓
        Deep feature vector  (B, 768)
             ↓
        ProjectionHead: Linear(768→512) → GELU → Dropout → Linear(512→512)
             ↓
        Latent Embedding  (B, latent_dim)
             ↓
        Optional L2 Normalization → unit-sphere embedding

    Properties:
      - device-agnostic  : no .cuda() / .to() calls here
      - training-free    : no loss, optimizer, or scheduler
      - fusion-agnostic  : plain float tensor output
      - SHAP-compatible  : linear projection head supports gradient attribution
      - backbone-swappable: change config.backbone_name to swap ConvNeXt → ViT

    Args:
        config : ImageEncoderConfig instance. If None, uses all defaults.
    """

    def __init__(self, config: Optional[ImageEncoderConfig] = None) -> None:
        super().__init__()

        # ── Resolve config safely — avoids mutable default argument bug ───────
        # Never use `config: ImageEncoderConfig = ImageEncoderConfig()` as a
        # default argument — Python evaluates that object ONCE at definition time,
        # creating shared state across all callers that don't pass a config.
        if config is None:
            config = ImageEncoderConfig()

        # ── Store for external inspection (train.py / logging / serialization) ─
        self.config     = config
        self.latent_dim = config.latent_dim
        self.normalize  = config.normalize_embeddings

        # ── Backbone ──────────────────────────────────────────────────────────
        # num_classes=0 strips the classifier head → outputs raw (B, 768) features
        logger.info(
            f"Loading backbone: '{config.backbone_name}' "
            f"(pretrained={config.pretrained}, num_classes=0)"
        )
        self.backbone = timm.create_model(
            config.backbone_name,
            pretrained  = config.pretrained,
            num_classes = 0,
        )

        # ── Dynamic feature dimension detection ──────────────────────────────
        # Queries the actual backbone output dim instead of assuming 768.
        # This makes backbone swaps (ViT, Swin, EfficientNet) work without
        # code changes — only config.backbone_name needs to change.
        if not hasattr(self.backbone, 'num_features'):
            raise RuntimeError(
                f"Backbone '{config.backbone_name}' does not expose 'num_features'. "
                f"Cannot determine projection head input dimension. "
                f"Verify the timm model supports num_classes=0 feature extraction."
            )
        backbone_feature_dim = self.backbone.num_features
        self.backbone_feature_dim = backbone_feature_dim
        logger.info(
            f"Backbone loaded | model={config.backbone_name} | "
            f"feature_dim={backbone_feature_dim}"
        )

        # ── Selective backbone freezing ───────────────────────────────────────
        if config.freeze_backbone:
            self._freeze_backbone()

        # ── Projection head ───────────────────────────────────────────────────
        self.projection = ProjectionHead(
            in_dim     = backbone_feature_dim,
            hidden_dim = config.hidden_dim,
            latent_dim = config.latent_dim,
            dropout    = config.dropout,
        )

        logger.info(
            f"ImageEncoder ready | latent_dim={config.latent_dim} | "
            f"normalize={config.normalize_embeddings} | "
            f"freeze_backbone={config.freeze_backbone} | "
            f"trainable_params={self._count_trainable_params():,}"
        )

    # =========================================================================
    # Backbone Freezing
    # =========================================================================

    def _freeze_backbone(self) -> None:
        """
        Freezes ConvNeXt stages 0-2; selectively unfreezes stage 3.

        Rationale:
          - Stages 0-2 encode low-level priors (edges, textures, patterns)
            already well-learned from ImageNet — no benefit retraining them
          - Stage 3 encodes semantic concepts (product silhouettes, categories)
            that benefit from domain adaptation to ecommerce images
          - Projection head is always fully trainable

        VRAM savings:
          Frozen parameters skip gradient computation entirely —
          critical for batch_size=16 on Tesla T4 / Colab.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

        unfrozen = 0
        if hasattr(self.backbone, "stages"):
            for param in self.backbone.stages[-1].parameters():
                param.requires_grad = True
                unfrozen += 1
            logger.info(
                f"Backbone frozen | unfrozen: final stage "
                f"({unfrozen} parameter tensors remain trainable)"
            )
        else:
            logger.warning(
                "backbone.stages not found — backbone fully frozen. "
                "Override _freeze_backbone() for non-ConvNeXt architectures."
            )

    def unfreeze_backbone(self, stages: Optional[int] = None) -> None:
        """
        Progressively unfreezes backbone stages for staged fine-tuning.
        Call from train.py after warm-up phase converges.

        Args:
            stages : Trailing stages to unfreeze (from deepest). None = all.

        Example:
            encoder.unfreeze_backbone(stages=2)  # open last 2 stages
            encoder.unfreeze_backbone()           # open full backbone
        """
        if stages is None:
            for param in self.backbone.parameters():
                param.requires_grad = True
            logger.info("Full backbone unfrozen for fine-tuning.")
        elif hasattr(self.backbone, "stages"):
            for stage in self.backbone.stages[-stages:]:
                for param in stage.parameters():
                    param.requires_grad = True
            logger.info(f"Unfroze last {stages} backbone stage(s).")
        else:
            logger.warning("backbone.stages not found — cannot selectively unfreeze.")

    def freeze_backbone(self) -> None:
        """Re-freezes entire backbone. Use when switching training phases."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone fully re-frozen.")

    # =========================================================================
    # Utility
    # =========================================================================

    def _count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_embedding_dim(self) -> int:
        """Returns latent_dim — used by FusionModel to validate input contracts."""
        return self.latent_dim

    # =========================================================================
    # Forward Pass
    # =========================================================================

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encodes a batch of preprocessed images into latent embeddings.

        Args:
            images : Float tensor (B, 3, 224, 224) on the same device as model.
                     Must be normalized with ImageNet mean/std via build_*_transforms().

        Returns:
            embeddings : Float tensor (B, latent_dim).
                         L2-normalized to unit sphere if config.normalize_embeddings=True.

        Raises:
            ValueError : If input is not 4D or channels ≠ 3.
        """
        # ── Validate tensor dimensionality (Edge Case 7) ──────────────────────
        if images.ndim != 4:
            raise ValueError(
                f"ImageEncoder.forward() expected 4D tensor (B,C,H,W), "
                f"got {tuple(images.shape)}"
            )

        # ── Validate channel count (Edge Case 8) ──────────────────────────────
        if images.shape[1] != 3:
            raise ValueError(
                f"ImageEncoder.forward() expected 3 channels (RGB), "
                f"got {images.shape[1]}. Verify safe_load_image() is in use."
            )

        # ── ConvNeXt feature extraction → (B, 768) ────────────────────────────
        features   = self.backbone(images)

        # ── Projection → (B, latent_dim) ──────────────────────────────────────
        embeddings = self.projection(features)

        # ── Optional L2 normalization → unit sphere ───────────────────────────
        # Required for cosine retrieval, contrastive loss, multimodal alignment
        if self.normalize:
            embeddings = nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

# =============================================================================
# Factory Function
# =============================================================================

def build_encoder(config: Optional[ImageEncoderConfig] = None) -> ImageEncoder:
    """
    Clean factory entry point for train.py, inference.py, and notebooks.
    Follows the same pattern as build_text_encoder(), build_tabular_encoder().

    The None default is intentional — avoids the mutable default argument trap.
    A fresh ImageEncoderConfig() is instantiated inside ImageEncoder.__init__
    if no config is passed.

    Usage:
        from image_encoder import build_encoder, get_transforms, safe_load_image
        from image_encoder import ImageEncoderConfig

        config    = ImageEncoderConfig(latent_dim=256, freeze_backbone=True)
        encoder   = build_encoder(config)
        transform = get_transforms(config, mode="train")   # geometry from config

        image  = safe_load_image("/path/to/B001J63LJQ.jpg")
        tensor = transform(image).unsqueeze(0)          # (1, 3, 224, 224)

        encoder.to(device)
        with torch.no_grad():
            emb = encoder(tensor.to(device))            # (1, 256)

    Args:
        config : ImageEncoderConfig or None (defaults applied internally).

    Returns:
        ImageEncoder on CPU — caller is responsible for .to(device).
    """
    return ImageEncoder(config)

# =============================================================================
# Smoke Test  —  python image_encoder.py
# =============================================================================

if __name__ == "__main__":

    # ── Configure logging for smoke test only ─────────────────────────────────
    # In production this lives in train.py / inference.py, NOT in module scope.
    logging.basicConfig(
        level  = logging.INFO,
        format = "[%(asctime)s] [%(levelname)s] %(name)s — %(message)s",
        datefmt= "%H:%M:%S",
    )

    logger.info("=" * 60)
    logger.info("  image_encoder.py — smoke test")
    logger.info("=" * 60)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {device}")

        # ── Config construction ───────────────────────────────────────────────────
        config  = ImageEncoderConfig(latent_dim=512, freeze_backbone=True)
        encoder = build_encoder(config)
        encoder.to(device)
        encoder.eval()

        # ── safe_load_image edge cases ────────────────────────────────────────────
        logger.info("Testing safe_load_image() edge cases...")
        assert safe_load_image(None).size  == INPUT_SIZE   # Edge Case 1 — None
        assert safe_load_image("").size    == INPUT_SIZE   # Edge Case 1 — empty
        assert safe_load_image("/nonexistent/B001J63LJQ.jpg").size == INPUT_SIZE  # Edge Case 2
        logger.info("safe_load_image(): PASSED")

        # ── Transform pipelines ───────────────────────────────────────────────────
        _ = get_transforms(config, "train")
        _ = get_transforms(config, "eval")
        logger.info("get_transforms(): PASSED")

        # ── Forward pass ──────────────────────────────────────────────────────────
        dummy = torch.randn(4, 3, 224, 224).to(device)
        with torch.no_grad():
            emb = encoder(dummy)

        assert emb.shape == (4, config.latent_dim), f"Shape mismatch: {emb.shape}"

        # ── L2 norm verification ──────────────────────────────────────────────────
        norms = emb.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "L2 norm failed"

        logger.info(f"Output shape  : {tuple(emb.shape)}")
        logger.info(f"Norms (≈ 1.0) : {norms.tolist()}")
        logger.info(f"Trainable params: {encoder._count_trainable_params():,}")
        logger.info("=" * 60)
        logger.info("  ✅  Smoke test PASSED — ImageEncoder is integration-ready.")
        logger.info("=" * 60)

    except Exception as e:
        logger.exception(f"❌ SMOKE TEST FAILED: {e}")
        sys.exit(1)




