# =============================================================================
# image_encoder.py
# Visual Representation Encoder -- Multimodal AI Pipeline
# =============================================================================
#
# Responsibilities (this file ONLY):
#   - ConvNeXt backbone feature extraction
#   - Projection head (768 -> 512 -> 512)
#   - L2 normalization of latent embeddings
#   - Backbone freeze / unfreeze control
#
# Responsibilities that live ELSEWHERE (do NOT add here):
#   +-----------------------------+---------------------------+
#   | Responsibility              | Correct File              |
#   +-----------------------------+---------------------------+
#   | image loading               | data_pipeline/transforms  |
#   | PIL mode conversion         | data_pipeline/transforms  |
#   | augmentation pipelines      | data_pipeline/transforms  |
#   | resize / crop / normalize   | data_pipeline/transforms  |
#   | tensor validation           | data_pipeline/transforms  |
#   | modality dropout / fusion   | fusion.py                 |
#   | train/eval mode switching   | train.py                  |
#   | optimizer / scheduler       | train.py                  |
#   +-----------------------------+---------------------------+
#
# This file is:
#   - encoder-only         (no training logic)
#   - device-agnostic      (GPU transfer belongs in train.py / inference.py)
#   - training-independent (no loss, optimizer, or scheduler)
#   - fusion-independent   (latent vectors are modality-agnostic)
#   - preprocessing-free   (receives ready tensors only)
#
# Module dependency order (critical for import-time safety):
#   imports -> constants -> dataclass config -> projection head ->
#   encoder -> factory -> smoke test
#
# Compatible with:
#   - torch.utils.data.DataLoader pipelines
#   - CUDA / CPU execution
#   - FP16 mixed precision (enabled externally in train.py)
#   - Tesla T4 / Colab execution
#   - Future ViT / SwinTransformer backbone swaps
# =============================================================================

import sys
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

# ---------------------------------------------------------------
# Project Import Routing (local + Colab compatible)
# ---------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn
import timm

# =============================================================================
# Logging
# =============================================================================

# Module-scoped logger -- NO basicConfig here.
# Logging must be configured ONLY in top-level entry points (train.py / inference.py).
# This prevents handler conflicts and duplicate output across multimodal modules.
logger = logging.getLogger(__name__)

# =============================================================================
# Global Constants
# =============================================================================
# Defined FIRST -- ImageEncoderConfig dataclass defaults reference these at
# class-body evaluation time (import-time). Any constant used as a dataclass
# default field MUST exist before the @dataclass decorator is reached.

# -- ConvNeXt Tiny feature dimension (timm, num_classes=0) --------------------
# Default reference for documentation and smoke-test fallback.
# At runtime, ImageEncoder.__init__ queries backbone.num_features dynamically
# to support backbone swaps (ViT, SwinTransformer, etc.) without code changes.
CONVNEXT_FEATURE_DIM: int = 768

# -- Latent space defaults -----------------------------------------------------
DEFAULT_LATENT_DIM : int   = 512
DEFAULT_HIDDEN_DIM : int   = 512
DEFAULT_DROPOUT    : float = 0.2

# =============================================================================
# ImageEncoderConfig -- Structured Configuration
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
        configs/image_encoder_config.py  <- move dataclass here
        configs/multimodal_config.py     <- umbrella config for all encoders
    """

    # -- Backbone --------------------------------------------------------------
    backbone_name       : str            = "convnext_tiny"
    # timm model string -- change here to swap backbone without touching encoder internals
    pretrained          : bool           = True
    # Always True in production; False only for architecture unit tests

    # -- Latent Space ----------------------------------------------------------
    latent_dim          : int            = DEFAULT_LATENT_DIM
    # Output embedding size -- must match FusionConfig.image_dim downstream
    hidden_dim          : int            = DEFAULT_HIDDEN_DIM
    # Projection head bottleneck: Linear(CONVNEXT_FEATURE_DIM -> hidden_dim -> latent_dim)

    # -- Regularization --------------------------------------------------------
    dropout             : float          = DEFAULT_DROPOUT
    # Projection head dropout -- increase if overfitting on small datasets

    # -- Embedding Geometry ----------------------------------------------------
    normalize_embeddings: bool           = True
    # L2-normalize output to unit sphere
    # Required for cosine retrieval, contrastive loss, multimodal alignment

    # -- Training Control ------------------------------------------------------
    freeze_backbone     : bool           = True
    # Freeze ConvNeXt stages 0-2; keep stage 3 + projection trainable
    # Reduces VRAM; prevents early overfitting; call unfreeze_backbone() later

    # -- Preprocessing Geometry -- MIGRATED ------------------------------------
    # input_size and resize_size have been migrated to:
    #   data_pipeline/transforms.py -> TransformConfig
    # ImageEncoder no longer owns preprocessing geometry.

# =============================================================================
# Preprocessing and Augmentation -- MIGRATED
# =============================================================================
# safe_load_image(), build_train_transforms(), build_eval_transforms(),
# get_transforms(), and safe_image_to_rgb() have been migrated to:
#
#   data_pipeline/transforms.py
#
# That module is now the SINGLE IMAGE PREPROCESSING AUTHORITY for the project.
# Usage:
#   from data_pipeline.transforms import safe_load_image, get_transforms
#   from data_pipeline.transforms import TransformConfig, validate_tensor_output
#
# ImageEncoder now adheres to the pure tensor contract: it receives only
# fully prepared float tensors of shape (B, 3, 224, 224).
# =============================================================================

# =============================================================================
# Projection Head
# =============================================================================

class ProjectionHead(nn.Module):
    """
    Two-layer MLP that compresses backbone features into compact latent space.

    Architecture:
        Linear(in_dim -> hidden_dim) -> GELU -> Dropout -> Linear(hidden_dim -> latent_dim)

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
# ImageEncoder -- Main Module
# =============================================================================

class ImageEncoder(nn.Module):
    """
    Pure visual representation encoder for multimodal learning.

    This encoder owns EXACTLY:
      - ConvNeXt backbone (feature extraction)
      - Projection head (768 -> 512 -> 512)
      - L2 normalization
      - Backbone freeze / unfreeze control

    This encoder does NOT own:
      - Image loading / PIL operations   -> data_pipeline/transforms.py
      - Augmentation / preprocessing     -> data_pipeline/transforms.py
      - Modality dropout / masking       -> fusion.py
      - train/eval mode switching        -> train.py

    Architecture:
        Input Tensor  (B, 3, 224, 224)  <- pre-processed by transforms.py
             |
        ConvNeXt Tiny pretrained backbone
             |
        Deep feature vector  (B, 768)
             |
        ProjectionHead: Linear(768->512) -> GELU -> Dropout -> Linear(512->512)
             |
        Latent Embedding  (B, latent_dim)
             |
        Optional L2 Normalization -> unit-sphere embedding

    Args:
        config : ImageEncoderConfig instance. If None, uses all defaults.
    """

    def __init__(self, config: Optional[ImageEncoderConfig] = None) -> None:
        super().__init__()

        # -- Resolve config safely -- avoids mutable default argument bug ------
        if config is None:
            config = ImageEncoderConfig()

        # -- Store for external inspection (train.py / logging / serialization)
        self.config     = config
        self.latent_dim = config.latent_dim
        self.normalize  = config.normalize_embeddings

        # -- Backbone ----------------------------------------------------------
        # num_classes=0 strips the classifier head -> outputs raw (B, 768) features
        logger.info(
            f"Loading backbone: '{config.backbone_name}' "
            f"(pretrained={config.pretrained}, num_classes=0)"
        )
        self.backbone = timm.create_model(
            config.backbone_name,
            pretrained  = config.pretrained,
            num_classes = 0,
        )

        # -- Dynamic feature dimension detection ------------------------------
        # Queries the actual backbone output dim instead of assuming 768.
        # This makes backbone swaps (ViT, Swin, EfficientNet) work without
        # code changes -- only config.backbone_name needs to change.
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

        # -- Selective backbone freezing ---------------------------------------
        if config.freeze_backbone:
            self._freeze_backbone()

        # -- Projection head ---------------------------------------------------
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
            already well-learned from ImageNet -- no benefit retraining them
          - Stage 3 encodes semantic concepts (product silhouettes, categories)
            that benefit from domain adaptation to ecommerce images
          - Projection head is always fully trainable

        VRAM savings:
          Frozen parameters skip gradient computation entirely --
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
                "backbone.stages not found -- backbone fully frozen. "
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
            logger.warning("backbone.stages not found -- cannot selectively unfreeze.")

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
        """Returns latent_dim -- used by FusionModel to validate input contracts."""
        return self.latent_dim

    # =========================================================================
    # Forward Pass
    # =========================================================================

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encodes a batch of preprocessed images into latent embeddings.

        Accepts ONLY pre-processed tensors -- raw image handling belongs in
        data_pipeline/transforms.py. This boundary ensures the encoder is
        DataLoader-safe and testable in isolation.

        Args:
            images : Float tensor (B, 3, 224, 224) on the same device as model.
                     Must be normalized with ImageNet mean/std via transforms.py.

        Returns:
            embeddings : Float tensor (B, latent_dim).
                         L2-normalized to unit sphere if config.normalize_embeddings=True.

        Raises:
            TypeError  : If input is not a torch.Tensor.
            ValueError : If input fails rank, batch, channel, dtype, or NaN checks.
        """
        # -- Type check --------------------------------------------------------
        if not isinstance(images, torch.Tensor):
            raise TypeError(
                f"ImageEncoder.forward() expected torch.Tensor, "
                f"got {type(images).__name__}"
            )

        # -- Rank check (4D) ---------------------------------------------------
        if images.ndim != 4:
            raise ValueError(
                f"ImageEncoder.forward() expected 4D tensor (B,C,H,W), "
                f"got {images.ndim}D shape {tuple(images.shape)}"
            )

        # -- Batch size > 0 ----------------------------------------------------
        if images.shape[0] == 0:
            raise ValueError("ImageEncoder.forward() received empty batch (B=0).")

        # -- Channel count = 3 -------------------------------------------------
        if images.shape[1] != 3:
            raise ValueError(
                f"ImageEncoder.forward() expected 3 channels (RGB), "
                f"got {images.shape[1]}. Verify transforms.py preprocessing."
            )

        # -- Floating point dtype ----------------------------------------------
        if not images.is_floating_point():
            raise ValueError(
                f"ImageEncoder.forward() expected floating point dtype, "
                f"got {images.dtype}. Ensure transforms.py returns float tensors."
            )

        # -- NaN / Inf check ---------------------------------------------------
        if not torch.isfinite(images).all():
            raise ValueError(
                "ImageEncoder.forward() received tensor with NaN or Inf values. "
                "Verify transforms.py preprocessing and data pipeline integrity."
            )

        # -- Spatial contract (warn if not 224x224) ----------------------------
        h, w = images.shape[2], images.shape[3]
        if h != 224 or w != 224:
            logger.warning(
                f"ImageEncoder spatial contract: expected (224, 224), "
                f"got ({h}, {w}). Model may produce unexpected results."
            )

        # -- ConvNeXt feature extraction -> (B, 768) ---------------------------
        features = self.backbone(images)

        # -- Projection -> (B, latent_dim) -------------------------------------
        embeddings = self.projection(features)

        # -- Optional L2 normalization -> unit sphere --------------------------
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

    Usage:
        from models.image_encoder import build_encoder, ImageEncoderConfig
        from data_pipeline.transforms import get_transforms, safe_load_image

        config  = ImageEncoderConfig(latent_dim=512, freeze_backbone=True)
        encoder = build_encoder(config)
        tfm     = get_transforms(mode="eval")

        image  = safe_load_image("/path/to/img.jpg")
        tensor = tfm(image).unsqueeze(0)       # (1, 3, 224, 224)

        encoder.to(device)
        with torch.no_grad():
            emb = encoder(tensor.to(device))   # (1, 512)

    Args:
        config : ImageEncoderConfig or None (defaults applied internally).

    Returns:
        ImageEncoder on CPU -- caller is responsible for .to(device).
    """
    return ImageEncoder(config)

# =============================================================================
# Smoke Test  --  python image_encoder.py
# =============================================================================

if __name__ == "__main__":

    # -- Configure logging for smoke test only ---------------------------------
    logging.basicConfig(
        level  = logging.INFO,
        format = "[%(asctime)s] [%(levelname)s] %(name)s -- %(message)s",
        datefmt= "%H:%M:%S",
    )

    logger.info("=" * 60)
    logger.info("  image_encoder.py -- smoke test")
    logger.info("=" * 60)

    try:
        # Import preprocessing from the centralized authority
        from data_pipeline.transforms import (
            TransformConfig,
            safe_load_image,
            get_transforms,
            validate_tensor_output,
            INPUT_SIZE,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {device}")

        # -- Config and encoder ------------------------------------------------
        # Smoke test uses pretrained=False to avoid network dependency.
        # Production default remains pretrained=True (see ImageEncoderConfig).
        config  = ImageEncoderConfig(latent_dim=512, freeze_backbone=True, pretrained=False)
        encoder = build_encoder(config)
        encoder.to(device)
        encoder.eval()

        # -- safe_load_image edge cases (now in data_pipeline) -----------------
        logger.info("Testing safe_load_image() edge cases...")
        assert safe_load_image(None).size == INPUT_SIZE
        assert safe_load_image("").size == INPUT_SIZE
        assert safe_load_image("/nonexistent/B001J63LJQ.jpg").size == INPUT_SIZE
        logger.info("safe_load_image(): PASSED")

        # -- Transform pipelines (now in data_pipeline) ------------------------
        _ = get_transforms(mode="train")
        _ = get_transforms(mode="eval")
        logger.info("get_transforms(): PASSED")

        # -- Forward pass ------------------------------------------------------
        dummy = torch.randn(4, 3, 224, 224).to(device)
        with torch.no_grad():
            emb = encoder(dummy)

        assert emb.shape == (4, config.latent_dim), f"Shape mismatch: {emb.shape}"

        # -- L2 norm verification ----------------------------------------------
        norms = emb.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "L2 norm failed"

        logger.info(f"Output shape     : {tuple(emb.shape)}")
        logger.info(f"Norms (approx 1) : {[round(n, 4) for n in norms.tolist()]}")
        logger.info(f"Trainable params : {encoder._count_trainable_params():,}")
        logger.info("=" * 60)
        logger.info("  PASS  Smoke test PASSED -- ImageEncoder is integration-ready.")
        logger.info("=" * 60)

    except Exception as e:
        logger.exception(f"[FAIL] SMOKE TEST FAILED: {e}")
        sys.exit(1)
