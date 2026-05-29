# =============================================================================
# tabular_encoder.py
# Structured Metadata Representation Encoder — Multimodal AI Pipeline
# =============================================================================
#
# Responsibilities:
#   - Project structured product metadata into the shared multimodal manifold.
#   - Generate stable 512-dim latent embeddings from numerical features.
#   - Ensure latent symmetry with Visual (ConvNeXt) and Semantic (MiniLM) encoders.
#   - Guard against numerical instability (NaN/Inf) common in tabular data.
#
# Refinements:
#   - Xavier Initialization for latent space stability and reproducibility.
#   - get_embedding_dim() for cross-modal interface symmetry.
#   - Rank validation for production-grade defensive engineering.
#   - Centralized Colab import routing for Google Drive execution.
#
# This file is:
#   - representation-only   (no dataset orchestration or scaling logic)
#   - device-agnostic       (GPU transfer belongs in train.py / inference.py)
#   - training-independent  (no loss, optimizer, or scheduler logic)
#   - path-agnostic         (no dataset/checkpoint paths — only project import routing)
#   - architecturally symmetric with image_encoder.py and text_encoder.py
#
# Module dependency order (critical for import-time safety):
#   imports → Colab path → constants → dataclass config →
#   encoder class → factory → smoke test
#
# Compatible with:
#   - torch.utils.data.DataLoader pipelines
#   - CUDA / CPU execution
#   - FP16 mixed precision (enabled externally in train.py)
#   - Tesla T4 / Colab execution
#   - Future categorical embedding expansion
# =============================================================================

import sys
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────
# Google Colab + Project Import Safety
# ─────────────────────────────────────────────────────────────
# Ensures `from models.tabular_encoder import ...` works reliably
# across Colab notebooks, train.py, inference.py, and experimentation
# pipelines when the project is mounted via Google Drive.
#
# This is PROJECT IMPORT ROUTING only — no dataset paths, no checkpoint
# paths, no preprocessing paths. Encoders must remain PATH-AGNOSTIC.
PROJECT_PATH = Path("/content/drive/MyDrive/multi-model-ai")
if str(PROJECT_PATH) not in sys.path:
    sys.path.append(str(PROJECT_PATH))

# ─────────────────────────────────────────────────────────────
# Logging Configuration
# ─────────────────────────────────────────────────────────────

# Module-scoped logger — NO basicConfig here.
# Logging must be configured ONLY in top-level entry points (train.py / inference.py).
# This prevents handler conflicts and duplicate output across multimodal modules.
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Global Constants
# ─────────────────────────────────────────────────────────────
# Defined FIRST — TabularEncoderConfig dataclass defaults reference these at
# class-body evaluation time (import-time). Any constant used as a dataclass
# default field MUST exist before the @dataclass decorator is reached.

# ── Latent Space Contract ───────────────────────────────────────────────────
# ALL encoders in this system MUST output 512-dimensional vectors to ensure
# seamless fusion, retrieval indexing, and SHAP explainability.
DEFAULT_LATENT_DIM: int = 512

# ── Default Architecture Geometry ───────────────────────────────────────────
# Lightweight MLP defaults optimized for Tesla T4 execution.
DEFAULT_INPUT_DIM : int   = 8      # Placeholder: Adjusted via config for specific features
DEFAULT_HIDDEN_DIM: int   = 512
DEFAULT_DROPOUT   : float = 0.1

# ─────────────────────────────────────────────────────────────
# TabularEncoderConfig — Structured Configuration
# ─────────────────────────────────────────────────────────────
# Placed AFTER constants so all default values resolve at import time.
# Placed BEFORE encoder class so TabularEncoder.__init__ can type-hint it.

@dataclass
class TabularEncoderConfig:
    """
    Single source of truth for TabularEncoder hyperparameters.

    Design Rationale:
      - Encapsulation: Prevents scattered constants in the training loop.
      - Scalability: Easy to expand as more metadata fields are engineered.
      - Symmetry: Matches the configuration pattern of Image and Text encoders.
      - Follows the same pattern as ImageEncoderConfig, TextEncoderConfig,
        FusionConfig for consistent multimodal orchestration.

    Integration:
        config  = TabularEncoderConfig(input_dim=12)
        encoder = TabularEncoder(config)

    Future centralization (no encoder changes required):
        configs/tabular_encoder_config.py  ← move dataclass here
        configs/multimodal_config.py       ← umbrella config for all encoders
    """

    # ── Geometry ──────────────────────────────────────────────────────────────
    input_dim : int = DEFAULT_INPUT_DIM
    # Number of numerical/encoded features (e.g., price, rating_number, category_onehot)

    hidden_dim: int = DEFAULT_HIDDEN_DIM
    # Internal representation capacity for the MLP

    latent_dim: int = DEFAULT_LATENT_DIM
    # Final output dimension — MUST match multimodal latent contract (512)

    # ── Regularization ────────────────────────────────────────────────────────
    dropout: float = DEFAULT_DROPOUT
    # Prevents the MLP from memorizing specific tabular rows

    # ── Embedding Geometry ────────────────────────────────────────────────────
    normalize_embeddings: bool = True
    # If True, L2-normalizes output to the unit sphere.
    # CRITICAL for stable cosine-similarity retrieval and fusion magnitude balance.

# ─────────────────────────────────────────────────────────────
# TabularEncoder Implementation
# ─────────────────────────────────────────────────────────────

class TabularEncoder(nn.Module):
    """
    MLP-based encoder for projecting structured metadata into latent space.

    Why MLP over Tab-Transformers?
      1. Purity: We need representation learning, not standalone prediction.
      2. Efficiency: Extremely lightweight on T4 GPUs during multimodal training.
      3. End-to-End: Differentiable projection allows features like 'price'
         to be fine-tuned against visual/semantic features in the shared manifold.

    Architecture:
        Linear(in → hid) → GELU → Dropout → Linear(hid → hid) → GELU → Dropout → Linear(hid → latent)

    Properties:
      - device-agnostic  : no .cuda() / .to() calls here
      - training-free    : no loss, optimizer, or scheduler
      - fusion-agnostic  : plain float tensor output
      - SHAP-compatible  : linear projection head supports gradient attribution

    Args:
        config : TabularEncoderConfig instance. If None, uses all defaults.
    """

    def __init__(self, config: Optional[TabularEncoderConfig] = None) -> None:
        super().__init__()

        # ── Resolve config safely — avoids mutable default argument bug ───────
        # Never use `config: TabularEncoderConfig = TabularEncoderConfig()` as a
        # default argument — Python evaluates that object ONCE at definition time,
        # creating shared state across all callers that don't pass a config.
        self.config = config if config is not None else TabularEncoderConfig()

        # ── Future Architectural Note ────────────────────────────────────────
        # Current implementation assumes numeric structured tensors (pre-scaled).
        # Future versions may support learned categorical embeddings (nn.Embedding)
        # for richer metadata representation if the feature set expands.
        # ──────────────────────────────────────────────────────────────────────

        self.mlp = nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),

            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),

            nn.Linear(self.config.hidden_dim, self.config.latent_dim),
        )

        # Apply explicit weight initialization for multimodal stability
        self._initialize_weights()

        logger.info(
            f"TabularEncoder ready | input_dim={self.config.input_dim} | "
            f"latent_dim={self.config.latent_dim} | "
            f"normalize={self.config.normalize_embeddings} | "
            f"trainable_params={self._count_trainable_params():,}"
        )

    # =========================================================================
    # Weight Initialization
    # =========================================================================

    def _initialize_weights(self) -> None:
        """
        Applies Xavier Uniform initialization to Linear layers.

        Why: Xavier initialization maintains variance across layers, preventing
        vanishing/exploding gradients during the early stages of multimodal fusion
        where the tabular signal must compete with heavy image/text backbones.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # =========================================================================
    # Utility
    # =========================================================================

    def _count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_embedding_dim(self) -> int:
        """Returns latent_dim — used by FusionModel to validate input contracts."""
        return self.config.latent_dim

    # =========================================================================
    # Forward Pass
    # =========================================================================

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projects preprocessed numeric features into the 512-dim manifold.

        Args:
            x : Float tensor (B, input_dim). Must be pre-scaled
                (StandardScaler / MinMaxScaler) by dataset.py.

        Returns:
            Latent embeddings of shape (B, latent_dim).
            L2-normalized to unit sphere if config.normalize_embeddings=True.

        Raises:
            ValueError : If tensor rank, batch size, or feature count is invalid.
        """
        # ── Edge Case 1: Rank Validation (Surgical Fix) ───────────────────────
        if x.ndim != 2:
            raise ValueError(
                f"RANK ERROR: TabularEncoder expected 2D tensor (Batch, Features), "
                f"but received {x.ndim}D tensor with shape {tuple(x.shape)}"
            )

        # ── Edge Case 2: Empty Batch Validation ──────────────────────────────
        if x.shape[0] == 0:
            raise ValueError("TabularEncoder received an empty batch (B=0).")

        # ── Edge Case 3: Feature Count Validation ─────────────────────────────
        if x.shape[1] != self.config.input_dim:
            raise ValueError(
                f"DIM MISMATCH: TabularEncoder expected {self.config.input_dim} features, "
                f"but received {x.shape[1]}. Verify dataset.py feature engineering."
            )

        # ── Edge Case 4: Numerical Stability (NaN/Inf) ────────────────────────
        # Structured data is notorious for corrupt values. We sanitize here
        # to prevent NaN propagation from destroying the shared fusion weights.
        if not torch.isfinite(x).all():
            logger.warning(
                "Detected non-finite values (NaN/Inf) in tabular batch. "
                "Sanitizing to 0.0."
            )
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Latent Projection ─────────────────────────────────────────────────
        latent = self.mlp(x)

        # ── Normalization ─────────────────────────────────────────────────────
        # Prevents the tabular modality from 'overpowering' semantic/visual
        # modalities due to magnitude differences in latent space.
        if self.config.normalize_embeddings:
            latent = F.normalize(latent, p=2, dim=1)

        return latent

# ─────────────────────────────────────────────────────────────
# Factory Function
# ─────────────────────────────────────────────────────────────

def build_tabular_encoder(config: Optional[TabularEncoderConfig] = None) -> TabularEncoder:
    """
    Clean factory entry point for train.py, inference.py, and notebooks.
    Follows the identical pattern as build_encoder() and build_text_encoder().

    The None default is intentional — avoids the mutable default argument trap.
    A fresh TabularEncoderConfig() is instantiated inside TabularEncoder.__init__
    if no config is passed.

    Args:
        config : TabularEncoderConfig or None (defaults applied internally).

    Returns:
        TabularEncoder on CPU — caller is responsible for .to(device).
    """
    return TabularEncoder(config=config)

# ─────────────────────────────────────────────────────────────
# Preprocessing Ownership Note
# ─────────────────────────────────────────────────────────────
# IMPORTANT:
# Feature scaling (StandardScaler, OneHotEncoding) MUST be handled in
# the preprocessing pipeline or dataset.py. The TabularEncoder assumes
# it is receiving cleaned, normalized tensors.
# This separation ensures the model remains a pure representation learner.

# =============================================================================
# Smoke Test  —  python tabular_encoder.py
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
    logger.info("  tabular_encoder.py — smoke test")
    logger.info("=" * 60)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {device}")

        # ── Config and encoder ────────────────────────────────────────────────────
        test_input_dim = 10
        config  = TabularEncoderConfig(input_dim=test_input_dim, normalize_embeddings=True)
        encoder = build_tabular_encoder(config)
        encoder.to(device)
        encoder.eval()

        # ── Valid forward pass ────────────────────────────────────────────────────
        logger.info("Testing standard forward pass...")
        dummy = torch.randn(4, test_input_dim).to(device)
        with torch.no_grad():
            emb = encoder(dummy)

        assert emb.shape == (4, 512), f"Shape mismatch: {emb.shape}"
        logger.info(f"Output shape     : {tuple(emb.shape)}  ✅")

        # ── L2 norm verification ──────────────────────────────────────────────────
        norms = emb.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "L2 norm failed"
        logger.info(f"Norms (≈ 1.0)    : {[round(n, 4) for n in norms.tolist()]}  ✅")

        # ── Interface symmetry ────────────────────────────────────────────────────
        assert encoder.get_embedding_dim() == 512
        logger.info(f"get_embedding_dim: {encoder.get_embedding_dim()}  ✅")

        # ── NaN sanitization ──────────────────────────────────────────────────────
        logger.info("Testing NaN sanitization...")
        nan_input = torch.randn(4, test_input_dim).to(device)
        nan_input[0, 0] = float("nan")
        with torch.no_grad():
            nan_emb = encoder(nan_input)
        assert torch.isfinite(nan_emb).all(), "NaN propagated through encoder"
        logger.info("NaN sanitization : PASSED  ✅")

        # ── Rank guard ────────────────────────────────────────────────────────────
        logger.info("Testing rank guard...")
        try:
            encoder(torch.randn(4, test_input_dim, 1).to(device))
            assert False, "Should have raised ValueError"
        except ValueError as e:
            logger.info(f"Rank guard       : caught → {e}  ✅")

        # ── Empty batch guard ─────────────────────────────────────────────────────
        logger.info("Testing empty batch guard...")
        try:
            encoder(torch.randn(0, test_input_dim).to(device))
            assert False, "Should have raised ValueError"
        except ValueError as e:
            logger.info(f"Batch guard      : caught → {e}  ✅")

        # ── Dimension guard ───────────────────────────────────────────────────────
        logger.info("Testing dimension guard...")
        try:
            encoder(torch.randn(4, test_input_dim + 3).to(device))
            assert False, "Should have raised ValueError"
        except ValueError as e:
            logger.info(f"Dim guard        : caught → {e}  ✅")

        # ── Trainable parameter count ─────────────────────────────────────────────
        logger.info(f"Trainable params : {encoder._count_trainable_params():,}")

        logger.info("=" * 60)
        logger.info("  ✅  Smoke test PASSED — TabularEncoder is infrastructure-grade stable.")
        logger.info("=" * 60)

    except Exception as e:
        logger.exception(f"❌ SMOKE TEST FAILED: {e}")
        sys.exit(1)