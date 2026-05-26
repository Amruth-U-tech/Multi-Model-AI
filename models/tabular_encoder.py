# =============================================================================
# tabular_encoder.py
# Structured Metadata Representation Encoder — Multimodal AI Pipeline
# =============================================================================
#
# Responsibilities:
#   - Project structured product metadata into the shared multimodal manifold.
#   - Generate stable 512-bit latent embeddings from numerical features.
#   - Ensure latent symmetry with Visual (ConvNeXt) and Semantic (MiniLM) encoders.
#   - Guard against numerical instability (NaN/Inf) common in tabular data.
#
# This file is:
#   - representation-only (no dataset orchestration or scaling logic)
#   - device-agnostic      (GPU transfer belongs in train.py / inference.py)
#   - training-independent (no loss, optimizer, or scheduler logic)
#   - architecturally symmetric with image_encoder.py and text_encoder.py
#
# Integration:
#   This encoder provides the 'third view' for the FusionModel, allowing 
#   structured signals like 'price' or 'rating' to influence multimodal retrieval 
#   and classification without dominating the semantic/visual signals.
# =============================================================================

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────
# Logging Configuration
# ─────────────────────────────────────────────────────────────

# Module-scoped logger. basicConfig is avoided to prevent handler conflicts
# in the larger multimodal orchestration system.
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Global Constants
# ─────────────────────────────────────────────────────────────

# ── Latent Space Contract ───────────────────────────────────────────────────
# ALL encoders in this system MUST output 512-dimensional vectors to ensure
# seamless fusion, retrieval indexing, and SHAP explainability.
DEFAULT_LATENT_DIM: int = 512

# ── Default Architecture Geometry ───────────────────────────────────────────
# Lightweight MLP defaults optimized for Tesla T4 execution.
DEFAULT_INPUT_DIM: int  = 8     # Placeholder: Adjusted via config for specific features
DEFAULT_HIDDEN_DIM: int = 512
DEFAULT_DROPOUT: float  = 0.1

# ─────────────────────────────────────────────────────────────
# TabularEncoderConfig
# ─────────────────────────────────────────────────────────────

@dataclass
class TabularEncoderConfig:
    """
    Single source of truth for TabularEncoder hyperparameters.
    
    Design Rationale:
      - Encapsulation: Prevents scattered constants in the training loop.
      - Scalability: Easy to expand as more metadata fields are engineered.
      - Symmetry: Matches the configuration pattern of Image and Text encoders.
    """
    
    # ── Geometry ──────────────────────────────────────────────────────────────
    input_dim: int = DEFAULT_INPUT_DIM
    # Number of numerical/encoded features (e.g., price, rating, category_onehot)
    
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    # Internal representation capacity for the MLP
    
    latent_dim: int = DEFAULT_LATENT_DIM
    # Final output dimension — MUST match multimodal latent contract
    
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
    """

    def __init__(self, config: Optional[TabularEncoderConfig] = None) -> None:
        super().__init__()
        
        # Resolve config safely — avoids mutable default argument issues in notebooks
        self.config = config if config is not None else TabularEncoderConfig()
        
        # ── Feature Projection Pipeline ───────────────────────────────────────
        # We use GELU as the activation to match modern Transformer/ConvNeXt 
        # architectures, ensuring smoother gradient flow across the fusion layer.
        self.mlp = nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            
            nn.Linear(self.config.hidden_dim, self.config.latent_dim)
        )
        
        logger.info(
            f"TabularEncoder Initialized | input_dim: {self.config.input_dim} "
            f"| latent_dim: {self.config.latent_dim} | normalize: {self.config.normalize_embeddings}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projects structured features into the multimodal latent manifold.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim). 
               Must be pre-scaled (StandardScaler/MinMaxScaler).
        
        Returns:
            Latent embeddings of shape (batch_size, 512).
        """
        # ── Edge Case 1: Empty Batch Validation ──────────────────────────────
        if x.shape[0] == 0:
            raise ValueError("TabularEncoder received an empty batch.")

        # ── Edge Case 2: Dimension Mismatch ───────────────────────────────────
        if x.shape[1] != self.config.input_dim:
            raise ValueError(
                f"Input dimension mismatch. Expected {self.config.input_dim}, "
                f"but received {x.shape[1]} features."
            )

        # ── Edge Case 3: Numerical Stability (NaN/Inf) ────────────────────────
        # Structured data is notorious for corrupt values. We sanitize here 
        # to prevent NaN propagation from destroying the shared fusion weights.
        if not torch.isfinite(x).all():
            logger.warning("Detected Non-finite values (NaN/Inf) in tabular batch. Sanitizing to 0.0.")
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Latent Projection ─────────────────────────────────────────────────
        latent = self.mlp(x)
        
        # ── Normalization ─────────────────────────────────────────────────────
        # Prevents the tabular modality from 'overpowering' semantic/visual
        # modalities due to magnitude differences in latent space.
        if self.config.normalize_embeddings:
            latent = F.normalize(latent, p=2, dim=1)
            
        return latent

    def get_embedding_dim(self) -> int:
        """Standardized interface for FusionModel dimension validation."""
        return self.config.latent_dim

# ─────────────────────────────────────────────────────────────
# Factory Function
# ─────────────────────────────────────────────────────────────

def build_tabular_encoder(config: Optional[TabularEncoderConfig] = None) -> TabularEncoder:
    """
    Clean entry point for building the encoder. 
    Matches build_image_encoder() and build_text_encoder() signatures.
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

# ─────────────────────────────────────────────────────────────
# Smoke Test (Google Colab / Local Debugging)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Initialize logging for standalone test
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s'
    )
    
    print("\n" + "─"*60)
    print("RUNNING TABULAR ENCODER SMOKE TEST")
    print("─"*60)

    # 1. Setup Config
    test_input_dim = 12
    config = TabularEncoderConfig(input_dim=test_input_dim, normalize_embeddings=True)
    
    # 2. Instantiate Encoder
    encoder = build_tabular_encoder(config)
    encoder.eval() # Set to eval mode for deterministic smoke test
    
    # 3. Create Dummy Data (Batch of 4 products)
    # Includes an intentional NaN to test sanitization logic
    dummy_input = torch.randn(4, test_input_dim)
    dummy_input[0, 0] = float('nan') 
    
    # 4. Perform Forward Pass
    try:
        with torch.no_grad():
            embeddings = encoder(dummy_input)
            
        print(f"✅ Input Shape      : {tuple(dummy_input.shape)}")
        print(f"✅ Output Shape     : {tuple(embeddings.shape)}")
        
        # 5. Validate Latent Contract
        assert embeddings.shape == (4, DEFAULT_LATENT_DIM), "❌ Latent Dimension Mismatch!"
        print(f"✅ Latent Contract  : {DEFAULT_LATENT_DIM}-bit verified.")

        # 6. Validate Normalization
        norms = torch.norm(embeddings, p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms)), "❌ Normalization Failed!"
        print(f"✅ Normalization    : Unit-sphere L2 verified.")

        # 7. Validate Parameter Count (Lightweight Check)
        total_params = sum(p.numel() for p in encoder.parameters())
        print(f"✅ Model Parameters : {total_params:,}")
        
        print("\n" + "─"*60)
        print("RESULT: SMOKE TEST PASSED — ENCODER IS INTEGRATION READY")
        print("─"*60 + "\n")

    except Exception as e:
        print(f"\n❌ SMOKE TEST FAILED: {str(e)}\n")

















        # =============================================================================
# tabular_encoder.py
# Structured Metadata Representation Encoder — Multimodal AI Pipeline
# =============================================================================
#
# Responsibilities:
#   - Project structured product metadata into the shared multimodal manifold.
#   - Generate stable 512-bit latent embeddings from numerical features.
#   - Ensure latent symmetry with Visual (ConvNeXt) and Semantic (MiniLM) encoders.
#
# Refinements:
#   - Added get_embedding_dim() for cross-modal interface symmetry.
#   - Added Xavier Initialization for latent space stability and reproducibility.
#   - Documented future categorical embedding paths for scalability.
# =============================================================================

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────
# Logging Configuration
# ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Global Constants
# ─────────────────────────────────────────────────────────────
DEFAULT_LATENT_DIM: int = 512
DEFAULT_INPUT_DIM: int  = 8     
DEFAULT_HIDDEN_DIM: int = 512
DEFAULT_DROPOUT: float  = 0.1

# ─────────────────────────────────────────────────────────────
# TabularEncoderConfig
# ─────────────────────────────────────────────────────────────

@dataclass
class TabularEncoderConfig:
    """Single source of truth for TabularEncoder hyperparameters."""
    input_dim: int = DEFAULT_INPUT_DIM
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    latent_dim: int = DEFAULT_LATENT_DIM
    dropout: float = DEFAULT_DROPOUT
    normalize_embeddings: bool = True

# ─────────────────────────────────────────────────────────────
# TabularEncoder Implementation
# ─────────────────────────────────────────────────────────────

class TabularEncoder(nn.Module):
    """
    MLP-based encoder for projecting structured metadata into latent space.
    
    Architecture:
        Linear → GELU → Dropout → Linear → GELU → Dropout → Linear (Projection)
    """

    def __init__(self, config: Optional[TabularEncoderConfig] = None) -> None:
        super().__init__()
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
            
            nn.Linear(self.config.hidden_dim, self.config.latent_dim)
        )
        
        # Apply explicit weight initialization for multimodal stability
        self._initialize_weights()
        
        logger.info(
            f"TabularEncoder Initialized | input_dim: {self.config.input_dim} "
            f"| latent_dim: {self.config.latent_dim} | symmetry: get_embedding_dim() enabled"
        )

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects preprocessed numeric features into the 512-dim manifold."""
        # Validate batch and dimensions
        if x.shape[0] == 0:
            raise ValueError("TabularEncoder received an empty batch.")
        if x.shape[1] != self.config.input_dim:
            raise ValueError(f"Dim mismatch: Expected {self.config.input_dim}, got {x.shape[1]}")

        # Sanitize potential numerical instability
        if not torch.isfinite(x).all():
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        latent = self.mlp(x)
        
        if self.config.normalize_embeddings:
            latent = F.normalize(latent, p=2, dim=1)
            
        return latent

    def get_embedding_dim(self) -> int:
        """
        Returns the latent dimension of the encoder.
        Provides architectural symmetry with Visual and Semantic encoders.
        """
        return self.config.latent_dim

# ─────────────────────────────────────────────────────────────
# Factory Function
# ─────────────────────────────────────────────────────────────

def build_tabular_encoder(config: Optional[TabularEncoderConfig] = None) -> TabularEncoder:
    """Clean entry point for building the symmetric tabular encoder."""
    return TabularEncoder(config=config)

# ─────────────────────────────────────────────────────────────
# Smoke Test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    
    print("\n" + "─"*60)
    print("RUNNING SYMMETRIC TABULAR ENCODER SMOKE TEST")
    print("─"*60)

    encoder = build_tabular_encoder(TabularEncoderConfig(input_dim=10))
    encoder.eval()
    
    # Test Data
    dummy_input = torch.randn(4, 10)
    
    with torch.no_grad():
        emb = encoder(dummy_input)
        
    print(f"✅ Output Shape      : {tuple(emb.shape)}")
    print(f"✅ Interface Symmetry: get_embedding_dim() -> {encoder.get_embedding_dim()}")
    
    # Assertions
    assert emb.shape == (4, 512)
    assert encoder.get_embedding_dim() == 512
    
    print("\n" + "─"*60)
    print("RESULT: SYMMETRY REFINEMENT SUCCESSFUL")
    print("─"*60 + "\n")