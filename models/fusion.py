# =============================================================================
# fusion_model.py
# Multimodal Orchestration Intelligence — INFRASTRUCTURE-GRADE STABLE
# =============================================================================

import sys
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

# ─────────────────────────────────────────────────────────────
# Google Colab + Project Import Safety
# ─────────────────────────────────────────────────────────────
PROJECT_PATH = "/content/drive/MyDrive/multi-model-ai"
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

logger = logging.getLogger(__name__)

@dataclass
class FusionConfig:
    """
    Configuration for the Multimodal System Brain.
    Ensures symmetry across the 512-dimensional latent manifold.
    """
    embedding_dim: int = 512
    hidden_dim: int = 1024
    dropout: float = 0.2
    modality_dropout_prob: float = 0.1
    normalize_embeddings: bool = True

class FusionModel(nn.Module):
    """
    The SYSTEM Orchestration Brain.
    Handles adaptive modality trust and information preservation.
    """
    def __init__(self, config: Optional[FusionConfig] = None):
        super().__init__()
        self.config = config or FusionConfig()
        
        # 1. Modality Gating (Stable Adaptive Trust)
        self.modality_gate = nn.Sequential(
            nn.Linear(self.config.embedding_dim * 3, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, 3),
            nn.Softmax(dim=1)
        )
        
        # 2. Fusion Projection MLP (Interaction Space)
        self.fusion_projection = nn.Sequential(
            nn.Linear(self.config.embedding_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.embedding_dim),
            nn.Dropout(self.config.dropout)
        )
        
        # 3. Residual Path Hook
        self.residual_proj = nn.Identity() 
        
        # 4. Prediction Head (Regression)
        self.prediction_head = nn.Sequential(
            nn.Linear(self.config.embedding_dim, 256),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(256, 1)
        )

        # Initialize weights for gradient stability
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Applies Xavier initialization to prevent initial modality-favoritism."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _validate_inputs(self, img: torch.Tensor, txt: torch.Tensor, tab: torch.Tensor):
        """
        STRICT INFRASTRUCTURE VALIDATION.
        Protects against rank mismatch, batch misalignment, and dimension drift.
        """
        # A. Rank Validation (Final Surgical Fix)
        for name, tensor in [("Image", img), ("Text", txt), ("Tabular", tab)]:
            if tensor.ndim != 2:
                raise ValueError(
                    f"RANK ERROR: {name} must be a 2D tensor [Batch, Dim]. "
                    f"Received {tensor.ndim}D shape: {list(tensor.shape)}"
                )

        # B. Batch Size Consistency
        b_img, b_txt, b_tab = img.shape[0], txt.shape[0], tab.shape[0]
        if not (b_img == b_txt == b_tab):
            raise ValueError(
                f"BATCH MISMATCH: img({b_img}), txt({b_txt}), tab({b_tab}). "
                "All modalities must have identical batch sizes."
            )

        # C. Latent Dimension Contract
        expected = self.config.embedding_dim
        for name, tensor in [("Image", img), ("Text", txt), ("Tabular", tab)]:
            if tensor.shape[1] != expected:
                raise ValueError(f"DIM MISMATCH: {name} expected {expected}, got {tensor.shape[1]}.")

    def _apply_modality_dropout(self, img, txt, tab) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Survival-guaranteed dropout to ensure training resilience."""
        if not self.training:
            return img, txt, tab

        probs = torch.ones(3, device=img.device) * (1 - self.config.modality_dropout_prob)
        mask = torch.bernoulli(probs)
        
        # Guarantee at least one modality survives
        if mask.sum() == 0:
            mask[torch.randint(0, 3, (1,)).item()] = 1.0
            
        return img * mask[0], txt * mask[1], tab * mask[2]

    def forward(self, img_emb: torch.Tensor, txt_emb: torch.Tensor, tab_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 1. Validation & Sanitation
        self._validate_inputs(img_emb, txt_emb, tab_emb)
        img_emb, txt_emb, tab_emb = torch.nan_to_num(img_emb), torch.nan_to_num(txt_emb), torch.nan_to_num(tab_emb)

        # 2. Resilient Dropout
        img_d, txt_d, tab_d = self._apply_modality_dropout(img_emb, txt_emb, tab_emb)

        # 3. Gated Fusion logic
        concat_context = torch.cat([img_d, txt_d, tab_d], dim=1)
        gate_weights = self.modality_gate(concat_context) 
        
        weighted_fusion = (
            gate_weights[:, 0:1] * img_d +
            gate_weights[:, 1:2] * txt_d +
            gate_weights[:, 2:3] * tab_d
        )

        # 4. Residual MLP Path
        fused_embedding = self.fusion_projection(weighted_fusion) + self.residual_proj(weighted_fusion)
        
        # 5. Geometry Stabilization
        if self.config.normalize_embeddings:
            fused_embedding = F.normalize(fused_embedding, p=2, dim=1)

        return {
            "fused_embedding": fused_embedding,
            "rating_prediction": self.prediction_head(fused_embedding),
            "modality_weights": gate_weights
        }

# ─────────────────────────────────────────────────────────────
# Smoke Test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model = FusionModel()
    
    # Valid Test
    B = 4
    m_img, m_txt, m_tab = torch.randn(B, 512), torch.randn(B, 512), torch.randn(B, 512)
    
    try:
        out = model(m_img, m_txt, m_tab)
        assert torch.allclose(torch.norm(out["fused_embedding"], dim=1), torch.ones(B), atol=1e-5)
        print("✅ Standard Inference: PASSED")

        # Rank Validation Test
        print("Testing Rank Guard...")
        try:
            model(torch.randn(B, 512, 1), m_txt, m_tab) # Invalid 3D Rank
        except ValueError as e:
            print(f"✅ Rank Guard caught error: {e}")

        # Batch Guard Test
        print("Testing Batch Guard...")
        try:
            model(torch.randn(B, 512), torch.randn(B-1, 512), m_tab)
        except ValueError as e:
            print(f"✅ Batch Guard caught error: {e}")

        print("\nRESULT: FUSION MODEL IS INFRASTRUCTURE-GRADE STABLE.")
    except Exception as e:
        print(f"❌ SMOKE TEST FAILED: {e}")
        sys.exit(1)