# =============================================================================
# text_encoder.py
# Semantic Representation Encoder — Multimodal AI Pipeline
# =============================================================================
#
# Responsibilities (this file ONLY):
#   - Transformer forward pass (MiniLM-L6-v2)
#   - Mean pooling over attended token embeddings
#   - Projection head (384 → 512 → 512)
#   - L2 normalization of latent embeddings
#   - Backbone freeze / unfreeze control
#
# Responsibilities that live ELSEWHERE (do NOT add here):
#   ┌─────────────────────────────┬──────────────────┐
#   │ Responsibility              │ Correct File     │
#   ├─────────────────────────────┼──────────────────┤
#   │ text sanitization           │ dataset.py       │
#   │ tokenization                │ dataset.py       │
#   │ batching / collation        │ dataset.py       │
#   │ attention-mask assembly     │ dataset.py       │
#   │ modality dropout / masking  │ fusion_model.py  │
#   │ cross-modal weighting       │ fusion_model.py  │
#   │ train/eval mode switching   │ train.py         │
#   │ freeze schedule orchestration│ train.py        │
#   │ optimizer / scheduler       │ train.py         │
#   └─────────────────────────────┴──────────────────┘
#
# Why tokenization belongs in dataset.py and NOT here:
#   DataLoader spawns multiple worker processes. If tokenization runs inside
#   the model's forward(), it executes on the GPU-side main process, blocking
#   the entire training loop while workers sit idle. Tokenization in
#   dataset.__getitem__() runs in parallel across CPU workers, fills the
#   prefetch buffer, and keeps the GPU fed continuously. This is the correct
#   multiprocessing-aware design for scalable DataLoader pipelines.
#
# Why modality dropout belongs in fusion_model.py and NOT here:
#   Dropping an entire modality embedding is a cross-modal decision — it
#   affects how other modalities compensate. The fusion layer owns that
#   interaction. The encoder's only job is to produce the best possible
#   embedding from its input. Mixing fusion policy into encoder forward()
#   violates single-responsibility and makes the encoder untestable in
#   isolation.
#
# This file is:
#   - encoder-only         (no training logic)
#   - device-agnostic      (GPU transfer belongs in train.py / inference.py)
#   - training-independent (no loss, optimizer, or scheduler)
#   - fusion-independent   (latent vectors are modality-agnostic)
#
# Module dependency order (critical for import-time safety):
#   imports → constants → dataclass config → transitional utilities →
#   projection head → encoder → factory → smoke test
#
# Compatible with:
#   - torch.utils.data.DataLoader pipelines
#   - CUDA / CPU execution
#   - FP16 mixed precision (enabled externally in train.py)
#   - Tesla T4 / Colab execution
#   - Future fine-tuning via config.freeze_backbone = False
# =============================================================================

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# =============================================================================
# Logging
# =============================================================================

# Module-scoped logger — NO basicConfig here.
# Logging must be configured ONLY in top-level entry points (train.py / inference.py).
# This prevents handler conflicts and duplicate outputs across multimodal modules.
logger = logging.getLogger(__name__)

# =============================================================================
# Global Constants
# =============================================================================
# Defined FIRST — TextEncoderConfig dataclass defaults reference these at
# class-body evaluation time (import-time). Any constant used as a dataclass
# default field MUST exist before the @dataclass decorator is reached.

# ── Backbone ──────────────────────────────────────────────────────────────────
# MiniLM-L6-v2 chosen for:
#   - Strong semantic quality for short-to-medium ecommerce product text
#   - 384-dim output — lightweight enough for T4 with frozen backbone
#   - Excellent zero-shot retrieval alignment out of the box
#   - Fast inference — critical for multimodal batching throughput
DEFAULT_MODEL_NAME  : str   = "sentence-transformers/all-MiniLM-L6-v2"

# ── MiniLM backbone output dimension ─────────────────────────────────────────
# Fixed by model architecture — do not change without swapping backbone
MINILM_FEATURE_DIM  : int   = 384

# ── Latent space ──────────────────────────────────────────────────────────────
# 512 chosen deliberately (vs image encoder's 256):
#   - Text carries richer semantic density than a single product image
#   - Larger projection gives the bottleneck room to reorganize without
#     over-compressing the semantic signal
#   - FusionModel will project all modalities to a shared fusion dim anyway
#   - Keeps retrieval quality and RAG compatibility strong
DEFAULT_LATENT_DIM  : int   = 512
DEFAULT_HIDDEN_DIM  : int   = 512   # 384→512→512: expand then stabilize

# ── Tokenization ──────────────────────────────────────────────────────────────
# max_length=64 chosen for Amazon Fashion dataset specifically:
#   - Sample titles average 10-25 tokens
#   - Product identity is captured in the first 64 tokens for this domain
#   - Critical for T4 VRAM: attention cost scales with seq_len²
DEFAULT_MAX_LENGTH  : int   = 64

# ── Regularization ────────────────────────────────────────────────────────────
DEFAULT_DROPOUT     : float = 0.1
# Lower than image encoder (0.2) — transformer attention already provides
# internal regularization; excessive dropout here hurts semantic quality

# ── Fallback text ─────────────────────────────────────────────────────────────
# Used by transitional tokenizer utilities (see section below).
# Will move to dataset.py once that file is created.
FALLBACK_TEXT       : str   = "[NO_TEXT_AVAILABLE]"

# =============================================================================
# TextEncoderConfig — Structured Configuration
# =============================================================================

@dataclass
class TextEncoderConfig:
    """
    Single source of truth for all TextEncoder hyperparameters.

    Follows the identical design pattern as ImageEncoderConfig:
      - One config object replaces growing constructor argument lists
      - Inspectable, loggable, serializable (dataclasses.asdict())
      - Future-compatible with YAML / Hydra / argparse config systems
      - Consistent orchestration across all encoders in the multimodal system

    Integration:
        config  = TextEncoderConfig(latent_dim=512)
        encoder = TextEncoder(config)

    Future centralization (no encoder changes required):
        configs/text_encoder_config.py   ← move dataclass here
        configs/multimodal_config.py     ← umbrella config for all encoders
    """

    # ── Backbone ──────────────────────────────────────────────────────────────
    model_name           : str   = DEFAULT_MODEL_NAME
    # HuggingFace model string — change here to swap backbone without touching
    # encoder internals (adjust MINILM_FEATURE_DIM constant if output dim differs)

    # ── Latent Space ──────────────────────────────────────────────────────────
    latent_dim           : int   = DEFAULT_LATENT_DIM
    # Output embedding size — must align with FusionConfig.text_dim downstream
    hidden_dim           : int   = DEFAULT_HIDDEN_DIM
    # Projection bottleneck: Linear(MINILM_FEATURE_DIM → hidden_dim → latent_dim)

    # ── Tokenization ──────────────────────────────────────────────────────────
    max_length           : int   = DEFAULT_MAX_LENGTH
    # Token sequence cap — directly controls VRAM and batch latency.
    # Consumed by dataset.py tokenization; stored here so config is the
    # single source of truth for the entire text preprocessing geometry.

    # ── Regularization ────────────────────────────────────────────────────────
    dropout              : float = DEFAULT_DROPOUT

    # ── Embedding Geometry ────────────────────────────────────────────────────
    normalize_embeddings : bool  = True
    # L2-normalize output to unit sphere.
    # CRITICAL: prevents text magnitude from dominating image/tabular embeddings
    # in concatenation or attention-based fusion. Also required for cosine
    # retrieval and contrastive alignment losses.

    # ── Training Control ──────────────────────────────────────────────────────
    freeze_backbone      : bool  = True
    # ⚠️  CRITICAL MULTIMODAL DESIGN DECISION — read carefully before changing.
    #
    # WHY freeze initially:
    #   MiniLM produces highly dominant semantic representations out of the box.
    #   If the transformer is unfrozen from epoch 1, text gradients overpower
    #   image and tabular gradients during fusion — the model collapses into a
    #   language-only learner. Freezing forces the system to learn multimodal
    #   fusion using fixed, stable semantic anchors first.
    #
    # WHEN to unfreeze (controlled entirely from train.py):
    #   After the projection head and fusion model converge on stable latent
    #   geometry (~5-10 warm-up epochs), train.py calls:
    #       encoder.unfreeze_backbone()
    #   No architecture changes or config rewrites needed.
    #
    # HOW train.py will orchestrate this:
    #   phase_1: TextEncoderConfig(freeze_backbone=True)   ← head-only training
    #   phase_2: encoder.unfreeze_backbone(num_layers=2)   ← staged fine-tuning
    #   phase_3: encoder.unfreeze_backbone()               ← full fine-tuning

    # ── Modality Dropout Hook ─────────────────────────────────────────────────
    modality_dropout_prob: float = 0.1
    # ⚠️  Config field preserved for future fusion_model.py consumption.
    #
    # WHY this field exists here but execution is NOT here:
    #   Dropping an entire modality embedding is a cross-modal fusion decision.
    #   The encoder's job is to produce the best possible embedding from its
    #   input — it should not decide whether that embedding gets used.
    #   fusion_model.py will read config.modality_dropout_prob and apply
    #   the mask AFTER receiving embeddings from all encoders.
    #
    # Set to 0.0 to disable modality dropout entirely.

# =============================================================================
# Transitional Tokenizer Utilities
# =============================================================================
# ⚠️  TRANSITIONAL — these utilities will migrate to dataset.py once created.
#
# WHY they are here temporarily:
#   dataset.py does not yet exist. These helpers are needed now for the
#   smoke test and for notebook/inference use cases. Once dataset.py is
#   implemented, sanitize_text() and tokenize_batch() move there wholesale,
#   and these copies are deleted from this file.
#
# WHY they must eventually live in dataset.py:
#   Tokenization must happen inside Dataset.__getitem__() so DataLoader
#   worker processes can parallelize it across CPU cores. If tokenization
#   runs during the model forward pass, it serializes on the main process
#   and starves the GPU. Pre-tokenizing in the dataset fills the prefetch
#   buffer efficiently and keeps training throughput high.
#
# DO NOT build new logic on top of these functions — treat them as temporary.
# =============================================================================

def sanitize_text(text: Union[str, float, int, None]) -> str:
    """
    [TRANSITIONAL — will move to dataset.py]

    Converts any input to a clean, tokenizer-safe string.

    Handles:
      - None values                        (missing CSV cells)
      - float NaN (pandas missing-value)   (missing CSV cells)
      - Non-string types (int, float)      (numeric columns read as text)
      - Empty strings after stripping      (blank descriptions)

    Returns FALLBACK_TEXT for anything that reduces to empty — never returns
    an empty string, which causes tokenizers to produce degenerate embeddings.

    Args:
        text : Raw value from a dataset row.

    Returns:
        Non-empty string guaranteed safe for tokenization.
    """
    if text is None:
        return FALLBACK_TEXT
    try:
        # float NaN check: NaN != NaN is the only reliable identity test
        if isinstance(text, float) and (text != text):
            return FALLBACK_TEXT
    except Exception:
        pass
    cleaned = str(text).strip()
    return cleaned if cleaned else FALLBACK_TEXT


def load_tokenizer(model_name: str = DEFAULT_MODEL_NAME) -> AutoTokenizer:
    """
    [TRANSITIONAL — will move to dataset.py]

    Loads and returns the HuggingFace tokenizer for the given model.

    Separated from the encoder class so dataset.py can pre-load the
    tokenizer once in Dataset.__init__() and reuse it across all
    __getitem__ calls — avoids re-initializing per batch.

    Args:
        model_name : HuggingFace model identifier.

    Returns:
        AutoTokenizer instance.
    """
    logger.info(f"Loading tokenizer: '{model_name}'")
    tok = AutoTokenizer.from_pretrained(model_name)
    logger.info("Tokenizer loaded.")
    return tok


def tokenize_batch(
    texts     : List[str],
    tokenizer : AutoTokenizer,
    max_length: int,
    device    : Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    [TRANSITIONAL — will move to dataset.py]

    Tokenizes a list of sanitized strings into padded, truncated tensors.

    When this migrates to dataset.py it will be called inside __getitem__()
    per sample (returning individual token tensors), with DataLoader's default
    collate assembling them into batches. The current batch-level signature is
    retained for smoke test and inference compatibility.

    Args:
        texts      : List of sanitized strings (run through sanitize_text first).
        tokenizer  : Pre-loaded AutoTokenizer instance.
        max_length : Maximum token sequence length.
        device     : If provided, tensors are moved here before return.

    Returns:
        Dict with keys "input_ids" and "attention_mask" — both (B, max_length).

    Raises:
        ValueError : If texts list is empty.
    """
    if not texts:
        raise ValueError("tokenize_batch() received an empty text list.")

    try:
        encoded = tokenizer(
            texts,
            padding               = "max_length",
            truncation            = True,
            max_length            = max_length,
            return_tensors        = "pt",
            return_attention_mask = True,
        )
    except Exception as exc:
        logger.warning(
            f"Tokenization failed: {exc}. Replacing batch with fallback tokens."
        )
        encoded = tokenizer(
            [FALLBACK_TEXT] * len(texts),
            padding               = "max_length",
            truncation            = True,
            max_length            = max_length,
            return_tensors        = "pt",
            return_attention_mask = True,
        )

    if device is not None:
        encoded = {k: v.to(device) for k, v in encoded.items()}

    return encoded

# =============================================================================
# Projection Head
# =============================================================================

class ProjectionHead(nn.Module):
    """
    Two-layer MLP that shapes MiniLM features into the multimodal latent space.

    Architecture:
        Linear(in_dim → hidden_dim) → GELU → Dropout → Linear(hidden_dim → latent_dim)

    For text: 384 → 512 → 512
      - Expansion (384→512) gives the head room to reorganize features rather
        than compress them — MiniLM is already compact; further compression
        loses semantic resolution needed for retrieval and RAG
      - GELU: smoother gradients; standard for transformer-adjacent components
      - Dropout: regularizes the bottleneck; helps prevent latent collapse
      - No BatchNorm: stable at batch=1 during retrieval and SHAP inference

    Args:
        in_dim     : Input dimension (MINILM_FEATURE_DIM = 384).
        hidden_dim : Intermediate dimension.
        latent_dim : Output embedding dimension.
        dropout    : Dropout probability.
    """

    def __init__(
        self,
        in_dim    : int   = MINILM_FEATURE_DIM,
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
# TextEncoder — Main Module
# =============================================================================

class TextEncoder(nn.Module):
    """
    Pure semantic representation encoder for multimodal learning.

    This encoder owns EXACTLY:
      - Transformer backbone (MiniLM-L6-v2)
      - Mean pooling over attended token embeddings
      - Projection head (384 → 512 → 512)
      - L2 normalization
      - Backbone freeze / unfreeze control

    This encoder does NOT own:
      - Text sanitization or tokenization   → dataset.py
      - Modality dropout / masking          → fusion_model.py
      - train/eval mode switching           → train.py
      - Freeze schedule orchestration       → train.py
      - Optimizer / scheduler               → train.py

    Architecture:
        input_ids + attention_mask (B, seq_len)  ← pre-tokenized by dataset.py
             ↓
        MiniLM-L6-v2 Transformer (optionally frozen)
             ↓
        Mean pooling over attended tokens → (B, 384)
             ↓
        ProjectionHead: Linear(384→512) → GELU → Dropout → Linear(512→512)
             ↓
        Latent Embedding (B, latent_dim)
             ↓
        Optional L2 Normalization → unit-sphere embedding

    Why mean pooling over [CLS]:
      [CLS] works well for classification but mean pooling over attended tokens
      produces more uniform, retrieval-stable embeddings for variable-length
      ecommerce product text — especially short titles where [CLS] underfits.
      This also matches how sentence-transformers officially uses MiniLM.

    Args:
        config : TextEncoderConfig instance. If None, uses all defaults.
    """

    def __init__(self, config: Optional[TextEncoderConfig] = None) -> None:
        super().__init__()

        # ── Resolve config safely — avoids mutable default argument bug ───────
        # Never use `config: TextEncoderConfig = TextEncoderConfig()` as a
        # default argument. Python evaluates that object ONCE at definition time,
        # creating shared state across all callers. Use None + internal init.
        if config is None:
            config = TextEncoderConfig()

        self.config     = config
        self.latent_dim = config.latent_dim
        self.normalize  = config.normalize_embeddings

        # ── Tokenizer ─────────────────────────────────────────────────────────
        # Stored on encoder so inference.py / notebooks can access it via
        # encoder.tokenizer without a separate import.
        # In the full pipeline, dataset.py will load its own tokenizer instance
        # independently — two instances is correct (one per process boundary).
        logger.info(f"Loading tokenizer and backbone: '{config.model_name}'")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        # ── Transformer backbone ──────────────────────────────────────────────
        self.backbone = AutoModel.from_pretrained(config.model_name)
        logger.info(f"Backbone loaded | output_dim={MINILM_FEATURE_DIM}")

        # ── Selective backbone freezing ───────────────────────────────────────
        if config.freeze_backbone:
            self._freeze_backbone()

        # ── Projection head ───────────────────────────────────────────────────
        self.projection = ProjectionHead(
            in_dim     = MINILM_FEATURE_DIM,
            hidden_dim = config.hidden_dim,
            latent_dim = config.latent_dim,
            dropout    = config.dropout,
        )

        logger.info(
            f"TextEncoder ready | latent_dim={config.latent_dim} | "
            f"max_length={config.max_length} | "
            f"normalize={config.normalize_embeddings} | "
            f"freeze_backbone={config.freeze_backbone} | "
            f"trainable_params={self._count_trainable_params():,}"
        )

    # =========================================================================
    # Backbone Freezing
    # =========================================================================

    def _freeze_backbone(self) -> None:
        """
        Freezes all MiniLM transformer parameters.

        Unlike ImageEncoder (which keeps the final ConvNeXt stage trainable),
        the text backbone is frozen entirely in phase 1 because:

          MiniLM's pretrained sentence embeddings are already semantically
          aligned for ecommerce product text — partial freezing at a layer
          boundary introduces gradient instability with minimal benefit.
          The projection head alone is sufficient to adapt the latent geometry
          for multimodal fusion during warm-up.

        train.py controls when to unfreeze via encoder.unfreeze_backbone().
        No architecture changes needed — only a config flag or a method call.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info(
            "Transformer backbone fully frozen. "
            "Projection head remains trainable. "
            "train.py will call unfreeze_backbone() after warm-up convergence."
        )

    def unfreeze_backbone(self, num_layers: Optional[int] = None) -> None:
        """
        Progressively unfreezes transformer layers for staged fine-tuning.

        Called by train.py — never called inside this encoder's own logic.
        This keeps fine-tuning schedule ownership strictly in train.py.

        Args:
            num_layers : Trailing transformer layers to unfreeze (from deepest).
                         None = unfreeze entire backbone.

        Examples (called from train.py):
            encoder.unfreeze_backbone()              # full backbone
            encoder.unfreeze_backbone(num_layers=2)  # last 2 layers only

        Staged unfreezing strategy:
            Unfreezing deepest layers first reduces catastrophic forgetting
            of general semantic priors while allowing domain adaptation to
            Amazon Fashion product vocabulary.
        """
        if num_layers is None:
            for param in self.backbone.parameters():
                param.requires_grad = True
            logger.info("Full transformer backbone unfrozen for fine-tuning.")
        else:
            if hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layer"):
                for layer in self.backbone.encoder.layer[-num_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True
                logger.info(f"Unfroze last {num_layers} transformer layer(s).")
            else:
                # Fallback for non-BERT-family architectures
                for param in self.backbone.parameters():
                    param.requires_grad = True
                logger.warning(
                    "backbone.encoder.layer not found — unfroze full backbone. "
                    "Override unfreeze_backbone() for non-BERT-family architectures."
                )

    def freeze_backbone(self) -> None:
        """
        Re-freezes entire transformer backbone.
        Called by train.py when switching between training phases.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Transformer backbone fully re-frozen.")

    # =========================================================================
    # Mean Pooling
    # =========================================================================

    @staticmethod
    def _mean_pool(
        token_embeddings: torch.Tensor,
        attention_mask  : torch.Tensor,
    ) -> torch.Tensor:
        """
        Attention-mask-weighted mean over token embeddings.

        Ignores padding tokens in the mean — critical for variable-length
        product titles where padding can dominate the average if unmasked.

        Args:
            token_embeddings : (B, seq_len, hidden_dim) from transformer last layer.
            attention_mask   : (B, seq_len) — 1 for real tokens, 0 for padding.

        Returns:
            Pooled tensor (B, hidden_dim).
        """
        mask_expanded  = attention_mask.unsqueeze(-1).float()          # (B, seq, 1)
        sum_embeddings = (token_embeddings * mask_expanded).sum(dim=1) # (B, hidden)
        sum_mask       = mask_expanded.sum(dim=1).clamp(min=1e-9)      # (B, 1)
        return sum_embeddings / sum_mask                                # (B, hidden)

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

    def forward(
        self,
        input_ids      : torch.Tensor,
        attention_mask : torch.Tensor,
    ) -> torch.Tensor:
        """
        Encodes pre-tokenized inputs into latent semantic embeddings.

        Accepts ONLY pre-tokenized tensors — raw string handling belongs in
        dataset.py. This boundary ensures the encoder is DataLoader-safe and
        testable in isolation without any preprocessing dependencies.

        Modality dropout is NOT applied here — fusion_model.py applies it
        after receiving embeddings from all encoders, because it is a
        cross-modal interaction decision, not a representation decision.

        train.py is responsible for calling encoder.train() / encoder.eval()
        at the correct points in the training loop — the encoder itself does
        not manage its own training mode transitions.

        Args:
            input_ids      : Long tensor (B, seq_len) from dataset.py tokenizer.
            attention_mask : Long tensor (B, seq_len) — 1=real token, 0=padding.

        Returns:
            embeddings : Float tensor (B, latent_dim).
                         L2-normalized to unit sphere if config.normalize_embeddings=True.

        Raises:
            ValueError : If tensors are not 2D, shapes don't match, or batch is empty.
        """
        # ── Input validation ──────────────────────────────────────────────────
        if input_ids.ndim != 2:
            raise ValueError(
                f"TextEncoder.forward() expected 2D input_ids (B, seq_len), "
                f"got {tuple(input_ids.shape)}"
            )
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                f"attention_mask {tuple(attention_mask.shape)} does not "
                f"match input_ids {tuple(input_ids.shape)}"
            )
        if input_ids.shape[0] == 0:
            raise ValueError("TextEncoder.forward() received empty batch (B=0).")

        # ── Transformer forward pass → (B, seq_len, 384) ─────────────────────
        output = self.backbone(
            input_ids      = input_ids,
            attention_mask = attention_mask,
        )

        # ── Mean pooling → (B, 384) ───────────────────────────────────────────
        features = self._mean_pool(output.last_hidden_state, attention_mask)

        # ── Projection → (B, latent_dim) ──────────────────────────────────────
        embeddings = self.projection(features)

        # ── L2 normalization → unit sphere ────────────────────────────────────
        # Prevents text magnitude from dominating image/tabular in fusion.
        # Required for cosine retrieval and contrastive alignment losses.
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    # =========================================================================
    # Convenience Encoder (Inference / Notebooks / SHAP)
    # =========================================================================

    @torch.no_grad()
    def encode_texts(
        self,
        texts  : Union[str, List[str]],
        device : Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Convenience method for encoding raw strings outside a DataLoader context.

        Uses the transitional tokenizer utilities internally. Once dataset.py
        exists, inference.py will construct its own Dataset + DataLoader for
        batch inference — this method remains useful for notebooks, SHAP, and
        single-sample retrieval queries.

        NOT intended for training loops — use forward() with DataLoader tensors.

        Args:
            texts  : A single string or list of strings.
            device : Device to run on. If None, inferred from encoder parameters.

        Returns:
            embeddings : Float tensor (B, latent_dim), L2-normalized.
        """
        if isinstance(texts, str):
            texts = [texts]

        texts = [sanitize_text(t) for t in texts]

        if device is None:
            device = next(self.parameters()).device

        encoded = tokenize_batch(
            texts      = texts,
            tokenizer  = self.tokenizer,
            max_length = self.config.max_length,
            device     = device,
        )

        # Ensure eval mode for inference — modality dropout inactive,
        # projection dropout inactive, deterministic output.
        # train.py manages mode transitions in the training loop;
        # encode_texts() is inference-only so eval mode is always correct here.
        was_training = self.training
        self.eval()
        embeddings = self(encoded["input_ids"], encoded["attention_mask"])
        if was_training:
            self.train()

        return embeddings

# =============================================================================
# Factory Function
# =============================================================================

def build_text_encoder(config: Optional[TextEncoderConfig] = None) -> "TextEncoder":
    """
    Clean factory entry point for train.py, inference.py, and notebooks.
    Follows the identical pattern as build_encoder() in image_encoder.py.

    The None default is intentional — avoids the mutable default argument trap.
    A fresh TextEncoderConfig() is created inside TextEncoder.__init__ if
    no config is passed.

    Usage:
        from text_encoder import build_text_encoder, TextEncoderConfig

        config  = TextEncoderConfig(latent_dim=512, freeze_backbone=True)
        encoder = build_text_encoder(config)
        encoder.to(device)

        # DataLoader-style (primary training path):
        emb = encoder(input_ids, attention_mask)          # (B, 512)

        # Convenience / inference / SHAP:
        emb = encoder.encode_texts(["product title"])     # (1, 512)

    Args:
        config : TextEncoderConfig or None (defaults applied internally).

    Returns:
        TextEncoder on CPU — caller is responsible for .to(device).
    """
    return TextEncoder(config)

# =============================================================================
# Smoke Test  —  python text_encoder.py
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
    logger.info("  text_encoder.py — smoke test")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Config and encoder ────────────────────────────────────────────────────
    config  = TextEncoderConfig(latent_dim=512, freeze_backbone=True)
    encoder = build_text_encoder(config)
    encoder.to(device)
    encoder.eval()

    # ── sanitize_text edge cases ──────────────────────────────────────────────
    logger.info("Testing sanitize_text() edge cases...")
    assert sanitize_text(None)          == FALLBACK_TEXT
    assert sanitize_text("")            == FALLBACK_TEXT
    assert sanitize_text("   ")        == FALLBACK_TEXT
    assert sanitize_text(float("nan")) == FALLBACK_TEXT
    assert sanitize_text(42)           == "42"
    assert sanitize_text(3.14)         == "3.14"
    logger.info("sanitize_text(): PASSED  ✅")

    # ── Realistic Amazon Fashion product titles ───────────────────────────────
    raw_texts = [
        "Spanx Core In-Power Line Super High Shaping Sheers Very Black F",
        "KingSize Men's Big & Tall Lightweight Jersey Cargo Sweatpants",
        "Boho Tassel Earrings for Women Girls Multicolor Bohemian Fan Statement",
        "",    # Edge Case — becomes FALLBACK_TEXT
    ]
    clean = [sanitize_text(t) for t in raw_texts]

    # ── encode_texts convenience method ───────────────────────────────────────
    logger.info("Testing encode_texts() with Amazon product titles...")
    emb = encoder.encode_texts(clean, device=device)
    assert emb.shape == (4, config.latent_dim), f"Shape mismatch: {emb.shape}"
    logger.info(f"encode_texts() shape : {tuple(emb.shape)}  ✅")

    # ── L2 normalization ──────────────────────────────────────────────────────
    norms = emb.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    logger.info(f"Norms (≈ 1.0)        : {[round(n,4) for n in norms.tolist()]}  ✅")

    # ── DataLoader-style forward pass (primary training path) ─────────────────
    logger.info("Testing DataLoader-style forward pass...")
    encoded = tokenize_batch(
        texts      = clean,
        tokenizer  = encoder.tokenizer,
        max_length = config.max_length,
        device     = device,
    )
    with torch.no_grad():
        emb2 = encoder(encoded["input_ids"], encoded["attention_mask"])
    assert emb2.shape == (4, config.latent_dim), f"Forward shape mismatch: {emb2.shape}"
    logger.info(f"forward() shape      : {tuple(emb2.shape)}  ✅")

    # ── Trainable parameter count ─────────────────────────────────────────────
    logger.info(f"Trainable params     : {encoder._count_trainable_params():,}")

    logger.info("=" * 60)
    logger.info("  ✅  Smoke test PASSED — TextEncoder is integration-ready.")
    logger.info("=" * 60)