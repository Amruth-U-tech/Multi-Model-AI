# Multi-Model AI: Multimodal Product Intelligence Infrastructure

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-yellow.svg?style=flat-square)](https://huggingface.co/)
[![Validation-First](https://img.shields.io/badge/Validation--First-Ready-success.svg?style=flat-square)]()
[![Colab-Ready](https://img.shields.io/badge/Colab-Compatible-orange.svg?style=flat-square)]()

A validation-first multimodal AI system for product data understanding across images, text, and structured metadata.

---

## Project Mission

Real-world product catalogs are naturally multimodal, containing visual identities, semantic text descriptions, and structured metadata. However, training models on heterogeneous data sources is notoriously fragile.

This project is built around a single core directive: **making multimodal data trustworthy before training**. Rather than focusing solely on downstream model optimization, we construct a disciplined infrastructure system that establishes rigid contract enforcement between raw storage, CPU data loaders, and representation manifolds.

---

## Current Project Status

The project is currently in its **Infrastructure and Validation Phase**. The system is fully executable, verified, and ready for model training orchestration.

### Active Foundations
* **Multimodal Data Orchestration:** Synchronized data loaders running under strict resource constraints.
* **Contract Validation:** Stage-by-stage check gates asserting shape, dtype, and identity consistency across all modalities.
* **Encoder Integration:** Modular representation networks configured as stable, device-agnostic projection interfaces.
* **Global Verification:** A read-only verification suite that enforces contract safety across both local environments and remote instances.

### Upcoming Phases
* **Training & Optimization:** Optimizer scheduling, loss formulation, and checkpoint orchestration.
* **Retrieval & Indexing:** Latent space indexing for near-instantaneous multimodal similarity search.
* **Explainability Tools:** Proving visual and textual attention alignments to track features during inference.

---

## Core Engineering Philosophy

* **Validation Before Training:** Shape, range, and type validation execute at import and batch-construction time—preventing costly silent failures during GPU runtime.
* **Infrastructure Before Metrics:** Focus on reproducible pipeline components, explicit error-handling boundaries, and configuration safety over metric tuning.
* **Strict Ownership Boundaries:** Each subsystem operates as an independent contract authority. Tabular layers, tokenizers, and image transforms are decoupled to keep execution workers lightweight.
* **Identity Preservation:** The Amazon Standard Identification Number (ASIN) serves as the primary cryptographic key, preserving synchronization across modalities without physical file mutations.
* **Fail-Loud Architecture:** Unexpected format drifts or tensor alignment errors fail immediately. Fallbacks are restricted to safe, recoverable image omissions.
* **Local & Colab Parity:** Dynamically resolved root pathing guarantees absolute parity between local terminal runs and remote Jupyter runtimes.

---

## System Architecture

```text
configs/paths.py  (Centralized Filesystem Authority)
        ↓
dataset_registry.py  (CSV/Schema Discovery & Audit)
        ↓
dataset.py  (Synchronized Modality Sampling)
        ↓
transforms.py + tokenization.py  (Deterministic Preprocessing)
        ↓
collate.py  (Safe Batch Packing & Contract Enforcement)
        ↓
dataloader_factory.py  (Multiprocessing Worker Scheduling)
        ↓
image_encoder.py + text_encoder.py + tabular_encoder.py  (Representation Encoders)
        ↓
fusion.py  (Manifold Projection & Fusion Predictor)
        ↓
Future Training Orchestration
```

### Subsystem Roles

* **`configs/paths.py`:** Governs environment-aware path routing and offline model cache locations.
* **`data_pipeline/dataset_registry.py`:** Scans filesystem resources, executes schema validation, and compiles coverage reports.
* **`data_pipeline/dataset.py`:** Resolves visual, text, and metadata streams into synchronized, single-sample dictionaries.
* **`data_pipeline/transforms.py`:** Handles image decoding, RGB standardization, and resizing with strict aspect ratio preservation.
* **`data_pipeline/tokenization.py`:** Sanitizes text and structures sequence tokens on the CPU.
* **`data_pipeline/collate.py`:** Packs batches, validates sequence padding, and preserves trace metadata.
* **`data_pipeline/dataloader_factory.py`:** Schedules PyTorch loaders using host-optimized prefetching and worker configurations.
* **`models/*.py`:** Projection networks mapping visual representations, language sequences, and structured metadata into a shared latent space.
* **`models/fusion.py`:** Projects multi-source representations into a unified prediction space.
* **`validate_project.py`:** Coordinates end-to-end staged system audits.

---

## Validation-First Infrastructure

To ensure operational stability, the repository isolates testing into two distinct layers:
* **Local Smoke Tests:** Embedded directly within modules to allow developers to verify components in isolation.
* **Global Validator (`validate_project.py`):** An orchestrator checking environment settings, path structures, and model input invariants.

### Testing Command Reference

The validator runs as a read-only audit tool, guaranteeing no modifications to the working directory:

```bash
# Verify environment dependencies and basic path routing
python validate_project.py --quick

# Audit dataset construction, dataloader prefetching, and model forward passes
python validate_project.py --full

# Run all local subsystem tests as parallel subprocesses
python validate_project.py --full --run-smoke

# Output the complete validation suite report to standard output in JSON format
python validate_project.py --quick --json

# Save the structured validation report to a specified file
python validate_project.py --quick --json-out validation_report.json
```

### Validator Output Semantics

* `[PASS]`: Subsystem contract is verified and correct.
* `[EXPECTED]`: Security guards (such as path traversal blocks or duplicate identity detections) triggered correctly.
* `[WARN]`: Non-blocking operation concern (such as missing CUDA hardware in a CPU debugging run).
* `[SKIP]`: Heavier check bypassed in quick mode.
* `[FAIL]`: Critical contract failure requiring intervention.

```text
[ Screenshot Placeholder - Global Validation Report ]
```

---

## Dataset Routing & Security

The dataset layer is designed as an active security and validation trust boundary:
* **Dynamic Auditing:** Datasets are discovered and cataloged automatically without hardcoded paths.
* **Synchronized Sampling:** Tabular data and images are matched dynamically via ASIN hashes. The system avoids massive physical database joins that degrade scalability.
* **Path Traversal Protection:** Image path resolution uses directory limits to catch and block nested escape sequences.
* **Leakage Prevention:** Configs containing duplicate entries (e.g. `all_discovered`) raise clear errors, ensuring evaluation subsets remain pure.

---

## Model Representation Contracts

Models are constructed as rigid representation interfaces that enforce dimensionality boundaries:
* **Vision Representation Encoder:** Employs a projection head over a deep visual backbone, asserting shape, floating-point type, and NaN/Inf guards.
* **Language Representation Encoder:** Projects textual inputs using an underlying language model, freezing features during initial steps to stabilize warm-up phases.
* **Tabular Representation Encoder:** Maps metadata features into alignment-ready embeddings.
* **Multimodal Fusion Layer:** Combines projected modalities into a unified manifold space, validating dimension constraints at instantiation.

```text
[ Screenshot Placeholder - Model Contract Smoke Tests ]
```

---

## Long-Term System Direction

While currently focused on robust validation pipelines, this repository is designed as the foundation for a larger multimodal product intelligence engine:

* **Dense Product Retrieval:** Indexing shared embeddings for visual and textual search, enabling fast catalog exploration.
* **Multimodal Recommendations:** Modeling implicit relationships between items based on combined visual style, metadata similarity, and text context.
* **Explainable Classifications:** Visualizing alignment maps to show exactly which product features (e.g., visual regions or textual keywords) drive predictions.
* **High-Throughput Serving:** Transitioning data pipelines to dedicated caching daemons for ultra-low latency model deployment.

---

## How to Run

### 1. Installation

Set up your virtual environment and install the required dependencies:

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Remote Run Parity

When executing on Google Colab, paths automatically configure to your drive storage once mounted:

```python
from google.colab import drive
drive.mount("/content/drive")
```

### 3. Verification

Run the full system contract verification suite:

```bash
python validate_project.py --full --run-smoke
```

### 4. Running Subsystem Tests

To run local smoke tests directly:

```bash
# Direct module execution
python data_pipeline/transforms.py
python models/fusion.py

# Running via module routing
python -m data_pipeline.dataloader_factory
python -m models.image_encoder
```

---

## Repository Structure

```text
multi-model-ai/
├── checkpoints/              # Saved model weights
├── configs/
│   └── paths.py              # Centralized path authority
├── data_analysis/            # Notebooks for profiling and EDA
├── data_pipeline/
│   ├── collate.py            # Collator and sequence padding
│   ├── dataset.py            # Multimodal dataset loader
│   ├── dataset_registry.py   # Dataset discovery and schema checks
│   ├── dataloader_factory.py # PyTorch DataLoader orchestration
│   ├── tokenization.py       # Text sanitization and token processing
│   └── transforms.py         # Image preprocessing
├── experiments/              # Local experiment parameters
├── logs/                     # System validation and runtime logs
├── models/
│   ├── fusion.py             # Multimodal projection and fusion layer
│   ├── image_encoder.py      # Vision representation encoder
│   ├── tabular_encoder.py    # Tabular representation encoder
│   └── text_encoder.py       # Language representation encoder
├── preprocessed-datasets/    # Discovered CSV datasets
├── preprocessing/            # Ingestion scripts and async downloaders
├── requirements.txt          # Package dependencies
├── validate_project.py       # Global validation orchestrator
└── README.md
```

---

## Related Problem Space

This repository addresses themes common to standard multimodal architectures:
* **Contrastive Representation Learning:** Aligning visual and linguistic representations across shared latent manifolds.
* **Multimodal Catalogs:** Fusing visual catalogs and metadata details to capture complex product properties.
* **Data Ingestion Architectures:** Building scalable preprocessing workers capable of high-throughput data loading.
* **Product Catalog Embeddings:** Mapping text and image metadata to represent semantic attributes in catalogs.

---

## System Engineering Differentiators

* **No Notebook Dependency:** Fully structured in modular Python classes, ensuring clean tracebacks and execution safety.
* **Validation at the Core:** Subsystem states are audited dynamically, verifying environment compatibility on CPU, local GPU, and Google Colab backends.
* **Resource Optimization:** Dataloader scheduling uses dynamic worker allocations based on the host operating system.
* **Contract Security:** Encoders validate shape, type, NaN, and Inf bounds prior to forward execution.

---

This project is developed as a long-term multimodal AI infrastructure system. The focus is on building the surrounding engineering foundation required for trustworthy multimodal learning.

Contributions, reviews, and engineering discussions are welcome.
