# Multi-Model AI: Multimodal Product Intelligence

This repository builds an end-to-end system for learning from product images, descriptions, and structured catalog data, then predicting a product rating. The interesting problem is not simply putting three encoders in one model: it is preserving product identity and reliable tensor contracts as data moves from large raw catalog files through CPU workers, model training, checkpoints, and inference.

The project includes data preparation and analysis tools, a worker-aware multimodal input pipeline, a gated fusion model, a configurable training framework, and a checkpoint-backed prediction interface. It is designed for local development and Google Colab GPU runs.

## Why This Project Exists

Product catalogs are multimodal by nature. A title can describe an item, an image can reveal visual properties absent from the text, and tabular fields provide structured signals such as price and rating count. These sources arrive in different formats, may be incomplete, and can become misaligned if rows or files are handled carelessly.

This project treats data identity and subsystem boundaries as first-class engineering concerns. An ASIN anchors a product across its CSV row and image file. Each stage validates its inputs and outputs, and missing image or text data follows explicit fallback behavior. The result is a pipeline that can be inspected and exercised before committing to an expensive training run.

## Current Capabilities

### Data preparation and exploration

`preprocessing/data_preprocessing.py` reads Amazon metadata JSONL files as a stream, extracts product identifiers, title and description text, image URLs, price, rating count, category, and average rating, then writes a consistent CSV schema. The current script samples at most 1,600 rows per source using a fixed seed. The CSVs can be analyzed with the text, image, and tabular workflows in `data_analysis/` and the matching notebooks in `Colab-runs/`.

`preprocessing/image_downloader.py` downloads product images asynchronously in bounded batches, validates image responses, converts images to RGB, resizes oversized images, writes JPEGs named by ASIN, and synchronizes successful local paths into the CSV.

### Runtime data pipeline

The runtime path is separate from offline preprocessing. `data_pipeline/dataset_registry.py` discovers and audits processed CSV sources, checks the shared schema, measures image coverage, and supports resolving one or multiple sources. `data_pipeline/dataset.py` creates one synchronized sample at a time: it loads and transforms an image, sanitizes and tokenizes text, extracts numeric features, validates the target, and retains sample identity and optional trace metadata.

`data_pipeline/transforms.py` owns image decoding, RGB handling, train/evaluation transforms, fallback images, and tensor checks. `data_pipeline/tokenization.py` owns text sanitization and MiniLM tokenization. These CPU-side operations are kept outside model forward passes so dataset workers can prepare samples in parallel.

`data_pipeline/collate.py` validates sample identity and modality contracts, stacks sample tensors into contiguous CPU batches, and assembles bounded diagnostic metadata. `data_pipeline/dataloader_factory.py` wraps the dataset and collator in PyTorch `DataLoader`, selects worker settings by execution mode and environment, and configures optional pinned memory, prefetching, persistent workers, worker seeding, resource-risk reports, and an optional loader probe. The DataLoader performs the pinning and scheduling; the collator returns CPU batches ready for that handoff.

### Model and training

The three encoders project into a shared 512-dimensional embedding space:

| Modality | Implementation | Input contract |
| --- | --- | --- |
| Image | ConvNeXt Tiny backbone with a projection head | RGB image tensor, normally `(B, 3, 224, 224)` |
| Text | `sentence-transformers/all-MiniLM-L6-v2` with attended-token pooling and projection | `input_ids` and `attention_mask`, normally `(B, 64)` |
| Tabular | Configurable multilayer perceptron | Numeric feature tensor; the current dataset supplies `price` and `rating_number` |

`models/fusion.py` uses learned modality gates, modality dropout during training, a projection path, normalization, and a scalar rating-regression head. The current fusion contract expects the three modality embeddings to have compatible dimensions; the provided encoders default to 512 dimensions.

The `training/` package contains the training configuration, runtime context, optimizer and scheduler construction, evaluation, trainer, and application entry point. `training/train.py` assembles the dataset registry, dataset selection and split, loaders, model bundle, and training stack. It offers plan-only and one-batch dry-run modes before full training. The trainer handles train/evaluation mode, device transfer, optional mixed precision, gradient accumulation and clipping, metrics, run history, checkpoints, and resuming. Checkpoint and run outputs are generated locally and are not committed.

### Prediction

`predict.py` loads a trained experiment checkpoint and accepts one product at a time using an image path or image URL, text, price, and rating count. It validates and preprocesses the inputs, executes the encoders and fusion model, and returns a rating clipped to the supported 1–5 range along with checkpoint and runtime metadata. The CLI supports direct and interactive use, optional JSON history, and an optional desktop popup.

The `--explain` option currently calls a placeholder: it reports that no explanation backend is configured. It does not calculate feature attribution, highlight image regions, or identify influential words. Explainability and retrieval are future directions, not implemented capabilities.

## Data and Path Setup

Raw data, processed CSVs, downloaded images, model caches, logs, and checkpoints are local runtime assets. They are excluded from Git, so clone users must provide the data and train or obtain a compatible checkpoint themselves.

The processed CSV schema used by the runtime pipeline includes `asin`, `text`, `image_url`, `price`, `rating_number`, `category`, and `rating`. Dataset and model inputs currently use `price` and `rating_number` as the numeric tabular features; `category` is retained in the prepared data and registry but is not currently encoded into the tabular model input. Images are resolved from an `image_path` when present, with ASIN-based image lookup as a fallback.

`configs/paths.py` resolves the project root using `MULTI_MODEL_AI_ROOT`, known Colab locations, the local development location, and finally the location of the config file. In Colab, mount Drive before importing project modules if your data is stored there:

```python
from google.colab import drive
drive.mount("/content/drive")
```

The older offline preprocessing and downloader scripts still define source and output paths in their configuration sections. Review those paths for your machine before running them. The runtime training pipeline uses the centralized path helpers.

## Installation

Use Python 3.10 or newer and install the project requirements in a virtual environment. PyTorch's installation instructions can help choose a build that matches your CUDA runtime; CPU-only use is also supported for checks and smaller workloads.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with `source venv/bin/activate` instead. The requirements file covers the model and runtime data pipeline. The exploratory analysis scripts and notebooks may need additional packages such as Matplotlib, Seaborn, SciPy, scikit-learn, or NLTK, depending on which analysis you run.

## Running the Project

Run commands from the repository root. Start with validation and a training plan before running the full training workload:

```bash
# Fast project and contract checks
python validate_project.py --quick

# Broader checks; --run-smoke also launches module smoke suites
python validate_project.py --full
python validate_project.py --full --run-smoke

# Training framework checks and execution planning
python training/train.py --smoke
python training/train.py --plan-only

# Build the plan, run preflight, and process a batch without training
python training/train.py --dry-run

# Run training with the current training configuration
python training/train.py --train

# Predictor checks, then interactive prediction
python predict.py --smoke
python predict.py --interactive
```

`--plan-only` reports the selected sources, splits, execution settings, and preflight results. `--dry-run` additionally checks a batch through the training path. Full training can consume substantial GPU time and memory; it depends on the local processed dataset and the configuration assembled by `training/train.py`.

### Direct prediction

Provide an experiment name, exactly one image source, text, price, and rating count. Checkpoints are resolved under `checkpoints/<experiment>/`; `best.pt` is the default, with `latest.pt` used as a fallback when the default best checkpoint is absent.

```bash
python predict.py \
  --experiment my_exp \
  --image "D:/multi-model-ai/preprocessed-datasets/images/B001.jpg" \
  --text "Wireless headphones" \
  --price 2499 \
  --rating-number 318
```

For a URL image, replace `--image` with `--image-url "https://example.com/product.jpg"`. You can select a different checkpoint with `--checkpoint latest.pt`, choose a device with `--device cpu` or `--device cuda`, and request a prediction history record with `--save-history`.

## How the Runtime Handoff Works

```text
processed CSV + ASIN-named images
              |
              v
       Dataset workers       image loading, transforms, tokenization, features
              |
              v
          Collator            CPU tensor stacking + identity/shape checks
              |
              v
          DataLoader          worker scheduling, prefetch, optional pinning
              |
              v
       Trainer / train.py     device transfer, model execution, loss, updates
              |
              v
 Encoders -> gated fusion -> predicted rating -> versioned checkpoint
              |
              v
          predict.py          one-product inference from a saved checkpoint
```

Dataset and collate work remains CPU-side. When enabled, DataLoader pinning prepares CPU tensors for transfer; training code owns device movement and model execution. Worker counts and prefetch settings affect CPU, memory, and I/O load, so the loader factory exposes `debug`, `safe`, `colab`, and `throughput` execution modes and reports risks rather than assuming every host can sustain the same settings.

## Repository Map

```text
configs/
  paths.py                    Project-root and filesystem path authority
data_pipeline/
  dataset_registry.py          Processed CSV discovery and source auditing
  dataset.py                   Synchronized multimodal sample construction
  transforms.py                Image loading and transforms
  tokenization.py              Text cleaning and tokenization
  collate.py                   Batch contracts and CPU tensor collation
  dataloader_factory.py        Worker, prefetch, pinning, and loader setup
models/
  image_encoder.py             ConvNeXt image representation
  text_encoder.py              MiniLM text representation
  tabular_encoder.py           Numeric metadata representation
  fusion.py                    Gated fusion and rating prediction
training/
  train_config.py              Validated training configuration
  run_context.py               Resolved runtime and run metadata
  optimizer.py                 Optimizer construction
  scheduler.py                 Learning-rate scheduling
  evaluation.py                Loss, metrics, and validation results
  trainer.py                   Training loop, checkpoints, and resume
  train.py                     End-to-end training bootloader
preprocessing/                 Offline metadata and image preparation
data_analysis/                 Text, image, and tabular dataset exploration
Colab-runs/                    Colab analysis and prediction notebooks
predict.py                     Checkpoint-backed single-product prediction
validate_project.py            Global project validation entry point
```

## What Comes Next

The working system now spans data preparation, validation, multimodal training, checkpoint recovery, and prediction. The next research-facing steps are to measure the value each modality contributes, evaluate on held-out product sources, and make prediction quality easier to inspect. A retrieval index over learned embeddings and a real explainability backend are natural extensions, but should be built and evaluated as distinct capabilities. Until then, the current predictor provides rating output and run metadata rather than causal or feature-level explanations.

## Contributing

Changes are easiest to review when they preserve the existing ownership boundaries: preprocessing prepares files, the data pipeline builds CPU samples and batches, the DataLoader manages worker and memory scheduling, encoders create modality embeddings, fusion combines them, and the training or prediction entry points orchestrate execution. Start with `python validate_project.py --quick` when changing contracts, then use the relevant smoke or dry-run command for the affected subsystem.
