# 🚀 Multi-Modal AI Pipeline

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![AsyncIO](https://img.shields.io/badge/AsyncIO-Enabled-green?style=for-the-badge)
![aiohttp](https://img.shields.io/badge/aiohttp-Async%20Networking-orange?style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Engineering-black?style=for-the-badge&logo=pandas)
![Pillow](https://img.shields.io/badge/Pillow-Image%20Processing-red?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

---

# 🧠 Overview

A scalable multimodal AI preprocessing pipeline designed for handling large-scale **Image + Text + Tabular** datasets efficiently.

This project focuses not only on model building, but also on:

- scalable data engineering
- asynchronous systems
- memory optimization
- production-oriented preprocessing
- future multimodal AI architecture

The long-term goal is to build a complete multimodal AI system capable of integrating:

- CNN-based image understanding
- text embeddings
- tabular feature learning
- explainable AI
- future LLM + RAG integration

---

# 🔥 Key Engineering Challenges Solved

## ✅ Large Dataset Memory Bottleneck

Initial preprocessing failed with:

```python
MemoryError
```

because large `.jsonl` Amazon metadata files were being loaded fully into RAM.

### ✔ Solution

Implemented chunk-based streaming using:

```python
pd.read_json(..., chunksize=...)
```

This transformed the preprocessing system into a scalable ingestion pipeline capable of handling datasets much larger than available memory.

---

## ✅ High-Speed Image Downloading

Downloading thousands of images sequentially was extremely slow.

### ✔ Solution

Built an asynchronous image downloading system using:

- `asyncio`
- `aiohttp`
- connection reuse
- semaphores
- batched task execution

This enabled scalable concurrent downloading while maintaining stable memory usage.

---

## ✅ Image Storage Optimization

Raw downloaded images consumed large storage space and introduced training inefficiencies.

### ✔ Solution

Implemented:

- RGB normalization
- smart resizing
- aspect-ratio preservation
- JPEG compression
- deterministic ASIN-based storage

Benefits:

- reduced storage usage
- faster training pipelines
- lower memory overhead
- consistent CNN-ready inputs

---

# 🏗️ Current Pipeline Architecture

```text
Raw Amazon Metadata
        ↓
Chunk-Based JSONL Processing
        ↓
Feature Extraction + Cleaning
        ↓
Structured CSV Datasets
        ↓
Async Image Download Pipeline
        ↓
Image Validation + Processing
        ↓
Local Image Storage
        ↓
CSV Synchronization
        ↓
Final Multimodal Dataset
```

---

# 📂 Project Structure

```text
multi-model-ai/
│
├── datasets/
│
├── preprocessing/
│   ├── data_preprocessing.py
│   └── image_downloader.py
│
├── preprocessed-datasets/
│
├── images/
│
├── README.md
└── .gitignore
```

---

# ⚙️ Features Implemented

## ✅ Data Preprocessing

- JSONL dataset parsing
- chunk-based processing
- feature extraction
- schema normalization
- missing-value handling
- CSV export pipeline

---

## ✅ Async Image Pipeline

- concurrent downloading
- shared HTTP session reuse
- retry handling
- timeout handling
- batch execution
- resumable downloads

---

## ✅ Image Processing

- image validation
- corrupted image detection
- RGB conversion
- smart resizing
- JPEG compression
- deterministic filename mapping

---

## ✅ Scalability Optimizations

- chunked dataset streaming
- memory-safe preprocessing
- batched async execution
- caching support
- duplicate download prevention

---

# 🖼️ Image Processing Details

## RGB Standardization

```python
image.convert("RGB")
```

Standardizes inconsistent image formats into CNN-compatible 3-channel RGB images.

---

## Smart Resizing

```python
Image.thumbnail((512, 512))
```

Only oversized images are resized while preserving aspect ratio.

Benefits:

- faster training
- lower storage usage
- reduced GPU memory usage

---

## Compression

```python
quality=85
optimize=True
```

Reduces storage requirements with minimal visual quality loss.

---

# 💾 Deterministic Image Storage

Images are stored using ASIN-based naming:

```text
B085DH1C9V.jpg
```

This avoids:

- filename collisions
- row-order dependency
- dataset desynchronization

ASIN acts as a stable primary key across:

- image data
- text data
- metadata
- prediction targets

---

# 🛡️ Edge Cases Handled

- missing image URLs
- HTTP 403 / 404 responses
- corrupted images
- invalid content types
- network timeouts
- duplicate downloads
- memory scalability limitations

---

# 🛠️ Technologies Used

| Area | Tools |
|---|---|
| Data Processing | pandas |
| Async Networking | asyncio, aiohttp |
| Image Processing | Pillow (PIL) |
| Progress Monitoring | tqdm |
| Deep Learning (planned) | PyTorch |
| Explainability (planned) | SHAP, GradCAM |

---

# 🧠 Key Learnings

This project provided hands-on experience with:

- scalable preprocessing systems
- asynchronous programming
- memory optimization strategies
- multimodal dataset architecture
- production-oriented ML engineering
- pipeline reliability and fault tolerance

---

# ⭐ Current Status

## ✅ Completed

- scalable preprocessing pipeline
- chunk-based dataset loading
- async image downloader
- image validation system
- RGB conversion
- smart resizing
- compression pipeline
- deterministic local image storage
- CSV synchronization
- resume/caching support

---

# 🔥 Final Result

A scalable multimodal AI dataset pipeline capable of handling:

- image data
- text data
- tabular metadata

while remaining:

- memory efficient
- resumable
- scalable
- future-ready for multimodal AI systems

---

# 👨‍💻 Author

Built as part of a deep exploration into:

```text
Scalable Multi-Modal AI Systems
```

with focus on:

- system architecture
- scalability
- optimization
- real-world AI engineering
