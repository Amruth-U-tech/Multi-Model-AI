# =============================================================================
# CELL 1 — Title & Project Info
# =============================================================================
# %% [markdown]
"""
# 🛒 Multi-Modal AI — Data Preprocessing
## Amazon Metadata: Multiple Categories (Scalable Pipeline)

**Pipeline Overview:**
- Load multiple raw JSON-lines files
- Extract & clean required fields (text, image, tabular) per file
- Build `text`, `image_url` columns
- Apply cleaning rules & sampling (max 1600 rows per file)
- Export one clean CSV per file — identical schema, directly mergeable

**Target Variable:** `rating` (last column, consistent across ALL outputs)
"""

# =============================================================================
# CELL 2 — Install & Import Dependencies
# =============================================================================
# %%
# ── Install any missing packages (uncomment if running on fresh Colab) ────────
# !pip install pandas numpy tqdm --quiet

import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# ── Reproducibility seed for random sampling ──────────────────────────────────
RANDOM_SEED = 42

print("✅ All libraries imported successfully.")
print(f"   pandas  version : {pd.__version__}")
print(f"   numpy   version : {np.__version__}")

# =============================================================================
# CELL 3 — Global Configuration (Single Source of Truth)
# =============================================================================
# %%

# ── Output directory (shared across ALL datasets) ─────────────────────────────
OUTPUT_DIR          = "D:/multi-model-ai/preprocessed-datasets"

# ── Dataset constraints (applied identically to every file) ──────────────────
MAX_ROWS            = 1600     # Cap for random sampling per dataset
FILL_RATING_NUMBER  = 0        # Fill value for missing rating_number

# ── Required source columns (before rename) — same for all Amazon metadata ───
SOURCE_COLUMNS = [
    "parent_asin",      # → asin
    "title",
    "description",
    "images",
    "price",
    "rating_number",
    "main_category",    # → category
    "average_rating",   # → rating
]

# ── Final output column order (rating MUST be last) — enforced on all outputs ─
FINAL_COLUMNS = [
    "asin",
    "text",
    "image_url",
    "price",
    "rating_number",
    "category",
    "rating",           # ← TARGET
]

# ── Input files to process ────────────────────────────────────────────────────
# Add or remove paths here to scale to new datasets — no other code needs change
DATASET_FILES = [
    "D:/multi-model-ai/datasets/datasets-meta/meta_Amazon_Fashion.jsonl",
    "D:/multi-model-ai/datasets/datasets-meta/meta_Cell_Phones_and_Accessories.jsonl",  # new file
    "D:/multi-model-ai/datasets/datasets-meta/meta_Digital_Music.jsonl",     #new file
]

print("✅ Global configuration loaded.")
print(f"   Output dir      : {OUTPUT_DIR}")
print(f"   Max rows/file   : {MAX_ROWS}")
print(f"   Files to process: {len(DATASET_FILES)}")
for f in DATASET_FILES:
    print(f"     • {f}")

# =============================================================================
# CELL 4 — Helper Functions
# =============================================================================
# %%

def build_text_column(row: pd.Series) -> Optional[str]:
    """
    Combine 'title' and 'description' into a single text string.

    Rules:
      - If description is a list  → join all elements with a space
      - If description is a str   → use as-is
      - If description is missing → use title only
      - Returns None if title is also missing/empty

    Args:
        row: A DataFrame row with 'title' and 'description' fields.

    Returns:
        Combined text string or None.
    """
    title = row.get("title", None)
    desc  = row.get("description", None)

    # ── Normalise title ───────────────────────────────────────────────────────
    if not isinstance(title, str) or not title.strip():
        title = None

    # ── Normalise description ─────────────────────────────────────────────────
    if isinstance(desc, list):
        # Filter out empty strings inside the list before joining
        desc = " ".join([d for d in desc if isinstance(d, str) and d.strip()])
        if not desc.strip():
            desc = None
    elif isinstance(desc, str):
        desc = desc.strip() if desc.strip() else None
    else:
        desc = None

    # ── Combine ───────────────────────────────────────────────────────────────
    if title and desc:
        return f"{title} {desc}"
    elif title:
        return title
    elif desc:
        return desc
    else:
        return None     # Both missing → will be dropped downstream


def extract_image_url(images_field) -> Optional[str]:
    """
    Extract the first valid 'large' image URL from the images field.

    Expected format: list of dicts, e.g. [{"large": "https://...", ...}, ...]

    Args:
        images_field: Raw value from the 'images' column.

    Returns:
        URL string or None if unavailable.
    """
    if not isinstance(images_field, list) or len(images_field) == 0:
        return None

    first_image = images_field[0]

    if not isinstance(first_image, dict):
        return None

    url = first_image.get("large", None)

    # Validate it is a non-empty string
    return url if isinstance(url, str) and url.strip() else None


def generate_output_path(input_path: str, output_dir: str) -> str:
    """
    Derive a unique output CSV path from the input JSONL filename.

    Naming rule: <original_filename_lowercase_no_ext>_processed.csv
    Example: meta_Handmade_Products.jsonl → meta_handmade_products_processed.csv

    Args:
        input_path : Absolute path to the source .jsonl file.
        output_dir : Directory where the output CSV will be saved.

    Returns:
        Full absolute path to the output CSV file.
    """
    base_name      = os.path.basename(input_path)          # meta_Handmade_Products.jsonl
    name_no_ext    = os.path.splitext(base_name)[0]        # meta_Handmade_Products
    name_lower     = name_no_ext.lower()                   # meta_handmade_products
    output_name    = f"{name_lower}_processed.csv"         # meta_handmade_products_processed.csv
    return os.path.join(output_dir, output_name)


print("✅ Helper functions defined:")
print("   - build_text_column(row)")
print("   - extract_image_url(images_field)")
print("   - generate_output_path(input_path, output_dir)")

# =============================================================================
# CELL 5 — Core Pipeline: process_dataset()
# =============================================================================
# %%

def process_dataset(input_path: str, output_dir: str) -> bool:
    """
    Run the full preprocessing pipeline for a single JSONL dataset file.

    Steps (original logic preserved exactly):
      1.  Validate input file exists
      2.  Load JSON-lines file
      3.  Select required source columns
      4.  Rename columns
      5.  Build 'text' column (title + description)
      6.  Extract 'image_url' from images list
      7.  Clean data (drop nulls, fill medians, deduplicate)
      8.  Sample to MAX_ROWS if needed
      9.  Reorder to FINAL_COLUMNS schema
      10. Validate (no nested, rating numeric, rating last)
      11. Save output CSV

    Args:
        input_path : Absolute path to source .jsonl file.
        output_dir : Directory to write the processed CSV into.

    Returns:
        True  if file was processed and saved successfully.
        False if file was skipped due to an error.
    """

    # ── Derive output path from input filename ────────────────────────────────
    output_path  = generate_output_path(input_path, output_dir)
    dataset_name = os.path.basename(input_path)

    print("\n" + "=" * 60)
    print(f"  🗂️  Processing: {dataset_name}")
    print("=" * 60)

    # =========================================================================
    # EDGE CASE 1 — Input file does not exist: skip and log
    # =========================================================================
    if not os.path.exists(input_path):
        print(f"❌ [SKIP] File not found — path does not exist:")
        print(f"   → {input_path}")
        return False

    # =========================================================================
    # CELL 5 (original) — Load Raw JSON-Lines File
    # =========================================================================

    print(f"\n📂 Loading raw data from: {input_path}")
    print("   (This may take a moment for large files...)\n")

    # ── Safe load with pd.read_json lines=True ────────────────────────────────
    # EDGE CASE 2 — JSON parse failure: skip and log
    try:
        df_raw = pd.read_json(input_path, lines=True)
    except FileNotFoundError:
        print(f"❌ [SKIP] File disappeared between check and read:")
        print(f"   → {input_path}")
        return False
    except ValueError as e:
        print(f"❌ [SKIP] Failed to parse JSON file. Ensure it is valid JSON-lines format.")
        print(f"   Error: {e}")
        return False

    print(f"✅ Raw data loaded successfully.")
    print(f"   Shape          : {df_raw.shape}")
    print(f"   Columns present: {list(df_raw.columns)}")

    rows_before_cleaning = len(df_raw)      # ← logged at end

    # =========================================================================
    # CELL 6 (original) — Select Required Source Columns
    # =========================================================================

    # ── Keep only the source columns we need ──────────────────────────────────
    # Drop any column from SOURCE_COLUMNS that doesn't exist in the file (safety)
    existing_source_cols = [c for c in SOURCE_COLUMNS if c in df_raw.columns]
    missing_source_cols  = [c for c in SOURCE_COLUMNS if c not in df_raw.columns]

    if missing_source_cols:
        print(f"⚠️  Warning: The following expected columns are MISSING from raw file:")
        for col in missing_source_cols:
            print(f"      - {col}")

    df = df_raw[existing_source_cols].copy()

    print(f"\n✅ Selected {len(existing_source_cols)} columns from raw data.")
    print(f"   Columns kept: {existing_source_cols}")
    print(f"   Shape        : {df.shape}")

    # =========================================================================
    # CELL 7 (original) — Rename Columns (parent_asin → asin, etc.)
    # =========================================================================

    COLUMN_RENAME_MAP = {
        "parent_asin"    : "asin",
        "main_category"  : "category",
        "average_rating" : "rating",
    }

    df.rename(columns=COLUMN_RENAME_MAP, inplace=True)

    print("✅ Columns renamed:")
    for old, new in COLUMN_RENAME_MAP.items():
        if old in existing_source_cols:
            print(f"   '{old}'  →  '{new}'")

    print(f"\n   Current columns: {list(df.columns)}")

    # =========================================================================
    # CELL 8 (original) — Build 'text' Column (Title + Description)
    # =========================================================================

    print("🔧 Building 'text' column from title + description...")

    # Apply vectorised-safe row-wise function with tqdm progress bar
    tqdm.pandas(desc="Building text")
    df["text"] = df.progress_apply(build_text_column, axis=1)

    # Stats
    n_null_text = df["text"].isna().sum()
    print(f"\n✅ 'text' column created.")
    print(f"   Total rows   : {len(df)}")
    print(f"   Non-null text: {df['text'].notna().sum()}")
    print(f"   Null text    : {n_null_text}  ← will be dropped in cleaning step")

    # =========================================================================
    # CELL 9 (original) — Build 'image_url' Column
    # =========================================================================

    print("🔧 Extracting 'image_url' from images field...")

    # Safe extraction — handles missing / malformed images gracefully
    tqdm.pandas(desc="Extracting image_url")
    df["image_url"] = df["images"].progress_apply(extract_image_url)

    n_null_img = df["image_url"].isna().sum()
    print(f"\n✅ 'image_url' column created.")
    print(f"   Total rows      : {len(df)}")
    print(f"   Valid image URLs: {df['image_url'].notna().sum()}")
    print(f"   Missing URLs    : {n_null_img}  ← will be dropped in cleaning step")

    # Drop intermediate 'images' and raw text columns no longer needed
    df.drop(columns=["images", "title", "description"], errors="ignore", inplace=True)

    print(f"\n   Dropped intermediate columns: ['images', 'title', 'description']")
    print(f"   Current columns: {list(df.columns)}")

    # =========================================================================
    # CELL 10 (original) — Data Cleaning
    # =========================================================================

    print("🧹 Starting data cleaning...\n")
    initial_row_count = len(df)

    # ── Step 1: Drop rows with null in critical columns ───────────────────────
    critical_cols = ["image_url", "text", "rating"]
    before_drop = len(df)
    df.dropna(subset=critical_cols, inplace=True)
    after_drop = len(df)
    print(f"   [Drop nulls]  Removed {before_drop - after_drop} rows "
          f"(image_url / text / rating is null).")

    # ── Step 2: Fill missing numeric values ───────────────────────────────────
    # price → median of remaining column (computed AFTER null drop for accuracy)
    price_median = df["price"].median()
    df["price"].fillna(price_median, inplace=True)
    print(f"   [Fill price]  Missing prices filled with median = {price_median:.2f}")

    # rating_number → 0 (unknown review count)
    n_missing_rn = df["rating_number"].isna().sum()
    df["rating_number"].fillna(FILL_RATING_NUMBER, inplace=True)
    print(f"   [Fill rating_number]  Filled {n_missing_rn} missing values with {FILL_RATING_NUMBER}")

    # ── Step 3: Remove duplicates on 'asin' ──────────────────────────────────
    before_dedup = len(df)
    df.drop_duplicates(subset=["asin"], keep="first", inplace=True)
    after_dedup = len(df)
    print(f"   [Dedup asin]  Removed {before_dedup - after_dedup} duplicate ASIN rows.")

    print(f"\n✅ Cleaning complete.")
    print(f"   Rows before cleaning : {initial_row_count}")
    print(f"   Rows after  cleaning : {len(df)}")

    rows_after_cleaning = len(df)      # ← logged at end

    # =========================================================================
    # EDGE CASE 3 — Dataset empty after cleaning: do NOT save, log warning
    # =========================================================================
    if len(df) == 0:
        print(f"\n⚠️  [SKIP] Dataset is EMPTY after cleaning — no CSV will be saved.")
        print(f"   File: {dataset_name}")
        return False

    # =========================================================================
    # CELL 11 (original) — Sampling (Cap at MAX_ROWS)
    # =========================================================================

    current_rows = len(df)
    print(f"📊 Row count before sampling: {current_rows}")

    if current_rows > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=RANDOM_SEED)
        print(f"   Randomly sampled {MAX_ROWS} rows (seed={RANDOM_SEED}).")
    else:
        print(f"   Row count ≤ {MAX_ROWS} — keeping all {current_rows} rows.")

    # ── Reset index for clean sequential indexing ─────────────────────────────
    df.reset_index(drop=True, inplace=True)
    print(f"\n✅ Final row count after sampling: {len(df)}")

    rows_after_sampling = len(df)      # ← logged at end

    # =========================================================================
    # CELL 12 (original) — Reorder Columns to Final Schema
    # =========================================================================

    # ── Only include columns that actually exist in df ────────────────────────
    available_final_cols = [c for c in FINAL_COLUMNS if c in df.columns]
    missing_final_cols   = [c for c in FINAL_COLUMNS if c not in df.columns]

    if missing_final_cols:
        print(f"⚠️  Warning: The following expected final columns are missing: {missing_final_cols}")

    df = df[available_final_cols]

    print("✅ Columns reordered to final schema:")
    for i, col in enumerate(df.columns, 1):
        target_marker = "  ← TARGET" if col == "rating" else ""
        print(f"   {i}. {col}{target_marker}")

    # =========================================================================
    # CELL 13 (original) — Final Validation
    # =========================================================================

    print("🔍 Running final validation checks...\n")

    validation_passed = True
    issues = []

    # ── Check 1: No nested structures (all values must be flat) ───────────────
    for col in df.columns:
        has_nested = df[col].apply(lambda x: isinstance(x, (list, dict))).any()
        if has_nested:
            issues.append(f"   ❌ Column '{col}' contains nested structures (list/dict).")
            validation_passed = False

    # ── Check 2: 'rating' is numeric ─────────────────────────────────────────
    if not pd.api.types.is_numeric_dtype(df["rating"]):
        issues.append("   ❌ 'rating' column is NOT numeric — coercing...")
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        # Drop any rows where coercion produced NaN
        df.dropna(subset=["rating"], inplace=True)
        issues.append(f"   ⚠️  'rating' coerced to numeric. Rows dropped: {len(df)}")
        validation_passed = False

    # ── Check 3: 'rating' is the last column ─────────────────────────────────
    if df.columns[-1] != "rating":
        issues.append("   ❌ 'rating' is NOT the last column — reordering...")
        cols = [c for c in df.columns if c != "rating"] + ["rating"]
        df = df[cols]
        validation_passed = False

    # ── Check 4: No remaining nulls in critical columns ───────────────────────
    for col in ["image_url", "text", "rating"]:
        if df[col].isna().any():
            issues.append(f"   ❌ Column '{col}' still has null values after cleaning.")
            validation_passed = False

    # ── Check 5: 'text' and 'image_url' are string type ──────────────────────
    for col in ["text", "image_url"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # ── Report ────────────────────────────────────────────────────────────────
    if validation_passed and not issues:
        print("✅ ALL VALIDATIONS PASSED — dataset is clean and ML-ready!\n")
    else:
        print("⚠️  Validation issues found and auto-corrected:\n")
        for issue in issues:
            print(issue)
        print()

    # ── Summary stats ─────────────────────────────────────────────────────────
    print("📋 Final Dataset Summary:")
    print(f"   Shape          : {df.shape}")
    print(f"   Columns        : {list(df.columns)}")
    print(f"   Null counts    :\n{df.isnull().sum().to_string()}")
    print(f"\n   Rating stats   :")
    print(df["rating"].describe().to_string())

    # =========================================================================
    # CELL 14 (original) — Save Output CSV
    # =========================================================================

    # ── EDGE CASE 4: Create output directory if it doesn't exist ─────────────
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📁 Output directory ready: '{output_dir}/'")

    # ── Prevent overwriting existing file ────────────────────────────────────
    if os.path.exists(output_path):
        print(f"⚠️  Output file already exists — overwriting: {output_path}")

    # ── Save to CSV ───────────────────────────────────────────────────────────
    df.to_csv(output_path, index=False)

    # ── Verify file was written correctly ─────────────────────────────────────
    saved_size_kb = os.path.getsize(output_path) / 1024
    df_verify     = pd.read_csv(output_path, nrows=2)  # Quick verify read

    print(f"\n✅ Dataset saved successfully!")
    print(f"   Path    : {output_path}")
    print(f"   Size    : {saved_size_kb:.2f} KB")
    print(f"   Rows    : {len(df)}")
    print(f"   Columns : {list(df.columns)}")
    print(f"\n   Preview (first 2 rows):")
    print(df_verify.to_string(index=False))

    # =========================================================================
    # CELL 15 (original) — Quick EDA Summary
    # =========================================================================

    print("=" * 60)
    print("  📊  PRE-TRAINING DATA SANITY CHECK")
    print("=" * 60)

    print(f"\n📌 Total samples        : {len(df)}")
    print(f"📌 Feature columns      : {len(df.columns) - 1}")
    print(f"📌 Target column        : 'rating'")
    print(f"📌 Rating range         : {df['rating'].min():.1f} — {df['rating'].max():.1f}")
    print(f"📌 Rating mean          : {df['rating'].mean():.4f}")
    print(f"📌 Rating std           : {df['rating'].std():.4f}")
    print(f"📌 Unique categories    : {df['category'].nunique()}")
    print(f"📌 Rows with price=0    : {(df['price'] == 0).sum()}")
    print(f"📌 Text avg length (chars): {df['text'].str.len().mean():.0f}")
    print(f"📌 Null count (any col) : {df.isnull().sum().sum()}")

    print("\n📌 Category distribution:")
    print(df["category"].value_counts().to_string())

    print("\n🚀 Data is ready for:")
    print("   ✅ Text encoding  (NLP / BERT / TF-IDF)")
    print("   ✅ Image loading  (URL → PIL → tensor)")
    print("   ✅ Tabular input  (price, rating_number, category)")
    print("   ✅ Supervised ML  (target: 'rating')")

    # =========================================================================
    # STEP 7 — Per-dataset log summary (returned to caller loop)
    # =========================================================================
    print(f"\n📝 Dataset log — {dataset_name}")
    print(f"   Rows loaded          : {rows_before_cleaning}")
    print(f"   Rows after cleaning  : {rows_after_cleaning}")
    print(f"   Rows after sampling  : {rows_after_sampling}")
    print(f"   Saved to             : {output_path}")

    return True     # Signal success to the caller loop


print("✅ process_dataset() function defined and ready.")

# =============================================================================
# CELL 6 — Main Execution Loop (Process All Datasets)
# =============================================================================
# %%

print("\n" + "=" * 60)
print("  🚀  STARTING MULTI-DATASET PREPROCESSING")
print(f"  Total files : {len(DATASET_FILES)}")
print("=" * 60)

# ── Track results across all files ────────────────────────────────────────────
results = []   # list of dicts: {file, status, output_path}

for file_path in DATASET_FILES:
    success = process_dataset(
        input_path = file_path,
        output_dir = OUTPUT_DIR,
    )
    results.append({
        "file"       : os.path.basename(file_path),
        "status"     : "✅ success" if success else "❌ skipped",
        "output_path": generate_output_path(file_path, OUTPUT_DIR) if success else "—",
    })

# =============================================================================
# CELL 7 — Final Run Summary
# =============================================================================
# %%

print("\n" + "=" * 60)
print("  📋  PREPROCESSING RUN SUMMARY")
print("=" * 60)

n_success = sum(1 for r in results if "success" in r["status"])
n_skipped = len(results) - n_success

print(f"\n  Files processed : {n_success} / {len(results)}")
print(f"  Files skipped   : {n_skipped}")
print()

for r in results:
    print(f"  {r['status']}  {r['file']}")
    print(f"             → {r['output_path']}")

print("\n✅ All outputs share identical schema:")
print(f"   {FINAL_COLUMNS}")
print("\n🔗 CSVs are directly mergeable with pd.concat() for combined training.")
