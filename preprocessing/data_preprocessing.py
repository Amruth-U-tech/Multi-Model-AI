# =============================================================================
# CELL 1 — Title & Project Info
# =============================================================================
# %% [markdown]
"""
# 🛒 Multi-Modal AI — Data Preprocessing
## Amazon Metadata: `handmade_products` Category

**Pipeline Overview:**
- Load raw JSON-lines file
- Extract & clean required fields (text, image, tabular)
- Build `text`, `image_url` columns
- Apply cleaning rules & sampling (max 1600 rows)
- Export clean CSV ready for ML training

**Target Variable:** `rating` (last column)
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
# CELL 3 — Configuration (Single Source of Truth)
# =============================================================================
# %%
# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DATA_PATH       = "D:/multi-model-ai/datasets/datasets-meta/meta_Handmade_Products.jsonl"   # Input file
OUTPUT_DIR          = "D:/multi-model-ai/preprocessed-datasets"               # Output folder
OUTPUT_FILE_NAME    = "handmade_products_processed.csv"    # Output filename
OUTPUT_PATH         = os.path.join(OUTPUT_DIR, OUTPUT_FILE_NAME)

# ── Dataset constraints ───────────────────────────────────────────────────────
MAX_ROWS            = 1600     # Cap for random sampling
FILL_RATING_NUMBER  = 0        # Fill value for missing rating_number

# ── Required source columns (before rename) ────────────────────────────────────
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

# ── Final output column order (rating MUST be last) ───────────────────────────
FINAL_COLUMNS = [
    "asin",
    "text",
    "image_url",
    "price",
    "rating_number",
    "category",
    "rating",           # ← TARGET
]

# ── Early path validation — fail fast before loading anything ────────────────
assert os.path.exists(RAW_DATA_PATH), (
    f"❌ Dataset path is wrong or file does not exist:\n"
    f"   → {RAW_DATA_PATH}\n"
    f"   Please verify the path and try again."
)

print("✅ Configuration loaded.")
print(f"   Input  : {RAW_DATA_PATH}")
print(f"   Output : {OUTPUT_PATH}")
print(f"   Max rows after sampling: {MAX_ROWS}")

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


print("✅ Helper functions defined:")
print("   - build_text_column(row)")
print("   - extract_image_url(images_field)")

# =============================================================================
# CELL 5 — Load Raw JSON-Lines File
# =============================================================================
# %%

print(f"📂 Loading raw data from: {RAW_DATA_PATH}")
print("   (This may take a moment for large files...)\n")

# ── Safe load with pd.read_json lines=True ────────────────────────────────────
try:
    df_raw = pd.read_json(RAW_DATA_PATH, lines=True)
except FileNotFoundError:
    raise FileNotFoundError(
        f"❌ File not found: '{RAW_DATA_PATH}'\n"
        f"   Please place your dataset at the path above and re-run this cell."
    )
except ValueError as e:
    raise ValueError(f"❌ Failed to parse JSON file. Ensure it is valid JSON-lines format.\n{e}")

print(f"✅ Raw data loaded successfully.")
print(f"   Shape          : {df_raw.shape}")
print(f"   Columns present: {list(df_raw.columns)}")

# =============================================================================
# CELL 6 — Select Required Source Columns
# =============================================================================
# %%

# ── Keep only the source columns we need ──────────────────────────────────────
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

# =============================================================================
# CELL 7 — Rename Columns (parent_asin → asin, etc.)
# =============================================================================
# %%

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

# =============================================================================
# CELL 8 — Build 'text' Column (Title + Description)
# =============================================================================
# %%

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

# =============================================================================
# CELL 9 — Build 'image_url' Column
# =============================================================================
# %%

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

# =============================================================================
# CELL 10 — Data Cleaning
# =============================================================================
# %%

print("🧹 Starting data cleaning...\n")
initial_row_count = len(df)

# ── Step 1: Drop rows with null in critical columns ───────────────────────────
critical_cols = ["image_url", "text", "rating"]
before_drop = len(df)
df.dropna(subset=critical_cols, inplace=True)
after_drop = len(df)
print(f"   [Drop nulls]  Removed {before_drop - after_drop} rows "
      f"(image_url / text / rating is null).")

# ── Step 2: Fill missing numeric values ───────────────────────────────────────
# price → median of remaining column (computed AFTER null drop for accuracy)
price_median = df["price"].median()
df["price"].fillna(price_median, inplace=True)
print(f"   [Fill price]  Missing prices filled with median = {price_median:.2f}")

# rating_number → 0 (unknown review count)
n_missing_rn = df["rating_number"].isna().sum()
df["rating_number"].fillna(FILL_RATING_NUMBER, inplace=True)
print(f"   [Fill rating_number]  Filled {n_missing_rn} missing values with {FILL_RATING_NUMBER}")

# ── Step 3: Remove duplicates on 'asin' ──────────────────────────────────────
before_dedup = len(df)
df.drop_duplicates(subset=["asin"], keep="first", inplace=True)
after_dedup = len(df)
print(f"   [Dedup asin]  Removed {before_dedup - after_dedup} duplicate ASIN rows.")

print(f"\n✅ Cleaning complete.")
print(f"   Rows before cleaning : {initial_row_count}")
print(f"   Rows after  cleaning : {len(df)}")

# =============================================================================
# CELL 11 — Sampling (Cap at MAX_ROWS)
# =============================================================================
# %%

current_rows = len(df)
print(f"📊 Row count before sampling: {current_rows}")

if current_rows > MAX_ROWS:
    df = df.sample(n=MAX_ROWS, random_state=RANDOM_SEED)
    print(f"   Randomly sampled {MAX_ROWS} rows (seed={RANDOM_SEED}).")
else:
    print(f"   Row count ≤ {MAX_ROWS} — keeping all {current_rows} rows.")

# ── Reset index for clean sequential indexing ─────────────────────────────────
df.reset_index(drop=True, inplace=True)
print(f"\n✅ Final row count after sampling: {len(df)}")

# =============================================================================
# CELL 12 — Reorder Columns to Final Schema
# =============================================================================
# %%

# ── Only include columns that actually exist in df ────────────────────────────
available_final_cols = [c for c in FINAL_COLUMNS if c in df.columns]
missing_final_cols   = [c for c in FINAL_COLUMNS if c not in df.columns]

if missing_final_cols:
    print(f"⚠️  Warning: The following expected final columns are missing: {missing_final_cols}")

df = df[available_final_cols]

print("✅ Columns reordered to final schema:")
for i, col in enumerate(df.columns, 1):
    target_marker = "  ← TARGET" if col == "rating" else ""
    print(f"   {i}. {col}{target_marker}")

# =============================================================================
# CELL 13 — Final Validation
# =============================================================================
# %%

print("🔍 Running final validation checks...\n")

validation_passed = True
issues = []

# ── Check 1: No nested structures (all values must be flat) ───────────────────
for col in df.columns:
    has_nested = df[col].apply(lambda x: isinstance(x, (list, dict))).any()
    if has_nested:
        issues.append(f"   ❌ Column '{col}' contains nested structures (list/dict).")
        validation_passed = False

# ── Check 2: 'rating' is numeric ─────────────────────────────────────────────
if not pd.api.types.is_numeric_dtype(df["rating"]):
    issues.append("   ❌ 'rating' column is NOT numeric — coercing...")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    # Drop any rows where coercion produced NaN
    df.dropna(subset=["rating"], inplace=True)
    issues.append(f"   ⚠️  'rating' coerced to numeric. Rows dropped: {len(df)}")
    validation_passed = False

# ── Check 3: 'rating' is the last column ─────────────────────────────────────
if df.columns[-1] != "rating":
    issues.append("   ❌ 'rating' is NOT the last column — reordering...")
    cols = [c for c in df.columns if c != "rating"] + ["rating"]
    df = df[cols]
    validation_passed = False

# ── Check 4: No remaining nulls in critical columns ───────────────────────────
for col in ["image_url", "text", "rating"]:
    if df[col].isna().any():
        issues.append(f"   ❌ Column '{col}' still has null values after cleaning.")
        validation_passed = False

# ── Check 5: 'text' and 'image_url' are string type ──────────────────────────
for col in ["text", "image_url"]:
    if col in df.columns:
        df[col] = df[col].astype(str)

# ── Report ────────────────────────────────────────────────────────────────────
if validation_passed and not issues:
    print("✅ ALL VALIDATIONS PASSED — dataset is clean and ML-ready!\n")
else:
    print("⚠️  Validation issues found and auto-corrected:\n")
    for issue in issues:
        print(issue)
    print()

# ── Summary stats ─────────────────────────────────────────────────────────────
print("📋 Final Dataset Summary:")
print(f"   Shape          : {df.shape}")
print(f"   Columns        : {list(df.columns)}")
print(f"   Null counts    :\n{df.isnull().sum().to_string()}")
print(f"\n   Rating stats   :")
print(df["rating"].describe().to_string())

# =============================================================================
# CELL 14 — Save Output CSV
# =============================================================================
# %%

# ── Create output directory if it doesn't exist ───────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"📁 Output directory ready: '{OUTPUT_DIR}/'")

# ── Save to CSV ───────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_PATH, index=False)

# ── Verify file was written correctly ─────────────────────────────────────────
saved_size_kb = os.path.getsize(OUTPUT_PATH) / 1024
df_verify     = pd.read_csv(OUTPUT_PATH, nrows=2)  # Quick verify read

print(f"\n✅ Dataset saved successfully!")
print(f"   Path    : {OUTPUT_PATH}")
print(f"   Size    : {saved_size_kb:.2f} KB")
print(f"   Rows    : {len(df)}")
print(f"   Columns : {list(df.columns)}")
print(f"\n   Preview (first 2 rows):")
print(df_verify.to_string(index=False))

# =============================================================================
# CELL 15 — Quick EDA Summary (Optional but Useful Pre-Training Sanity Check)
# =============================================================================
# %%

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
