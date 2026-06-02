# =============================================================================
# configs/paths.py
# Centralized Filesystem Authority -- Multimodal AI Pipeline
# =============================================================================
#
# Purpose:
#   THE SINGLE SOURCE OF TRUTH for every filesystem path in the project.
#   All orchestration-layer modules (dataset.py, train.py, inference.py,
#   shap_analysis.py) import paths from HERE -- never hardcode their own.
#
# Design Philosophy:
#   - One constant to change when the project moves: PROJECT_ROOT
#   - Auto-detects Google Colab vs local execution -- zero manual toggles
#   - All exports are pathlib.Path objects -- no raw string mixing
#   - Runtime dirs (checkpoints, logs) are auto-created safely
#   - Data dirs (datasets, preprocessed) are NOT auto-created -- they must
#     exist intentionally as proof that the data pipeline ran
#   - Fully idempotent -- safe to re-import and re-run in notebooks
#   - Zero ML dependencies -- no torch, no transformers, no numpy
#
# Ownership Boundaries:
#   - This file ONLY handles paths and environment detection
#   - It does NOT load datasets, models, or contain business logic
#   - It does NOT import any ML library
#
# Usage:
#   from configs.paths import PROJECT_ROOT, CHECKPOINT_DIR, PREPROCESSED_DATASET_DIR
#
# Compatible with:
#   - Google Colab + mounted Drive
#   - Local Windows / Linux / macOS execution
#   - Jupyter / IPython notebooks
#   - CI/CD pipelines
#   - Future cloud migration
# =============================================================================


# %%
# =============================================================================
# CELL 1 -- Imports (Minimal, Zero ML Dependencies)
# =============================================================================

import os
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# %%
# =============================================================================
# CELL 2 -- Environment Detection (Auto-Detect Colab vs Local)
# =============================================================================
#
# Resolution priority:
#   1. Environment variable MULTI_MODEL_AI_ROOT if set and exists
#   2. Colab Drive mount: /content/drive/MyDrive/multi-model-ai
#   3. Colab runtime root: /content/multi-model-ai
#   4. Local Windows root: D:/multi-model-ai
#   5. Current-file-derived fallback if it contains project markers
#   6. Else raise clear FileNotFoundError
#
# To relocate the project:
#   - Set MULTI_MODEL_AI_ROOT environment variable, OR
#   - Update _COLAB_ROOT or _LOCAL_ROOT below
#   - Every downstream import automatically picks up the change

_COLAB_ROOT = Path("/content/drive/MyDrive/multi-model-ai")
_COLAB_RUNTIME_ROOT = Path("/content/multi-model-ai")
_LOCAL_ROOT = Path("D:/multi-model-ai")

# Expected project structure markers for fallback validation
_PROJECT_MARKERS = ("configs", "data_pipeline", "models", "preprocessed-datasets")


def _has_project_markers(candidate: Path) -> bool:
    """Check if a candidate directory has expected project structure."""
    return all((candidate / m).exists() for m in _PROJECT_MARKERS)


def _resolve_project_root() -> Path:
    """
    Determines the project root based on environment detection.

    Resolution order:
      1. MULTI_MODEL_AI_ROOT env var (resolved, expanded, normalized)
      2. Colab Drive mount:   /content/drive/MyDrive/multi-model-ai
      3. Colab runtime root:  /content/multi-model-ai
      4. Local Windows root:  D:/multi-model-ai
      5. File-derived root if it contains project markers
      6. Else raise with debugging guidance

    Returns:
        Path : Verified, absolute project root directory.

    Raises:
        FileNotFoundError : If no valid project root is found.
    """
    checked = []

    # 1. Environment variable override
    env_root_str = os.environ.get("MULTI_MODEL_AI_ROOT")
    if env_root_str:
        env_root = Path(env_root_str).expanduser().resolve()
        checked.append(("MULTI_MODEL_AI_ROOT", env_root))
        if env_root.exists() and _has_project_markers(env_root):
            logger.info(f"Environment detected: ENV_VAR | root={env_root}")
            return env_root
        elif env_root.exists():
            logger.warning(
                f"MULTI_MODEL_AI_ROOT={env_root} exists but missing project markers: "
                f"{[m for m in _PROJECT_MARKERS if not (env_root / m).exists()]}"
            )

    # 2. Colab Drive mount
    checked.append(("Colab Drive", _COLAB_ROOT))
    if _COLAB_ROOT.exists():
        logger.info(f"Environment detected: COLAB | root={_COLAB_ROOT}")
        return _COLAB_ROOT

    # 3. Colab runtime root
    checked.append(("Colab runtime", _COLAB_RUNTIME_ROOT))
    if _COLAB_RUNTIME_ROOT.exists():
        logger.info(f"Environment detected: COLAB_RUNTIME | root={_COLAB_RUNTIME_ROOT}")
        return _COLAB_RUNTIME_ROOT

    # 4. Local Windows root
    checked.append(("Local", _LOCAL_ROOT))
    if _LOCAL_ROOT.exists():
        logger.info(f"Environment detected: LOCAL | root={_LOCAL_ROOT}")
        return _LOCAL_ROOT

    # 5. File-derived fallback (configs/paths.py -> PROJECT_ROOT)
    derived = Path(__file__).resolve().parent.parent
    checked.append(("File-derived", derived))
    if derived.exists() and _has_project_markers(derived):
        logger.info(f"Environment detected: FILE_DERIVED | root={derived}")
        return derived

    # 6. Failure with full diagnostics
    details = "\n".join(f"    {name:20s}: {p} (exists={p.exists()})" for name, p in checked)
    raise FileNotFoundError(
        f"PROJECT ROOT NOT FOUND.\n"
        f"  Checked locations:\n{details}\n\n"
        f"  Expected project markers: {_PROJECT_MARKERS}\n\n"
        f"  Resolution:\n"
        f"    Option 1: Set MULTI_MODEL_AI_ROOT environment variable\n"
        f"    Option 2 (Colab): Mount Drive -> drive.mount('/content/drive')\n"
        f"    Option 3 (Local): Verify project exists at {_LOCAL_ROOT}\n"
        f"    Option 4: Update _COLAB_ROOT or _LOCAL_ROOT in configs/paths.py"
    )


# -- Resolve once at import time -- cached for all downstream imports ----------
PROJECT_ROOT: Path = _resolve_project_root()


# %%
# =============================================================================
# CELL 3 -- Colab sys.path Injection (Centralized Import Routing)
# =============================================================================
#
# Ensures `from models.image_encoder import ...` works in Colab notebooks
# by adding PROJECT_ROOT to sys.path exactly once.
#
# This REPLACES the per-file sys.path.append() calls that were previously
# scattered across image_encoder.py, text_encoder.py, tabular_encoder.py,
# and fusion.py. Those per-file calls remain as fallback safety nets, but
# THIS is the canonical, centralized injection point.

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    logger.debug(f"sys.path updated: {PROJECT_ROOT} inserted at position 0")


# %%
# =============================================================================
# CELL 4 -- Source Code Directories (Existing -- NOT Auto-Created)
# =============================================================================
#
# These directories contain source code and MUST already exist.
# They are derived from PROJECT_ROOT for import and reference purposes.
# If any is missing, the project structure itself is broken -- we do NOT
# silently create source directories.

# -- Core Source ---------------------------------------------------------------
CONFIG_DIR        : Path = PROJECT_ROOT / "configs"
MODEL_DIR         : Path = PROJECT_ROOT / "models"
PREPROCESSING_DIR : Path = PROJECT_ROOT / "preprocessing"
ANALYSIS_DIR      : Path = PROJECT_ROOT / "data_analysis"

# -- Future Orchestration (directories will be created as modules are built) ---
DATA_PIPELINE_DIR : Path = PROJECT_ROOT / "data_pipeline"
TRAINING_DIR      : Path = PROJECT_ROOT / "training"
INFERENCE_DIR     : Path = PROJECT_ROOT / "inference"
EXPLAINABILITY_DIR: Path = PROJECT_ROOT / "explainability"
UTILS_DIR         : Path = PROJECT_ROOT / "utils"
TEST_DIR          : Path = PROJECT_ROOT / "tests"


# %%
# =============================================================================
# CELL 5 -- Data Directories (Existing -- NOT Auto-Created)
# =============================================================================
#
# These directories contain user-provided data or preprocessing outputs.
# They MUST already exist before training -- we do NOT auto-create them
# because their absence means the data pipeline hasn't run yet, which is
# a REAL error that must be caught, not silently papered over.

# -- Raw Datasets --------------------------------------------------------------
# Original JSONL/JSON files from Amazon product datasets
DATASET_DIR       : Path = PROJECT_ROOT / "datasets"
RAW_META_DIR      : Path = DATASET_DIR / "datasets-meta"
RAW_REVIEW_DIR    : Path = DATASET_DIR / "datasets-review"

# -- Preprocessed Datasets -----------------------------------------------------
# Cleaned CSV outputs from data_preprocessing.py
PREPROCESSED_DATASET_DIR: Path = PROJECT_ROOT / "preprocessed-datasets"

# -- Downloaded Product Images -------------------------------------------------
# Images downloaded by image_downloader.py, stored as {asin}.jpg
IMAGE_DATASET_DIR : Path = PREPROCESSED_DATASET_DIR / "images"


# %%
# =============================================================================
# CELL 6 -- Runtime Directories (Auto-Created Safely)
# =============================================================================
#
# These directories are created at import time if they don't exist.
# They store outputs generated DURING training / inference -- not source
# data. Safe to auto-create because:
#   - Their absence doesn't indicate a broken pipeline
#   - They're always needed before the first training run
#   - Re-creation is idempotent (exist_ok=True)

CHECKPOINT_DIR : Path = PROJECT_ROOT / "checkpoints"
EXPERIMENT_DIR : Path = PROJECT_ROOT / "experiments"
CACHE_DIR      : Path = PROJECT_ROOT / "cache"
LOG_DIR        : Path = PROJECT_ROOT / "logs"
SHAP_OUTPUT_DIR: Path = PROJECT_ROOT / "shap_outputs"

# -- Runtime directories to auto-create ----------------------------------------
_RUNTIME_DIRS = [
    CHECKPOINT_DIR,
    EXPERIMENT_DIR,
    CACHE_DIR,
    LOG_DIR,
    SHAP_OUTPUT_DIR,
]


def _ensure_runtime_dirs() -> None:
    """
    Creates all runtime output directories if they don't already exist.

    Uses mkdir(parents=True, exist_ok=True) for full idempotency --
    safe to call repeatedly in notebooks, CI/CD, or multiprocess training.
    """
    for dir_path in _RUNTIME_DIRS:
        dir_path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Runtime directories verified: {len(_RUNTIME_DIRS)} dirs ready")


# Execute once at import time
_ensure_runtime_dirs()


# %%
# =============================================================================
# CELL 7 -- Data Directory Validation Utilities
# =============================================================================
#
# These are NOT called at import time -- they're called explicitly by
# dataset.py and train.py before training begins. This avoids crashing
# during import when only doing quick encoder tests or smoke tests.

def validate_data_dirs() -> None:
    """
    Validates that all required DATA directories exist.

    Call this from dataset.py or train.py BEFORE loading any data.
    This catches missing preprocessing outputs early with clear guidance.

    Raises:
        FileNotFoundError : If any required data directory is missing.
    """
    required = {
        "DATASET_DIR"              : DATASET_DIR,
        "PREPROCESSED_DATASET_DIR" : PREPROCESSED_DATASET_DIR,
        "IMAGE_DATASET_DIR"        : IMAGE_DATASET_DIR,
    }

    missing = {name: path for name, path in required.items() if not path.exists()}

    if missing:
        details = "\n".join(
            f"    - {name}: {path}" for name, path in missing.items()
        )
        raise FileNotFoundError(
            f"REQUIRED DATA DIRECTORIES NOT FOUND.\n"
            f"  Missing:\n{details}\n\n"
            f"  Resolution:\n"
            f"    1. Run preprocessing/data_preprocessing.py first\n"
            f"    2. Run preprocessing/image_downloader.py second\n"
            f"    3. Verify PROJECT_ROOT is correct: {PROJECT_ROOT}"
        )

    logger.info(
        f"Data directories validated | "
        f"datasets={DATASET_DIR.exists()} | "
        f"preprocessed={PREPROCESSED_DATASET_DIR.exists()} | "
        f"images={IMAGE_DATASET_DIR.exists()}"
    )


def validate_project_structure() -> None:
    """
    Validates the FULL project structure -- source + data + runtime dirs.

    Call this during system boot (train.py startup) to catch structural
    issues before any heavy model loading begins.

    Raises:
        FileNotFoundError : If any critical source directory is missing.
    """
    critical_source = {
        "CONFIG_DIR"       : CONFIG_DIR,
        "MODEL_DIR"        : MODEL_DIR,
        "PREPROCESSING_DIR": PREPROCESSING_DIR,
        "ANALYSIS_DIR"     : ANALYSIS_DIR,
    }

    missing = {name: path for name, path in critical_source.items() if not path.exists()}

    if missing:
        details = "\n".join(
            f"    - {name}: {path}" for name, path in missing.items()
        )
        raise FileNotFoundError(
            f"CRITICAL PROJECT STRUCTURE BROKEN.\n"
            f"  Missing source directories:\n{details}\n\n"
            f"  Resolution:\n"
            f"    - Verify git clone / Drive mount is complete\n"
            f"    - PROJECT_ROOT: {PROJECT_ROOT}"
        )

    # Also validate data dirs
    validate_data_dirs()

    logger.info("Full project structure validated -- all directories present.")


# %%
# =============================================================================
# CELL 8 -- Path Helper Utilities
# =============================================================================

def get_checkpoint_path(filename: str) -> Path:
    """
    Returns the full path for a checkpoint file.

    Args:
        filename : Checkpoint filename (e.g., 'epoch_10.pt', 'best_model.pt').

    Returns:
        Path : Absolute path inside CHECKPOINT_DIR.
    """
    if not filename:
        raise ValueError("Checkpoint filename cannot be empty.")
    return CHECKPOINT_DIR / filename


def get_experiment_dir(experiment_name: str) -> Path:
    """
    Returns and creates a named experiment subdirectory.

    Each experiment gets its own folder for logs, checkpoints, and metrics.

    Args:
        experiment_name : Human-readable experiment identifier
                          (e.g., 'baseline_v1', 'fusion_ablation_text_only').

    Returns:
        Path : Created experiment subdirectory inside EXPERIMENT_DIR.
    """
    if not experiment_name or not experiment_name.strip():
        raise ValueError("Experiment name cannot be empty or whitespace.")

    # Sanitize: replace spaces/special chars with underscores
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in experiment_name)
    exp_dir = EXPERIMENT_DIR / safe_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def _ensure_child_path(base: Path, candidate: Path, context: str) -> Path:
    """
    Validate that *candidate* resolves to a location inside *base*.

    Prevents path-traversal attacks (e.g., '../../escape.csv').

    Args:
        base      : The trusted parent directory.
        candidate : The candidate path to validate.
        context   : Human-readable label for error messages.

    Returns:
        Path : The resolved, validated candidate path.

    Raises:
        ValueError : If the candidate resolves outside the base.
    """
    base_resolved = base.resolve()
    candidate_resolved = candidate.resolve()
    if candidate_resolved != base_resolved and base_resolved not in candidate_resolved.parents:
        raise ValueError(
            f"PATH TRAVERSAL BLOCKED ({context}).\n"
            f"  Base       : {base_resolved}\n"
            f"  Candidate  : {candidate_resolved}\n"
            f"  Resolution : Use a simple filename without '..' or absolute paths."
        )
    return candidate_resolved


def get_dataset_csv(filename: str) -> Path:
    """
    Returns the full path for a preprocessed CSV dataset file.

    Args:
        filename : CSV filename (e.g., 'sample_100.csv',
                   'meta_amazon_fashion_processed.csv').

    Returns:
        Path : Absolute path inside PREPROCESSED_DATASET_DIR.

    Raises:
        FileNotFoundError : If the CSV file does not exist.
        ValueError         : If the path escapes PREPROCESSED_DATASET_DIR.
    """
    if not isinstance(filename, str):
        raise TypeError(f"Dataset CSV filename must be str, got {type(filename).__name__}")
    filename = filename.strip()
    if not filename:
        raise ValueError("Dataset CSV filename cannot be whitespace only.")

    # Reject absolute paths that point outside PREPROCESSED_DATASET_DIR
    if Path(filename).is_absolute():
        abs_resolved = Path(filename).resolve()
        base_resolved = PREPROCESSED_DATASET_DIR.resolve()
        if abs_resolved != base_resolved and base_resolved not in abs_resolved.parents:
            raise ValueError(
                f"PATH TRAVERSAL BLOCKED (get_dataset_csv).\n"
                f"  Absolute path '{filename}' is outside PREPROCESSED_DATASET_DIR.\n"
                f"  Resolution : Use a simple filename like 'sample_100.csv'."
            )

    csv_path = _ensure_child_path(
        PREPROCESSED_DATASET_DIR,
        PREPROCESSED_DATASET_DIR / filename,
        "get_dataset_csv",
    )

    if not csv_path.exists():
        # List available CSVs for debugging
        available = [f.name for f in PREPROCESSED_DATASET_DIR.glob("*.csv")] if PREPROCESSED_DATASET_DIR.exists() else []
        raise FileNotFoundError(
            f"DATASET NOT FOUND: {csv_path}\n"
            f"  Available CSVs: {available if available else 'NONE (directory may be empty)'}\n"
            f"  Resolution: Run preprocessing/data_preprocessing.py first."
        )

    return csv_path


def list_preprocessed_csvs() -> list:
    """
    Discover all CSV files in PREPROCESSED_DATASET_DIR.

    Returns:
        List[Path] : Sorted list of absolute paths to each .csv file.
                     Returns empty list if directory does not exist.
    """
    if not PREPROCESSED_DATASET_DIR.exists():
        return []
    return sorted(PREPROCESSED_DATASET_DIR.glob("*.csv"))


def resolve_preprocessed_csv(filename: str) -> Path:
    """
    Resolve a preprocessed CSV filename to an absolute path.

    Functionally equivalent to get_dataset_csv() but with a cleaner name
    for registry/dataset routing use. get_dataset_csv() remains for
    backward compatibility.

    Args:
        filename : CSV filename (e.g., 'sample_100.csv').

    Returns:
        Path : Absolute path inside PREPROCESSED_DATASET_DIR.

    Raises:
        FileNotFoundError : If the CSV file does not exist.
        ValueError         : If the path escapes PREPROCESSED_DATASET_DIR.
    """
    return get_dataset_csv(filename)


def resolve_image_file(filename_or_asin: str) -> Path:
    """
    Resolve an image filename or bare ASIN to an absolute path inside
    IMAGE_DATASET_DIR.

    Normalization rules:
      - 'B001.jpg'    -> IMAGE_DATASET_DIR / 'B001.jpg'
      - 'B001'        -> IMAGE_DATASET_DIR / 'B001.jpg'
      - 'B001.png'    -> IMAGE_DATASET_DIR / 'B001.png' (preserves extension)
      - '../escape'   -> ValueError (traversal blocked)
      - './B001.jpg'   -> IMAGE_DATASET_DIR / 'B001.jpg' (if resolves safely)
      - 'sub/B001.jpg' -> IMAGE_DATASET_DIR / 'sub/B001.jpg' (if under base)

    This does NOT validate that the file exists.  Existence checking
    belongs to the caller (dataset.py) so fallback behavior stays local.

    Args:
        filename_or_asin : Image filename or bare ASIN string.

    Returns:
        Path : Absolute path inside IMAGE_DATASET_DIR.

    Raises:
        ValueError : If the resolved path escapes IMAGE_DATASET_DIR.
    """
    if not filename_or_asin or not isinstance(filename_or_asin, str):
        raise ValueError(
            f"resolve_image_file requires a non-empty string, "
            f"got {filename_or_asin!r}"
        )
    name = filename_or_asin.strip()
    if not name:
        raise ValueError("resolve_image_file: filename/ASIN is empty after stripping.")

    # Reject absolute paths outside IMAGE_DATASET_DIR
    if Path(name).is_absolute():
        abs_resolved = Path(name).resolve()
        base_resolved = IMAGE_DATASET_DIR.resolve()
        if abs_resolved != base_resolved and base_resolved not in abs_resolved.parents:
            raise ValueError(
                f"PATH TRAVERSAL BLOCKED (resolve_image_file).\n"
                f"  Absolute path '{name}' is outside IMAGE_DATASET_DIR.\n"
                f"  Resolution : Use a simple filename like 'B001.jpg' or a bare ASIN."
            )

    # If no extension, assume .jpg (the standard image format for this pipeline)
    if "." not in name:
        name = f"{name}.jpg"
    return _ensure_child_path(
        IMAGE_DATASET_DIR,
        IMAGE_DATASET_DIR / name,
        "resolve_image_file",
    )



# %%
# =============================================================================
# CELL 9 -- Environment Summary (Import-Time Diagnostic)
# =============================================================================
#
# Prints a compact summary when paths.py is first imported.
# This is intentionally lightweight -- no heavy validation, just visibility.
# In production (train.py), call validate_project_structure() explicitly.

def _print_environment_summary() -> None:
    """Compact diagnostic printed once at import time."""
    env = "COLAB" if str(PROJECT_ROOT) == str(_COLAB_ROOT) else "LOCAL"

    logger.info("-" * 50)
    logger.info(f"  configs/paths.py loaded")
    logger.info(f"  Environment : {env}")
    logger.info(f"  Project Root: {PROJECT_ROOT}")
    logger.info(f"  Runtime dirs: {len(_RUNTIME_DIRS)} verified")
    logger.info("-" * 50)


_print_environment_summary()


# %%
# =============================================================================
# CELL 10 -- Smoke Test (python configs/paths.py)
# =============================================================================

if __name__ == "__main__":

    logging.basicConfig(
        level   = logging.DEBUG,
        format  = "[%(asctime)s] [%(levelname)s] %(name)s -- %(message)s",
        datefmt = "%H:%M:%S",
    )

    print("=" * 60)
    print("  configs/paths.py -- smoke test")
    print("=" * 60)

    try:
        # -- Environment ---------------------------------------------------
        env = "COLAB" if str(PROJECT_ROOT) == str(_COLAB_ROOT) else "LOCAL"
        print(f"\n  Environment    : {env}")
        print(f"  PROJECT_ROOT   : {PROJECT_ROOT}")
        print(f"  Root exists    : {PROJECT_ROOT.exists()}")

        # -- Source dirs ---------------------------------------------------
        print(f"\n  CONFIG_DIR     : {CONFIG_DIR.exists():<5}  {CONFIG_DIR}")
        print(f"  MODEL_DIR      : {MODEL_DIR.exists():<5}  {MODEL_DIR}")
        print(f"  PREPROCESSING  : {PREPROCESSING_DIR.exists():<5}  {PREPROCESSING_DIR}")
        print(f"  ANALYSIS       : {ANALYSIS_DIR.exists():<5}  {ANALYSIS_DIR}")

        # -- Data dirs -----------------------------------------------------
        print(f"\n  DATASET_DIR    : {DATASET_DIR.exists():<5}  {DATASET_DIR}")
        print(f"  PREPROCESSED   : {PREPROCESSED_DATASET_DIR.exists():<5}  {PREPROCESSED_DATASET_DIR}")
        print(f"  IMAGES         : {IMAGE_DATASET_DIR.exists():<5}  {IMAGE_DATASET_DIR}")

        # -- Runtime dirs --------------------------------------------------
        print(f"\n  CHECKPOINT_DIR : {CHECKPOINT_DIR.exists():<5}  {CHECKPOINT_DIR}")
        print(f"  EXPERIMENT_DIR : {EXPERIMENT_DIR.exists():<5}  {EXPERIMENT_DIR}")
        print(f"  CACHE_DIR      : {CACHE_DIR.exists():<5}  {CACHE_DIR}")
        print(f"  LOG_DIR        : {LOG_DIR.exists():<5}  {LOG_DIR}")
        print(f"  SHAP_OUTPUT    : {SHAP_OUTPUT_DIR.exists():<5}  {SHAP_OUTPUT_DIR}")

        # -- Helper utilities ----------------------------------------------
        print("\n  Testing get_checkpoint_path()...")
        cp = get_checkpoint_path("best_model.pt")
        print(f"    -> {cp}")

        print("  Testing get_experiment_dir()...")
        exp = get_experiment_dir("smoke_test_run")
        print(f"    -> {exp}  (created={exp.exists()})")

        print("  Testing get_dataset_csv() with available file...")
        try:
            csv_path = get_dataset_csv("sample_100.csv")
            print(f"    -> {csv_path}  FOUND")
        except FileNotFoundError as e:
            print(f"    -> Expected: file may not exist locally")

        print("  Testing get_dataset_csv() with missing file...")
        try:
            get_dataset_csv("nonexistent_dataset.csv")
            print("    -> ERROR: Should have raised FileNotFoundError")
        except FileNotFoundError:
            print("    -> FileNotFoundError raised correctly  PASS")

        # -- Edge case: empty filename guard -------------------------------
        print("  Testing empty filename guard...")
        try:
            get_checkpoint_path("")
            print("    -> ERROR: Should have raised ValueError")
        except ValueError:
            print("    -> ValueError raised correctly  PASS")

        try:
            get_experiment_dir("   ")
            print("    -> ERROR: Should have raised ValueError")
        except ValueError:
            print("    -> Whitespace experiment name caught  PASS")

        try:
            get_dataset_csv("")
            print("    -> ERROR: Should have raised ValueError")
        except ValueError:
            print("    -> Empty CSV filename caught  PASS")

        # -- New helper utilities ------------------------------------------
        print("\n  Testing list_preprocessed_csvs()...")
        csvs = list_preprocessed_csvs()
        print(f"    -> Found {len(csvs)} CSVs")
        if csvs:
            print(f"    -> First: {csvs[0].name}")
            all_paths_obj = all(isinstance(p, Path) for p in csvs)
            print(f"    -> All are Path: {all_paths_obj}  {'PASS' if all_paths_obj else 'FAIL'}")

        print("  Testing resolve_preprocessed_csv()...")
        try:
            rp = resolve_preprocessed_csv("sample_100.csv")
            print(f"    -> {rp}  FOUND")
        except FileNotFoundError:
            print(f"    -> Expected: file may not exist locally")

        print("  Testing resolve_image_file()...")
        img_p = resolve_image_file("B001")
        print(f"    -> B001 -> {img_p}")
        assert img_p == IMAGE_DATASET_DIR / "B001.jpg", f"Expected .jpg, got {img_p}"
        img_p2 = resolve_image_file("B001.png")
        print(f"    -> B001.png -> {img_p2}")
        assert img_p2 == IMAGE_DATASET_DIR / "B001.png"
        try:
            resolve_image_file("")
            print("    -> ERROR: Should have raised ValueError")
        except ValueError:
            print("    -> Empty input caught  PASS")
        print("    -> resolve_image_file  PASS")

        # -- Traversal guard tests -----------------------------------------
        print("\n  Testing traversal guards...")
        try:
            resolve_image_file("../escape.jpg")
            print("    -> ERROR: ../escape.jpg should have raised ValueError")
        except ValueError:
            print("    -> ../escape.jpg traversal blocked  PASS")

        try:
            get_dataset_csv("../escape.csv")
            print("    -> ERROR: ../escape.csv should have raised ValueError")
        except (ValueError, FileNotFoundError):
            print("    -> ../escape.csv traversal blocked  PASS")

        try:
            resolve_preprocessed_csv("../../etc/passwd")
            print("    -> ERROR: traversal should have raised")
        except (ValueError, FileNotFoundError):
            print("    -> ../../etc/passwd blocked  PASS")

        # Safe relative should work
        safe = resolve_image_file("./B001.jpg")
        print(f"    -> ./B001.jpg -> {safe.name}  PASS")

        # -- Idempotency: re-run runtime dir creation ----------------------
        print("\n  Testing idempotency (re-run _ensure_runtime_dirs)...")
        _ensure_runtime_dirs()
        print("    -> Re-creation safe  PASS")

        # -- Validation ----------------------------------------------------
        print("\n  Testing validate_project_structure()...")
        try:
            validate_project_structure()
            print("    -> Full validation passed  PASS")
        except FileNotFoundError as e:
            print(f"    -> Partial validation (some data dirs missing)")

        # -- sys.path ------------------------------------------------------
        print(f"\n  sys.path[0]    : {sys.path[0]}")
        print(f"  PROJECT_ROOT in sys.path: {str(PROJECT_ROOT) in sys.path}")

        # -- Type safety ---------------------------------------------------
        print("\n  Type verification...")
        path_exports = [
            ("PROJECT_ROOT", PROJECT_ROOT),
            ("CONFIG_DIR", CONFIG_DIR),
            ("MODEL_DIR", MODEL_DIR),
            ("CHECKPOINT_DIR", CHECKPOINT_DIR),
            ("PREPROCESSED_DATASET_DIR", PREPROCESSED_DATASET_DIR),
        ]
        all_paths = all(isinstance(p, Path) for _, p in path_exports)
        print(f"    All exports are pathlib.Path: {all_paths}  {'PASS' if all_paths else 'FAIL'}")

        print("\n" + "=" * 60)
        print("  [PASS]  Smoke test PASSED -- paths.py is infrastructure-grade stable.")
        print("=" * 60)

    except Exception as e:
        logger.exception(f"[FAIL] SMOKE TEST FAILED: {e}")
        sys.exit(1)
