# Image Downloader and Storage Pipeline

# ==============================================================================
# STEP 1 — IMPORTS
# ==============================================================================
import os
import time
import pandas as pd
from tqdm import tqdm
import asyncio
import aiohttp
from PIL import Image
from io import BytesIO

# ==============================================================================
# STEP 2 — CONFIGURATION SECTION
# ==============================================================================
# List of preprocessed CSV dataset paths. 
# Modify these paths depending on your environment (Local vs Google Colab).
CSV_FILES = [
    "d:/multi-model-ai/preprocessed-datasets/meta_amazon_fashion_processed.csv",
    "d:/multi-model-ai/preprocessed-datasets/meta_digital_music_processed.csv"
]

# Output directory for downloaded images
IMAGE_OUTPUT_DIR = "d:/multi-model-ai/images"

MAX_CONCURRENT_DOWNLOADS = 20
REQUEST_TIMEOUT = 15
MAX_RETRIES = 3
BATCH_SIZE = 100

semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)

# ==============================================================================
# STEP 3 — FUNCTION STRUCTURE
# ==============================================================================

async def download_image(session, image_url, asin):
    """
    Asynchronously downloads an image with retry and timeout handling.
    """
    if pd.isna(image_url) or not str(image_url).strip():
        return asin, None, "skipped_empty_url", 0

    retries = 0
    for attempt in range(MAX_RETRIES):
        try:
            async with semaphore:
                async with session.get(image_url, timeout=REQUEST_TIMEOUT) as response:
                    if response.status == 200:
                        content_type = response.headers.get("Content-Type", "")
                        if not content_type.startswith("image/"):
                            return asin, None, "skipped_invalid_content_type", retries
                            
                        image_bytes = await response.read()
                        return asin, image_bytes, "success", retries
                    elif response.status in [403, 404]:
                        return asin, None, f"skipped_{response.status}", retries
                    else:
                        # Temporary errors, let it retry
                        pass
        except (asyncio.TimeoutError, aiohttp.ClientError):
            pass
        except Exception:
            pass
        
        retries += 1
        if attempt < MAX_RETRIES - 1:
            await asyncio.sleep(1)
            
    return asin, None, "failed_retries_exhausted", retries

def process_image(image_bytes):
    """
    Opens an image safely using PIL, converts to RGB, and resizes if dimensions exceed 512x512
    while preserving aspect ratio. Smaller images are not upscaled.
    """
    try:
        image = Image.open(BytesIO(image_bytes))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        if image.width > 512 or image.height > 512:
            image.thumbnail((512, 512))
            
        return image
    except Exception:
        return None

def save_image(image, asin):
    """
    Saves the PIL image inside IMAGE_OUTPUT_DIR as {asin}.jpg.
    Uses quality=85 and optimize=True.
    """
    try:
        os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
        filename = f"{asin}.jpg"
        save_path = os.path.join(IMAGE_OUTPUT_DIR, filename)
        
        image.save(save_path, "JPEG", quality=85, optimize=True)
        return save_path
    except Exception:
        return None

async def process_row(session, asin, image_url):
    """
    Processes a single row from the dataset.
    """
    filename = f"{asin}.jpg"
    save_path = os.path.join(IMAGE_OUTPUT_DIR, filename)
    if os.path.exists(save_path):
        return asin, filename, "skipped_existing", 0

    asin, image_bytes, status, retries = await download_image(session, image_url, asin)
    
    if status != "success" or not image_bytes:
        return asin, None, status, retries
        
    image = process_image(image_bytes)
    if image is None:
        return asin, None, "failed_corrupted_image", retries
        
    saved_path = save_image(image, asin)
    if saved_path is None:
        return asin, None, "failed_save", retries
        
    return asin, filename, "success", retries

def update_csv(df, dataset_path, mapping_dict):
    """
    Updates the dataset with the new local image paths and saves it.
    """
    if "image_path" not in df.columns:
        df["image_path"] = pd.NA

    # Map ASINs using mapping dictionary, preserve existing if not in mapping
    df['image_path'] = df['asin'].map(mapping_dict).fillna(df['image_path'])

    # Save safely
    try:
        df.to_csv(dataset_path, index=False)
        return True
    except Exception as e:
        print(f"\nError saving CSV {dataset_path}: {e}")
        return False

async def process_dataset(dataset_path):
    """
    Processes a single dataset asynchronously.
    
    Responsibilities:
    1. Reads the CSV file using pandas.
    2. Validates the presence of required columns ('asin', 'image_url').
    3. If columns are missing, skips the dataset safely and logs a warning.
    4. Prints dataset name, row count, and a starting message.
    5. Creates a shared aiohttp.ClientSession.
    6. Processes downloads concurrently using asyncio.gather().
    """
    # Read the CSV file safely
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        print(f"\nError reading dataset '{dataset_path}': {e}")
        return

    # Validate required columns
    required_columns = ['asin', 'image_url']
    for col in required_columns:
        if col not in df.columns:
            print(f"\nWarning: Skipping dataset. Missing required column '{col}' in {dataset_path}")
            return

    # Extract dataset details for logging
    dataset_name = os.path.basename(dataset_path)
    row_count = len(df)
    
    print(f"\nDataset: {dataset_name}")
    print(f"Total Rows: {row_count}")
    print("Processing started...")

    # Create ONE shared aiohttp.ClientSession
    async with aiohttp.ClientSession() as session:
        pbar = tqdm(total=row_count, desc=f"Downloading {dataset_name}")
        
        async def wrapped_process(asin, url):
            try:
                res = await process_row(session, asin, url)
            except Exception as e:
                res = (asin, None, f"failed_exception_{e}", 0)
            finally:
                pbar.update(1)
            return res

        results = []
        batch_tasks = []
        
        for row in df.itertuples():
            asin = getattr(row, 'asin')
            image_url = getattr(row, 'image_url')
            
            batch_tasks.append(wrapped_process(asin, image_url))
            
            # Process in batches to reduce memory pressure
            if len(batch_tasks) >= BATCH_SIZE:
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                results.extend(batch_results)
                batch_tasks.clear()
                
        # Process any remaining tasks
        if batch_tasks:
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            results.extend(batch_results)
            batch_tasks.clear()
            
        pbar.close()
        
    # Logging
    successful = 0
    skipped = 0
    failed = 0
    skipped_existing = 0
    corrupted = 0
    total_retries = 0
    
    for res in results:
        if isinstance(res, Exception):
            failed += 1
            continue
            
        asin, image_path, status, retries = res
        total_retries += retries
            
        if status == "success":
            successful += 1
        elif status == "skipped_existing":
            skipped_existing += 1
            skipped += 1
        elif status == "failed_corrupted_image":
            corrupted += 1
            failed += 1
        elif status and status.startswith("skipped"):
            skipped += 1
        else:
            failed += 1
            
    print(f"  Successful downloads/saves: {successful}")
    print(f"  Failed: {failed} (Corrupted: {corrupted})")
    print(f"  Skipped: {skipped} (Existing: {skipped_existing})")
    print(f"  Total retry attempts: {total_retries}")
        
    # Create mapping dictionary for successful / existing images
    mapping_dict = {}
    for res in results:
        if isinstance(res, Exception):
            continue
        asin, image_path, status, _ = res
        if image_path and status in ["success", "skipped_existing"]:
            mapping_dict[asin] = image_path

    if mapping_dict:
        success_save = update_csv(df, dataset_path, mapping_dict)
        if success_save:
            print(f"\nSuccessfully updated {len(mapping_dict)} image paths in {dataset_name}.")
        else:
            print(f"\nFailed to update CSV for {dataset_name}.")
    else:
        print(f"\nNo new or cached images to update in {dataset_name}.")

# ==============================================================================
# MAIN DATASET ITERATION PIPELINE
# ==============================================================================
async def main():
    """
    Main execution pipeline.
    Iterates through the list of CSV datasets and processes each one sequentially.
    """
    print("Starting Image Downloader and Storage Pipeline...")
    
    for dataset_path in CSV_FILES:
        await process_dataset(dataset_path)
        
    print("\nPipeline execution finished.")

if __name__ == "__main__":
    asyncio.run(main())
