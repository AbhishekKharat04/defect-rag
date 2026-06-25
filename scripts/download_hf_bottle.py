import logging
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("download_hf_bottle")

DATA_DIR = Path("data/bottle")

# Source Hugging Face URLs
TRAIN_BASE_URL = "https://huggingface.co/datasets/Mahinur/mvtec-bottle/resolve/main"
TEST_BASE_URL = "https://huggingface.co/datasets/Mahinur/mvtec-bottle-v2/resolve/main"

def download_file(url: str, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    try:
        with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
            out_file.write(response.read())
    except Exception as e:
        logger.error(f"Failed to download {url} to {dest_path}: {e}")
        raise e

def main():
    logger.info("Starting parallel MVTec AD bottle dataset reconstruction from Hugging Face...")

    download_tasks = []

    # 1. Train Good split (209 images)
    train_good_dir = DATA_DIR / "train" / "good"
    for i in range(209):
        filename = f"{i:03d}.png"
        url = f"{TRAIN_BASE_URL}/{filename}"
        dest = train_good_dir / filename
        download_tasks.append((url, dest))

    # 2. Test Good split (20 images)
    test_good_dir = DATA_DIR / "test" / "good"
    for i in range(20):
        src_name = f"{i:03d} (2).png"
        dest_name = f"{i:03d}.png"
        url = f"{TEST_BASE_URL}/{urllib.parse.quote(src_name)}"
        dest = test_good_dir / dest_name
        download_tasks.append((url, dest))

    # 3. Test Scratch split (22 images)
    test_scratch_dir = DATA_DIR / "test" / "scratch"
    for i in range(22):
        filename = f"{i:03d}.png"
        url = f"{TEST_BASE_URL}/{filename}"
        dest = test_scratch_dir / filename
        download_tasks.append((url, dest))

    # 4. Test Contamination split (21 images)
    test_cont_dir = DATA_DIR / "test" / "contamination"
    for i in range(21):
        src_name = f"{i:03d} (1).png"
        dest_name = f"{i:03d}.png"
        url = f"{TEST_BASE_URL}/{urllib.parse.quote(src_name)}"
        dest = test_cont_dir / dest_name
        download_tasks.append((url, dest))

    # 5. Test Broken split (20 images)
    test_broken_dir = DATA_DIR / "test" / "broken"
    for i in range(20):
        src_name = f"{i:03d} (3).png"
        dest_name = f"{i:03d}.png"
        url = f"{TEST_BASE_URL}/{urllib.parse.quote(src_name)}"
        dest = test_broken_dir / dest_name
        download_tasks.append((url, dest))

    total_files = len(download_tasks)
    logger.info(f"Total files to download: {total_files}")

    # Use ThreadPoolExecutor to download in parallel
    max_workers = 16
    completed_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_file, url, dest): (url, dest) for url, dest in download_tasks}
        for future in as_completed(futures):
            url, dest = futures[future]
            try:
                future.result()
                completed_count += 1
                if completed_count % 20 == 0 or completed_count == total_files:
                    logger.info(f"Downloaded {completed_count}/{total_files} files...")
            except Exception as e:
                logger.error(f"Error downloading {url}: {e}")
                raise e

    logger.info("Dataset reconstruction complete!")

if __name__ == "__main__":
    main()
