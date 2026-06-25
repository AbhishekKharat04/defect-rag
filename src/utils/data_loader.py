"""Data loading and synthetic dataset generation for MVTec AD.

Provides two data-source strategies:

1. **Real data** — downloads a specific MVTec Anomaly Detection category
   archive from the official website (``download_mvtec_category``).
2. **Synthetic data** — generates Pillow-drawn bottle images with
   simulated defects for fast offline testing
   (``generate_synthetic_dataset``).

Both functions produce the same directory layout so the rest of the
pipeline is agnostic to the data source::

    <category>/
    ├── train/
    │   └── good/
    └── test/
        ├── good/
        ├── scratch/
        ├── contamination/
        └── broken/
"""

import logging
from pathlib import Path

from PIL import Image, ImageDraw

from src.config import settings

logger = logging.getLogger(__name__)

MVTEC_BASE_URL = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937370-1629951468"


def download_mvtec_category(category: str = "bottle", target_dir: Path | None = None) -> Path:
    """Download and extract a specific MVTec AD category dataset.

    If the category directory already exists and is non-empty, the
    download is skipped.  On network failure the function falls back to
    :func:`generate_synthetic_dataset`.

    Args:
        category: MVTec category name (e.g. ``'bottle'``, ``'hazelnut'``).
        target_dir: Base directory for the extracted dataset.
            Defaults to ``settings.DATA_DIR``.

    Returns:
        ``Path`` to the extracted (or generated) dataset directory.
    """
    dest_dir = target_dir or settings.DATA_DIR
    category_dir = dest_dir / category

    if category_dir.exists() and any(category_dir.iterdir()):
        logger.info(f"Dataset category '{category}' already exists at {category_dir}")
        return category_dir

    if category != "bottle":
        logger.warning(f"Category '{category}' is not supported for mirror download. Falling back to synthetic.")
        return generate_synthetic_dataset(category, dest_dir)

    import urllib.parse
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor, as_completed

    logger.info("Starting parallel MVTec AD bottle dataset download from HF mirror...")

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

    download_tasks = []

    # 1. train/good (209 images)
    train_good_dir = category_dir / "train" / "good"
    for i in range(209):
        filename = f"{i:03d}.png"
        download_tasks.append((f"{TRAIN_BASE_URL}/{filename}", train_good_dir / filename))

    # 2. test/good (20 images)
    test_good_dir = category_dir / "test" / "good"
    for i in range(20):
        src_name = f"{i:03d} (2).png"
        dest_name = f"{i:03d}.png"
        download_tasks.append((f"{TEST_BASE_URL}/{urllib.parse.quote(src_name)}", test_good_dir / dest_name))

    # 3. test/broken_small (22 images, mapped from scratch)
    test_broken_small_dir = category_dir / "test" / "broken_small"
    for i in range(22):
        filename = f"{i:03d}.png"
        download_tasks.append((f"{TEST_BASE_URL}/{filename}", test_broken_small_dir / filename))

    # 4. test/contamination (21 images)
    test_cont_dir = category_dir / "test" / "contamination"
    for i in range(21):
        src_name = f"{i:03d} (1).png"
        dest_name = f"{i:03d}.png"
        download_tasks.append((f"{TEST_BASE_URL}/{urllib.parse.quote(src_name)}", test_cont_dir / dest_name))

    # 5. test/broken_large (20 images, mapped from broken)
    test_broken_large_dir = category_dir / "test" / "broken_large"
    for i in range(20):
        src_name = f"{i:03d} (3).png"
        dest_name = f"{i:03d}.png"
        download_tasks.append((f"{TEST_BASE_URL}/{urllib.parse.quote(src_name)}", test_broken_large_dir / dest_name))

    try:
        total_files = len(download_tasks)
        completed_count = 0
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {executor.submit(download_file, url, dest): (url, dest) for url, dest in download_tasks}
            for future in as_completed(futures):
                future.result()
                completed_count += 1
                if completed_count % 50 == 0 or completed_count == total_files:
                    logger.info(f"Downloaded {completed_count}/{total_files} files...")
        logger.info("Successfully downloaded and structured MVTec bottle dataset from HF mirror.")
        return category_dir
    except Exception as e:
        logger.warning(f"Failed to download/extract MVTec dataset: {e}. Falling back to synthetic dataset.")
        return generate_synthetic_dataset(category, dest_dir)


def generate_synthetic_dataset(category: str = "bottle", target_dir: Path | None = None) -> Path:
    """Generate a synthetic defect dataset mimicking MVTec AD folder structure.

    Creates Pillow-drawn bottle images with simulated defects:

    - **good** — clean bottles with slight liquid-level variation.
    - **scratch** — diagonal dark lines across the bottle body.
    - **contamination** — dark elliptical spots on the surface.
    - **broken** — triangular cutouts and branching cracks.

    Args:
        category: Name of the dataset folder (default ``'bottle'``).
        target_dir: Base directory for the generated dataset.

    Returns:
        ``Path`` to the generated dataset directory.
    """
    dest_dir = target_dir or settings.DATA_DIR
    category_dir = dest_dir / category

    logger.info(f"Generating synthetic defect dataset in {category_dir}...")

    paths = {
        "train_good": category_dir / "train" / "good",
        "test_good": category_dir / "test" / "good",
        "test_broken_small": category_dir / "test" / "broken_small",
        "test_contamination": category_dir / "test" / "contamination",
        "test_broken_large": category_dir / "test" / "broken_large",
    }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    def _create_base_bottle() -> Image.Image:
        """Draw a simple bottle shape on a light-grey background."""
        img = Image.new("RGB", (256, 256), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        draw.rectangle([80, 80, 176, 220], fill=(173, 216, 230), outline=(100, 149, 237), width=2)
        draw.rectangle([108, 30, 148, 80], fill=(173, 216, 230), outline=(100, 149, 237), width=2)
        draw.rectangle([104, 20, 152, 30], fill=(220, 20, 60))
        draw.rectangle([82, 120, 174, 218], fill=(30, 144, 255))
        return img

    # Train good images (20)
    for i in range(20):
        img = _create_base_bottle()
        draw = ImageDraw.Draw(img)
        liquid_level = 115 + (i % 5) * 4
        draw.rectangle([82, liquid_level, 174, 218], fill=(30, 144, 255))
        img.save(paths["train_good"] / f"{i:03d}.png")

    # Test good images (5)
    for i in range(5):
        img = _create_base_bottle()
        draw = ImageDraw.Draw(img)
        liquid_level = 115 - (i % 3) * 3
        draw.rectangle([82, liquid_level, 174, 218], fill=(30, 144, 255))
        img.save(paths["test_good"] / f"{i:03d}.png")

    # Test broken_small images (5)
    for i in range(5):
        img = _create_base_bottle()
        draw = ImageDraw.Draw(img)
        start_x = 90 + i * 10
        start_y = 100 + i * 15
        draw.line([start_x, start_y, start_x + 30, start_y + 30], fill=(50, 50, 50), width=2)
        draw.line([start_x + 10, start_y + 20, start_x + 40, start_y + 10], fill=(50, 50, 50), width=1)
        img.save(paths["test_broken_small"] / f"{i:03d}.png")

    # Test contamination images (5)
    for i in range(5):
        img = _create_base_bottle()
        draw = ImageDraw.Draw(img)
        spot_x = 95 + i * 12
        spot_y = 130 + i * 10
        draw.ellipse([spot_x, spot_y, spot_x + 12, spot_y + 10], fill=(47, 79, 79))
        draw.ellipse([spot_x + 15, spot_y - 20, spot_x + 20, spot_y - 15], fill=(47, 79, 79))
        img.save(paths["test_contamination"] / f"{i:03d}.png")

    # Test broken_large images (5)
    for i in range(5):
        img = _create_base_bottle()
        draw = ImageDraw.Draw(img)
        draw.polygon([170, 140, 178, 150, 160, 160], fill=(240, 240, 240), outline=(240, 240, 240))
        draw.line([160, 160, 140, 155], fill=(70, 130, 180), width=1)
        draw.line([160, 160, 150, 175], fill=(70, 130, 180), width=1)
        img.save(paths["test_broken_large"] / f"{i:03d}.png")

    logger.info(f"Synthetic dataset generation complete. Paths created: {list(paths.keys())}")
    return category_dir
