import argparse
import logging
import os
from pathlib import Path

from PIL import Image

# Set up logging before other imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("index_dataset")

from src.config import settings
from src.embeddings.clip_encoder import CLIPEncoder
from src.utils.data_loader import download_mvtec_category, generate_synthetic_dataset
from src.vector_db.qdrant_client import QdrantDBClient


def main():
    parser = argparse.ArgumentParser(description="Download and index industrial defect dataset into Qdrant.")
    parser.add_argument("--category", type=str, default="bottle", help="Dataset category (e.g. bottle).")
    parser.add_argument("--synthetic", action="store_true", help="Force generation of synthetic dataset instead of downloading.")
    parser.add_argument("--recreate", action="store_true", help="Recreate Qdrant collection if it already exists.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for generating embeddings.")
    args = parser.parse_args()

    # 1. Download or generate dataset
    if args.synthetic:
        dataset_path = generate_synthetic_dataset(category=args.category)
    else:
        dataset_path = download_mvtec_category(category=args.category)

    logger.info(f"Dataset path resolved: {dataset_path}")

    # 2. Initialize CLIP Encoder & Qdrant Client
    encoder = CLIPEncoder()
    qdrant = QdrantDBClient()

    # 3. Create Qdrant Collection
    qdrant.create_collection(
        collection_name=settings.QDRANT_COLLECTION,
        vector_size=512,
        recreate=args.recreate
    )

    # 4. Scan files and compile paths & metadata
    image_paths = []
    metadata_list = []

    # Walk through the dataset directory structure: <category>/<split>/<defect_label>/<filename>
    for root, _dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                full_path = Path(root) / file

                # Parse structure
                parts = full_path.relative_to(dataset_path).parts
                if len(parts) >= 2:
                    split = parts[0]          # 'train' or 'test'
                    defect_label = parts[1]   # 'good', 'scratch', 'broken', etc.

                    # Estimate severity based on defect label
                    if defect_label == "good":
                        severity = "none"
                    elif defect_label in ["broken", "broken_large", "hole", "fracture"]:
                        severity = "high"
                    else:
                        severity = "medium"

                    image_paths.append(full_path)
                    metadata_list.append({
                        "image_path": str(full_path.resolve()),
                        "category": args.category,
                        "split": split,
                        "defect_label": defect_label,
                        "severity": severity,
                    })

    total_images = len(image_paths)
    if total_images == 0:
        logger.warning(f"No images found in {dataset_path}. Exiting.")
        return

    logger.info(f"Found {total_images} images to index.")

    # 5. Extract embeddings and index in batches
    for i in range(0, total_images, args.batch_size):
        batch_paths = image_paths[i : i + args.batch_size]
        batch_metadata = metadata_list[i : i + args.batch_size]

        # Load images
        images = []
        valid_indices = []
        for idx, path in enumerate(batch_paths):
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
                valid_indices.append(idx)
            except Exception as e:
                logger.error(f"Error opening image {path}: {e}")

        if not images:
            continue

        # Filter metadata for successfully opened images
        batch_metadata_filtered = [batch_metadata[idx] for idx in valid_indices]

        try:
            # Encode images
            logger.info(f"Encoding batch {i//args.batch_size + 1} ({len(images)} images)...")
            embeddings = encoder.encode_images(images, batch_size=args.batch_size)

            # Upsert into Qdrant
            qdrant.upsert_images(
                embeddings=embeddings,
                metadata_list=batch_metadata_filtered,
                collection_name=settings.QDRANT_COLLECTION
            )
        except Exception as e:
            logger.error(f"Failed to process batch starting at index {i}: {e}")
            raise e

    # 6. Verify collection status
    info = qdrant.get_collection_info(settings.QDRANT_COLLECTION)
    if info:
        logger.info(f"Indexing complete! Collection Info: {info}")
    else:
        logger.error("Failed to retrieve collection info after indexing.")

if __name__ == "__main__":
    main()
