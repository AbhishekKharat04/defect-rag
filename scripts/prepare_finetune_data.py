import argparse
import json
import logging
import os
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("prepare_data")

from src.config import settings

def prepare_finetune_dataset(dataset_dir: Path, output_file: Path):
    """Walks the dataset folder and creates standard Qwen-VL chat data."""
    if not dataset_dir.exists():
        logger.error(f"Dataset directory does not exist: {dataset_dir}")
        return

    logger.info(f"Scanning images in {dataset_dir}...")
    dialogues = []
    index = 0

    # Walk dataset directory
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                full_path = Path(root) / file
                
                # Deduce split and label from path
                relative_path = full_path.relative_to(dataset_dir)
                parts = relative_path.parts
                
                if len(parts) >= 2:
                    split = parts[0]          # 'train' or 'test'
                    defect_label = parts[1]   # 'good', 'scratch', 'broken', etc.
                    
                    # Deduce severity
                    if defect_label == "good":
                        severity = "none"
                    elif defect_label in ["broken", "broken_large", "hole"]:
                        severity = "high"
                    else:
                        severity = "medium"

                    # Build conversational dialogue structure
                    dialogue_entry = {
                        "id": f"defect_{index:05d}",
                        "image": str(full_path.resolve()),
                        "conversations": [
                            {
                                "from": "user",
                                "value": (
                                    "<image>\n"
                                    "Analyze this manufactured part image for quality control. "
                                    "Is it defective? If yes, what is the defect type and its severity level?"
                                )
                            },
                            {
                                "from": "assistant",
                                "value": (
                                    f"Visual quality analysis reveals the following status:\n"
                                    f"The part is classified with the defect type: '{defect_label}'.\n"
                                    f"The estimated severity level is: '{severity}'.\n\n"
                                    f"```\n"
                                    f"DEFECT_LABEL: {defect_label}\n"
                                    f"SEVERITY: {severity}\n"
                                    f"CONFIDENCE: 1.0\n"
                                    f"```"
                                )
                            }
                        ]
                    }
                    dialogues.append(dialogue_entry)
                    index += 1

    # Save to JSON
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dialogues, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Prepared {len(dialogues)} instruction dialogue entries. Saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Prepare instruction data for VLM fine-tuning.")
    parser.add_argument("--category", type=str, default="bottle", help="Dataset category.")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path.")
    args = parser.parse_args()

    dataset_path = settings.DATA_DIR / args.category
    output_path = Path(args.output) if args.output else settings.DATA_DIR / "finetune_data.json"

    prepare_finetune_dataset(dataset_path, output_path)

if __name__ == "__main__":
    main()
