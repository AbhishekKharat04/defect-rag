import json
import os
import pytest
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor

from scripts.prepare_finetune_data import prepare_finetune_dataset
from scripts.finetune_qwen import QwenVLDataset
from src.config import settings

@pytest.fixture
def mock_dataset(tmp_path):
    """Fixture to set up a mock dataset directory structured like MVTec AD."""
    dataset_dir = tmp_path / "mock_bottle"
    
    # Create splits and category folders
    train_good = dataset_dir / "train" / "good"
    test_broken_small = dataset_dir / "test" / "broken_small"
    
    train_good.mkdir(parents=True)
    test_broken_small.mkdir(parents=True)
    
    # Save a small blank image in each
    img = Image.new("RGB", (100, 100), color="blue")
    img.save(train_good / "000.png")
    img.save(test_broken_small / "001.png")
    
    return dataset_dir

def test_prepare_finetune_dataset(mock_dataset, tmp_path):
    """Tests that prepare_finetune_dataset correctly processes directories and outputs JSON dialogues."""
    output_file = tmp_path / "finetune_data.json"
    
    prepare_finetune_dataset(mock_dataset, output_file)
    
    assert output_file.exists()
    
    # Load and check JSON structure
    with open(output_file, "r") as f:
        dialogues = json.load(f)
        
    assert len(dialogues) == 2
    
    # Check dialog keys
    first_entry = dialogues[0]
    assert "id" in first_entry
    assert "image" in first_entry
    assert "conversations" in first_entry
    assert len(first_entry["conversations"]) == 2
    
    user_msg = first_entry["conversations"][0]
    assistant_msg = first_entry["conversations"][1]
    
    assert user_msg["from"] == "user"
    assert "<image>" in user_msg["value"]
    
    assert assistant_msg["from"] == "assistant"
    assert "DEFECT_LABEL:" in assistant_msg["value"]
    assert "SEVERITY:" in assistant_msg["value"]

def test_qwen_vl_dataset_loader(mock_dataset, tmp_path):
    """Tests loading the custom QwenVLDataset PyTorch wrapper and extracting features."""
    output_file = tmp_path / "finetune_data.json"
    prepare_finetune_dataset(mock_dataset, output_file)
    
    # Initialize real processor in CPU mode
    processor = AutoProcessor.from_pretrained(settings.QWEN_MODEL_NAME)
    
    # Load PyTorch Dataset
    dataset = QwenVLDataset(data_path=str(output_file.resolve()), processor=processor)
    
    assert len(dataset) == 2
    
    # Retrieve first item
    item = dataset[0]
    
    assert "input_ids" in item
    assert "labels" in item
    assert "pixel_values" in item or "image_grid_thw" in item  # Qwen2.5-VL format keys
    
    # Assert sequence dimensions
    assert item["input_ids"].ndim == 1
    assert item["labels"].ndim == 1
    assert item["input_ids"].shape == item["labels"].shape
