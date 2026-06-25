import argparse
import json
import logging
import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Logger setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("finetune")

# Redirect cache dirs (imported settings automatically redirects them)
from src.config import settings

class QwenVLDataset(Dataset):
    """Custom PyTorch dataset to load and tokenize Qwen2.5-VL multi-modal dialogue data."""

    def __init__(self, data_path: str, processor: AutoProcessor):
        """
        Args:
            data_path: Path to prepared finetune_data.json.
            processor: Hugging Face QwenAutoProcessor.
        """
        logger.info(f"Loading finetune dataset from: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item["image"]
        conversations = item["conversations"]

        # 1. Load image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            # Fallback to a placeholder image to prevent training crashes
            image = Image.new("RGB", (256, 256), color=(128, 128, 128))

        # 2. Format chat messages
        messages = []
        for msg in conversations:
            role = msg["from"]
            val = msg["value"]
            
            # Map role names ('user' -> 'user', 'assistant' -> 'assistant')
            # The prompt value contains <image> placeholder, Qwen VL processor expects specific tag processing
            messages.append({
                "role": role,
                "content": val
            })

        # Apply processor formatting (chat template + multi-modal preprocessing)
        try:
            text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            
            # The processor automatically handles the image content list conversion when we pass images
            inputs = self.processor(
                text=[text_prompt],
                images=[image],
                padding=True,
                return_tensors="pt"
            )
            
            # Remove batch dimension
            item_inputs = {k: v[0] for k, v in inputs.items()}
            
            # Set labels equal to input_ids for standard causal auto-regressive language modeling
            item_inputs["labels"] = item_inputs["input_ids"].clone()
            
            return item_inputs
        except Exception as e:
            logger.error(f"Error processing item index {idx}: {e}")
            raise e

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-VL using QLoRA.")
    parser.add_argument("--dataset", type=str, default=None, help="Path to prepared finetune data JSON.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save adapters.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per device.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--max-steps", type=int, default=-1, help="Maximum number of training steps (overrides epochs).")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode (useful for testing/CI dry-runs).")
    args = parser.parse_args()

    data_file = args.dataset or str(settings.DATA_DIR / "finetune_data.json")
    output_path = args.output_dir or str(settings.BASE_DIR / "models" / "qwen-vl-adapters")

    # 1. Load Processor
    logger.info(f"Loading processor for: {settings.QWEN_MODEL_NAME}")
    processor = AutoProcessor.from_pretrained(settings.QWEN_MODEL_NAME)

    # 2. BitsAndBytes 4-bit configuration (only if GPU is available and CPU mode is not forced)
    use_quantization = torch.cuda.is_available() and not args.cpu
    
    if use_quantization:
        logger.info("Initializing 4-bit BitsAndBytes quantization configuration...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
    else:
        logger.info("Quantization disabled (running in CPU / non-CUDA fallback)...")
        bnb_config = None

    # 3. Load Model
    logger.info(f"Loading base model: {settings.QWEN_MODEL_NAME}")
    device_map = "auto" if torch.cuda.is_available() and not args.cpu else None
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        settings.QWEN_MODEL_NAME,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.float16 if torch.cuda.is_available() and not args.cpu else torch.float32
    )

    # 4. Configure PEFT (LoRA)
    if use_quantization:
        model = prepare_model_for_kbit_training(model)

    logger.info("Configuring LoRA (PEFT)...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 5. Create Dataset
    train_dataset = QwenVLDataset(data_path=data_file, processor=processor)

    # 6. Configure Training Arguments
    training_args = TrainingArguments(
        output_dir=output_path,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        fp16=torch.cuda.is_available() and not args.cpu,
        logging_steps=1,
        save_strategy="no",
        max_steps=args.max_steps,
        remove_unused_columns=False, # Required for multimodal data loader to keep image tensors
        use_cpu=args.cpu
    )

    # Custom data collator to handle sequence packaging and padding for multimodal tensors
    def col_fn(features):
        batch = {}
        for k in features[0].keys():
            # Pad sequences
            tensors = [f[k] for f in features]
            if isinstance(tensors[0], torch.Tensor):
                # Padding token IDs or labels with processor pad values
                if k in ["input_ids", "labels"]:
                    pad_val = processor.tokenizer.pad_token_id if k == "input_ids" else -100
                    batch[k] = torch.nn.utils.rnn.pad_sequence(
                        tensors, batch_first=True, padding_value=pad_val
                    )
                else:
                    batch[k] = torch.stack(tensors)
            else:
                batch[k] = tensors
        return batch

    # 7. Start Training
    logger.info("Initializing SFT Trainer and starting fine-tuning...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=col_fn
    )

    try:
        trainer.train()
        logger.info(f"Training completed successfully. Saving LoRA adapter weights to {output_path}...")
        model.save_pretrained(output_path)
        logger.info("Fine-tuning pipeline finished.")
    except Exception as e:
        logger.error(f"Error during training loop: {e}")
        raise e

if __name__ == "__main__":
    main()
