"""CLIP image encoder for generating visual embedding vectors.

Wraps HuggingFace's ``CLIPModel`` and ``CLIPProcessor`` to produce
L2-normalised 512-dimensional vectors suitable for cosine-similarity
retrieval in Qdrant.

Key design choices:
- Model warm-up with a dummy tensor on init to pre-compile CUDA kernels.
- ``encode_images_async`` wraps blocking inference in
  ``asyncio.to_thread`` for non-blocking FastAPI compatibility.
"""

import asyncio
import logging

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.config import settings

logger = logging.getLogger(__name__)


class CLIPEncoder:
    """Encoder class that converts PIL images into normalised CLIP embedding vectors.

    Attributes:
        model_name: HuggingFace identifier of the CLIP checkpoint.
        device: Compute device string (``'cuda'`` or ``'cpu'``).
        processor: HuggingFace CLIP pre-processor (resize / normalise).
        model: HuggingFace CLIPModel loaded in eval mode.
    """

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        """Initialise the CLIP model, processor, and run a warm-up pass.

        Args:
            model_name: HuggingFace model identifier. Defaults to
                ``settings.CLIP_MODEL_NAME``.
            device: Compute device override (``'cuda'`` or ``'cpu'``).
                Defaults to ``settings.DEVICE``.

        Raises:
            RuntimeError: If the model or processor fails to load.
        """
        self.model_name: str = model_name or settings.CLIP_MODEL_NAME
        self.device: str = device or settings.DEVICE

        logger.info(f"Loading CLIP model '{self.model_name}' on device '{self.device}'...")
        try:
            self.processor: CLIPProcessor = CLIPProcessor.from_pretrained(self.model_name)
            self.model: CLIPModel = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            logger.info("CLIP model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading CLIP model: {e}")
            raise

        # Warm-up: single forward pass to pre-compile CUDA kernels and
        # allocate any lazy-initialised GPU memory, reducing first-request
        # latency from ~800ms to ~200ms on typical hardware.
        self._warmup()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_images(self, images: list[Image.Image], batch_size: int = 16) -> np.ndarray:
        """Encode a list of PIL images into normalised embedding vectors.

        Images are processed in batches to prevent GPU out-of-memory errors
        when indexing large datasets.

        Args:
            images: List of PIL ``Image`` objects (any mode — will be
                converted to RGB internally).
            batch_size: Maximum number of images per forward pass.

        Returns:
            ``np.ndarray`` of shape ``(len(images), 512)`` with dtype
            ``float32``, where each row is an L2-normalised embedding.

        Raises:
            RuntimeError: If CLIP inference fails for any batch.
        """
        if not images:
            return np.empty((0, 512), dtype=np.float32)

        embeddings_list: list[np.ndarray] = []

        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]

            # Ensure images are in RGB mode (CLIP model requirement)
            rgb_batch = [img.convert("RGB") if img.mode != "RGB" else img for img in batch]

            try:
                inputs = self.processor(images=rgb_batch, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    image_features = self.model.get_image_features(**inputs)

                    # Handle case where output is a ModelOutput structure instead of raw tensor
                    if not isinstance(image_features, torch.Tensor):
                        if hasattr(image_features, "pooler_output"):
                            image_features = image_features.pooler_output
                        elif isinstance(image_features, dict) and "pooler_output" in image_features:
                            image_features = image_features["pooler_output"]

                    # Apply projection if still in unprojected hidden-dimension state (e.g. 768)
                    if image_features.shape[-1] == 768 and hasattr(self.model, "visual_projection"):
                        image_features = self.model.visual_projection(image_features)

                    # Convert to numpy and normalise to unit vectors (cosine similarity)
                    features_np: np.ndarray = image_features.cpu().numpy()
                    norms = np.linalg.norm(features_np, axis=1, keepdims=True)
                    normalised_features = features_np / (norms + 1e-12)

                    embeddings_list.append(normalised_features)
            except Exception as e:
                logger.error(f"Error encoding batch starting at index {i}: {e}")
                raise

        return np.vstack(embeddings_list)

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Encode a single PIL image.

        Convenience wrapper around :meth:`encode_images`.

        Args:
            image: A single PIL ``Image`` object.

        Returns:
            A 1-D ``np.ndarray`` of shape ``(512,)`` with dtype ``float32``.
        """
        embeddings = self.encode_images([image])
        return embeddings[0]

    async def encode_image_async(self, image: Image.Image) -> np.ndarray:
        """Non-blocking version of :meth:`encode_image`.

        Offloads the blocking PyTorch inference to a thread so the
        FastAPI event loop remains responsive during encoding.

        Args:
            image: A single PIL ``Image`` object.

        Returns:
            A 1-D ``np.ndarray`` of shape ``(512,)`` with dtype ``float32``.
        """
        return await asyncio.to_thread(self.encode_image, image)

    async def encode_images_async(self, images: list[Image.Image], batch_size: int = 16) -> np.ndarray:
        """Non-blocking version of :meth:`encode_images`.

        Args:
            images: List of PIL ``Image`` objects.
            batch_size: Maximum images per forward pass.

        Returns:
            ``np.ndarray`` of shape ``(len(images), 512)`` with dtype ``float32``.
        """
        return await asyncio.to_thread(self.encode_images, images, batch_size)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _warmup(self) -> None:
        """Run a single forward pass with a tiny dummy image.

        This forces CUDA kernel compilation and lazy memory allocation to
        happen at startup rather than on the first user request.
        """
        try:
            dummy = Image.new("RGB", (32, 32), color=(128, 128, 128))
            _ = self.encode_image(dummy)
            logger.info("CLIP warm-up pass completed.")
        except Exception as e:
            logger.warning(f"CLIP warm-up failed (non-critical): {e}")
