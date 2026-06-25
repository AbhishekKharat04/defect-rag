import numpy as np
import pytest
from PIL import Image

from src.embeddings.clip_encoder import CLIPEncoder


@pytest.fixture(scope="module")
def encoder():
    """Fixture to initialize the CLIPEncoder."""
    # We use CPU explicitly for testing to keep VRAM usage low
    return CLIPEncoder(device="cpu")

def test_encode_single_image(encoder):
    """Tests that a single image is successfully encoded to a 512-dim unit vector."""
    # Create a simple red 100x100 image
    img = Image.new("RGB", (100, 100), color="red")

    embedding = encoder.encode_image(img)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (512,)
    assert embedding.dtype == np.float32

    # Assert it is normalized (L2 norm is close to 1)
    norm = np.linalg.norm(embedding)
    assert pytest.approx(norm, rel=1e-5) == 1.0

def test_encode_multiple_images(encoder):
    """Tests batch encoding of multiple images."""
    img1 = Image.new("RGB", (100, 100), color="blue")
    img2 = Image.new("RGB", (100, 100), color="green")

    embeddings = encoder.encode_images([img1, img2], batch_size=2)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, 512)

    # Check norms for all embeddings
    for emb in embeddings:
        norm = np.linalg.norm(emb)
        assert pytest.approx(norm, rel=1e-5) == 1.0
