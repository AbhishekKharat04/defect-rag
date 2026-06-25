"""End-to-end RAG pipeline tests at the Python-module level (no HTTP layer).

Validates:
1. Model-to-database interaction: encoding an image, indexing it in Qdrant,
   searching for matches, and using the matched context to generate a mock VLM answer.
2. Synchronous vs asynchronous encoding equivalence.
3. Qdrant backend health check utility.
"""

import numpy as np
import pytest
from PIL import Image

from src.embeddings.clip_encoder import CLIPEncoder
from src.vector_db.qdrant_client import QdrantDBClient
from src.vlm.qwen_generator import QwenVLGenerator


@pytest.fixture(scope="module")
def encoder() -> CLIPEncoder:
    """CLIPEncoder instance running on CPU."""
    return CLIPEncoder(device="cpu")


@pytest.fixture(scope="module")
def qdrant_client() -> QdrantDBClient:
    """In-memory Qdrant client."""
    client = QdrantDBClient(location=":memory:")
    yield client
    client.delete_collection("test_pipeline_defects")


@pytest.fixture(scope="module")
def vlm_generator() -> QwenVLGenerator:
    """Mock-mode QwenVLGenerator."""
    return QwenVLGenerator(mode="mock")


@pytest.mark.anyio
async def test_async_encode_matches_sync(encoder: CLIPEncoder):
    """Verifies that async image encoding produces identical vectors to sync encoding."""
    img = Image.new("RGB", (100, 100), color="blue")

    # Sync encoding
    sync_features = encoder.encode_image(img)

    # Async encoding
    async_features = await encoder.encode_image_async(img)

    # Check shape, type, and similarity
    assert sync_features.shape == (512,)
    assert async_features.shape == (512,)
    assert np.allclose(sync_features, async_features, atol=1e-5)


@pytest.mark.anyio
async def test_qdrant_health_check(qdrant_client: QdrantDBClient):
    """Verifies that is_healthy returns True for the in-memory client."""
    healthy = await qdrant_client.is_healthy()
    assert healthy is True


def test_embed_index_search_generate(
    encoder: CLIPEncoder,
    qdrant_client: QdrantDBClient,
    vlm_generator: QwenVLGenerator,
):
    """Simulates the entire RAG pipeline from PIL image to structured VLM answer."""
    col_name = "test_pipeline_defects"

    # 1. Setup the collection
    success = qdrant_client.create_collection(collection_name=col_name, vector_size=512, recreate=True)
    assert success is True

    # 2. Embed and Index a synthetic image (e.g., a broken_small defect)
    img_broken_small = Image.new("RGB", (100, 100), color="red")
    embedding = encoder.encode_image(img_broken_small)

    metadata = {
        "image_path": "/tmp/synthetic_broken_small.png",
        "defect_label": "broken_small",
        "severity": "medium",
    }

    upsert_ok = qdrant_client.upsert_images(
        embeddings=np.expand_dims(embedding, axis=0),
        metadata_list=[metadata],
        ids=[12345],
        collection_name=col_name,
    )
    assert upsert_ok is True

    # 3. Search using the same vector (should return the exact same image as nearest neighbor)
    results = qdrant_client.search_similar(
        query_vector=embedding,
        top_k=1,
        collection_name=col_name,
    )
    assert len(results) == 1
    assert results[0]["id"] == 12345
    assert results[0]["payload"]["defect_label"] == "broken_small"

    # 4. Generate structured VLM answer based on retrieved source
    response = vlm_generator.generate_answer(
        query_image_path="/tmp/query_bottle.png",
        question="Is there a defect? Rate the severity.",
        retrieved_examples=results,
    )

    # Assert response contains required schema keys
    assert "answer" in response
    assert response["predicted_defect"] == "broken_small"
    assert response["predicted_severity"] == "medium"
    assert "confidence" in response
    assert isinstance(response["confidence"], float)
    assert response["retrieved_sources"] == ["/tmp/synthetic_broken_small.png"]
