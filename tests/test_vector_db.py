import numpy as np
import pytest

from src.vector_db.qdrant_client import QdrantDBClient


@pytest.fixture
def qdrant_client():
    """Fixture to initialize QdrantDBClient in memory."""
    client = QdrantDBClient(host=":memory:")
    yield client
    # Clean up collections after test
    client.delete_collection("test_defects")

def test_collection_creation_and_deletion(qdrant_client):
    """Tests creating and deleting collections in Qdrant."""
    col_name = "test_defects"

    # Assert collection does not exist initially
    info = qdrant_client.get_collection_info(col_name)
    assert info is None

    # Create collection
    success = qdrant_client.create_collection(collection_name=col_name, vector_size=512)
    assert success is True

    # Verify collection exists and contains 0 points
    info = qdrant_client.get_collection_info(col_name)
    assert info is not None
    assert info["status"] in ["green", "ok"]
    assert info["points_count"] == 0

    # Delete collection
    success = qdrant_client.delete_collection(col_name)
    assert success is True

    # Verify it is gone
    info = qdrant_client.get_collection_info(col_name)
    assert info is None

def test_upsert_and_search(qdrant_client):
    """Tests upserting vectors with payload and searching for matches."""
    col_name = "test_defects"
    qdrant_client.create_collection(collection_name=col_name, vector_size=512)

    # Create two synthetic normalized embeddings
    v1 = np.random.randn(512).astype(np.float32)
    v1 = v1 / np.linalg.norm(v1)

    v2 = np.random.randn(512).astype(np.float32)
    v2 = v2 / np.linalg.norm(v2)

    embeddings = np.vstack([v1, v2])

    payloads = [
        {"image_path": "/path/to/defect1.png", "defect_label": "scratch", "severity": "high"},
        {"image_path": "/path/to/defect2.png", "defect_label": "dent", "severity": "medium"}
    ]

    # Upsert points
    upsert_success = qdrant_client.upsert_images(
        embeddings=embeddings,
        metadata_list=payloads,
        collection_name=col_name
    )
    assert upsert_success is True

    # Verify points count
    info = qdrant_client.get_collection_info(col_name)
    assert info["points_count"] == 2

    # Search using v1 as query (should match v1 with highest score)
    results = qdrant_client.search_similar(
        query_vector=v1,
        top_k=2,
        collection_name=col_name
    )

    assert len(results) == 2
    # The first result should be v1 (since cosine similarity with itself is 1.0)
    assert results[0]["payload"]["defect_label"] == "scratch"
    assert results[0]["payload"]["severity"] == "high"
    assert pytest.approx(results[0]["score"], rel=1e-3) == 1.0

    # Search with visual metadata filtering (filter only 'dent' label)
    filtered_results = qdrant_client.search_similar(
        query_vector=v1,
        top_k=2,
        collection_name=col_name,
        filter_dict={"defect_label": "dent"}
    )

    assert len(filtered_results) == 1
    assert filtered_results[0]["payload"]["defect_label"] == "dent"
