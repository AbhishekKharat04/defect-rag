"""Shared pytest fixtures for the Vision-Language RAG test suite.

Provides:
- ``test_client`` — a FastAPI ``TestClient`` with in-memory Qdrant,
  CPU-only CLIP encoder, and mock VLM injected (no Docker / GPU needed).
- ``seeded_test_client`` — same as above but with a pre-indexed synthetic
  dataset so retrieval endpoints return real matches.
- ``sample_image_bytes`` / ``sample_image_file`` — synthetic PNG test images.
"""

import io
import os
from typing import Generator

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


# ---------------------------------------------------------------------------
# Synthetic image helpers
# ---------------------------------------------------------------------------

def _make_png_bytes(color: str = "red", size: tuple = (100, 100)) -> bytes:
    """Create a minimal PNG image in memory and return its raw bytes."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


@pytest.fixture()
def sample_image_bytes() -> bytes:
    """A small synthetic red PNG image as raw bytes."""
    return _make_png_bytes("red")


@pytest.fixture()
def sample_image_file(sample_image_bytes: bytes):
    """A ``(filename, BytesIO, content_type)`` tuple ready for ``TestClient.post(files=...)``."""
    return ("test_part.png", io.BytesIO(sample_image_bytes), "image/png")


# ---------------------------------------------------------------------------
# FastAPI TestClient with lightweight in-memory components
# ---------------------------------------------------------------------------

def _build_test_app():
    """Construct a FastAPI ``TestClient`` backed by in-memory / CPU / mock components.

    Instead of running the normal lifespan (which may download models or
    connect to Docker), we directly inject lightweight instances into the
    module-level globals that the endpoint handlers reference.
    """
    # Import the app *module* so we can set its globals
    from src.api import main as api_module

    # 1. CLIP Encoder on CPU
    from src.embeddings.clip_encoder import CLIPEncoder
    api_module.encoder = CLIPEncoder(device="cpu")

    # 2. In-memory Qdrant
    from src.vector_db.qdrant_client import QdrantDBClient
    api_module.qdrant = QdrantDBClient(location=":memory:")

    # 3. Mock VLM
    from src.vlm.qwen_generator import QwenVLGenerator
    api_module.vlm = QwenVLGenerator(mode="mock")

    # 4. MLflow tracker (optional — can be None for speed, but let's keep it)
    from src.utils.tracker import MLflowTracker
    api_module.tracker = MLflowTracker(experiment_name="Test_API_Integration")

    # 5. Reset operational counters
    import time
    api_module._start_time = time.time()
    api_module._total_queries = 0
    api_module._total_index_requests = 0
    api_module._query_latencies.clear()

    # Build the TestClient with raise_server_exceptions so assertion errors
    # propagate instead of returning opaque 500s.
    return TestClient(api_module.app, raise_server_exceptions=False)


@pytest.fixture(scope="module")
def test_client() -> Generator[TestClient, None, None]:
    """A ``TestClient`` wired to in-memory Qdrant, CPU CLIP, and mock VLM.

    The collection is **empty** — use ``seeded_test_client`` if you need
    pre-indexed vectors for retrieval tests.
    """
    client = _build_test_app()
    yield client


@pytest.fixture(scope="module")
def seeded_test_client() -> Generator[TestClient, None, None]:
    """A ``TestClient`` with a pre-indexed synthetic dataset.

    Indexes the ``bottle`` synthetic dataset (40 images) into the
    in-memory Qdrant collection so ``/search`` and ``/query`` return
    real matches.
    """
    client = _build_test_app()

    # Trigger synthetic dataset indexing via the API endpoint
    resp = client.post(
        "/index/dataset",
        params={"category": "bottle", "synthetic": True, "recreate": True},
    )
    assert resp.status_code == 200, f"Seeding failed: {resp.text}"

    yield client
