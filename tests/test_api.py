"""Integration tests for the FastAPI backend endpoints.

Tests the complete HTTP request/response cycle using ``TestClient``,
validating Pydantic response schemas, request-ID middleware, file
validation, CORS headers, and operational metrics.
"""

import io

from tests.conftest import _make_png_bytes

# =========================================================================
# /health
# =========================================================================

class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, test_client):
        """Health endpoint responds with 200 and correct schema."""
        resp = test_client.get("/health")
        assert resp.status_code == 200

        body = resp.json()
        assert body["status"] == "healthy"
        assert "version" in body
        assert body["device"] in ("cuda", "cpu")
        assert "clip_model" in body
        assert "qwen_model" in body
        assert "qdrant" in body
        assert "status" in body["qdrant"]

    def test_health_includes_version(self, test_client):
        """Health response includes the application version string."""
        from src.config import VERSION

        resp = test_client.get("/health")
        assert resp.json()["version"] == VERSION


# =========================================================================
# /metrics
# =========================================================================

class TestMetricsEndpoint:
    """Tests for GET /metrics."""

    def test_metrics_returns_200(self, test_client):
        """Metrics endpoint responds with operational counters."""
        resp = test_client.get("/metrics")
        assert resp.status_code == 200

        body = resp.json()
        assert "uptime_seconds" in body
        assert body["uptime_seconds"] > 0
        assert body["total_queries"] >= 0
        assert body["total_index_requests"] >= 0


# =========================================================================
# Request-ID middleware
# =========================================================================

class TestRequestIDMiddleware:
    """Tests for the X-Request-ID middleware."""

    def test_response_has_request_id(self, test_client):
        """Every response includes an X-Request-ID header."""
        resp = test_client.get("/health")
        assert "x-request-id" in resp.headers

    def test_custom_request_id_echoed(self, test_client):
        """A client-provided X-Request-ID is echoed back."""
        custom_id = "test-trace-12345"
        resp = test_client.get("/health", headers={"X-Request-ID": custom_id})
        assert resp.headers["x-request-id"] == custom_id


# =========================================================================
# CORS
# =========================================================================

class TestCORSHeaders:
    """Tests for CORS middleware configuration."""

    def test_cors_allows_origin(self, test_client):
        """Preflight OPTIONS request returns appropriate CORS headers."""
        resp = test_client.options(
            "/health",
            headers={
                "Origin": "http://localhost:7860",
                "Access-Control-Request-Method": "GET",
            },
        )
        # CORSMiddleware should respond with access-control-allow-origin
        assert "access-control-allow-origin" in resp.headers


# =========================================================================
# File validation
# =========================================================================

class TestFileValidation:
    """Tests for upload file validation (content-type and size)."""

    def test_rejects_non_image_file(self, test_client):
        """Uploading a non-image file to /query returns 422."""
        txt_bytes = b"this is not an image"
        resp = test_client.post(
            "/query",
            files={"file": ("notes.txt", io.BytesIO(txt_bytes), "text/plain")},
            data={"question": "what defect is this?"},
        )
        assert resp.status_code == 422
        assert "Unsupported file type" in resp.json()["detail"]

    def test_rejects_oversized_file(self, test_client):
        """Uploading a file larger than 10 MB to /query returns 422."""
        # Create an 11 MB payload (doesn't need to be a real image —
        # size check runs before PIL parsing)
        oversized = b"\x00" * (11 * 1024 * 1024)
        resp = test_client.post(
            "/query",
            files={"file": ("huge.png", io.BytesIO(oversized), "image/png")},
            data={"question": "what defect is this?"},
        )
        assert resp.status_code == 422
        assert "10 MB" in resp.json()["detail"]


# =========================================================================
# /index/dataset
# =========================================================================

class TestIndexEndpoint:
    """Tests for POST /index/dataset."""

    def test_index_synthetic_dataset(self, test_client):
        """Indexing a synthetic dataset returns success with vector count > 0."""
        resp = test_client.post(
            "/index/dataset",
            params={"category": "bottle", "synthetic": True, "recreate": True},
        )
        assert resp.status_code == 200

        body = resp.json()
        assert body["status"] == "success"
        assert "collection_info" in body
        assert body["collection_info"]["points_count"] > 0


# =========================================================================
# /search
# =========================================================================

class TestSearchEndpoint:
    """Tests for POST /search."""

    def test_search_returns_matches(self, seeded_test_client):
        """Searching with a valid image returns retrieved matches."""
        img_bytes = _make_png_bytes("blue")
        resp = seeded_test_client.post(
            "/search",
            files={"file": ("part.png", io.BytesIO(img_bytes), "image/png")},
            params={"top_k": 3},
        )
        assert resp.status_code == 200

        body = resp.json()
        assert "results" in body
        assert len(body["results"]) > 0

        # Each result should have id, score, payload
        first = body["results"][0]
        assert "id" in first
        assert "score" in first
        assert "payload" in first
        assert "defect_label" in first["payload"]


# =========================================================================
# /query (full RAG pipeline)
# =========================================================================

class TestQueryEndpoint:
    """Tests for POST /query — the full RAG pipeline."""

    def test_query_returns_structured_response(self, seeded_test_client):
        """A valid query returns the full QueryResponse schema."""
        img_bytes = _make_png_bytes("green")
        resp = seeded_test_client.post(
            "/query",
            files={"file": ("part.png", io.BytesIO(img_bytes), "image/png")},
            data={
                "question": "Is this part defective? What type of defect?",
                "top_k": 3,
            },
        )
        assert resp.status_code == 200

        body = resp.json()
        # All QueryResponse fields must be present
        assert "query" in body
        assert "answer" in body
        assert "predicted_defect" in body
        assert "predicted_severity" in body
        assert "confidence" in body
        assert isinstance(body["confidence"], float)
        assert 0.0 <= body["confidence"] <= 1.0
        assert "retrieved_matches" in body

    def test_query_increments_metrics(self, seeded_test_client):
        """After a /query call, /metrics should reflect the new query count."""
        # Get baseline
        baseline = seeded_test_client.get("/metrics").json()
        baseline_count = baseline["total_queries"]

        # Fire a query
        img_bytes = _make_png_bytes("yellow")
        seeded_test_client.post(
            "/query",
            files={"file": ("part.png", io.BytesIO(img_bytes), "image/png")},
            data={"question": "check for defects", "top_k": 2},
        )

        # Verify incremented counter
        updated = seeded_test_client.get("/metrics").json()
        assert updated["total_queries"] == baseline_count + 1
        assert updated["average_query_latency_seconds"] is not None
        assert updated["average_query_latency_seconds"] > 0


# =========================================================================
# Baselines & Threshold Gates
# =========================================================================

class TestRAGBaselinesAndThresholds:
    """Tests checking baseline defect classifications and similarity gate logic."""

    def test_good_part_baseline(self, seeded_test_client):
        """Uploading a conforming (good) part returns 'no defect detected'."""
        from pathlib import Path
        img_path = Path("data/bottle/train/good/000.png")
        if not img_path.exists():
            img_bytes = _make_png_bytes("white")
            file_obj = io.BytesIO(img_bytes)
        else:
            with open(img_path, "rb") as f:
                img_bytes = f.read()
            file_obj = io.BytesIO(img_bytes)

        resp = seeded_test_client.post(
            "/query",
            files={"file": ("good.png", file_obj, "image/png")},
            data={"question": "is this part defective?", "top_k": 3},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["predicted_defect"] == "no defect detected"
        assert body["predicted_severity"] == "none"
        assert body["confidence"] >= 0.90
        assert body["answer"] != "insufficient reference data"

    def test_broken_part_baseline(self, seeded_test_client):
        """Uploading a clearly broken part returns the correct defect type 'broken_large'."""
        from pathlib import Path
        img_path = Path("data/bottle/test/broken_large/000.png")
        if not img_path.exists():
            img_bytes = _make_png_bytes("red")
            file_obj = io.BytesIO(img_bytes)
        else:
            with open(img_path, "rb") as f:
                img_bytes = f.read()
            file_obj = io.BytesIO(img_bytes)

        resp = seeded_test_client.post(
            "/query",
            files={"file": ("broken_large.png", file_obj, "image/png")},
            data={"question": "what defect does this bottle have?", "top_k": 3},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["predicted_defect"] == "broken_large"
        assert body["predicted_severity"] == "high"
        assert body["answer"] != "insufficient reference data"

    def test_similarity_gate_triggered(self, seeded_test_client):
        """If similarity score is below 0.75, bypass VLM and return 'insufficient reference data'."""
        img_bytes = _make_png_bytes("green")
        resp = seeded_test_client.post(
            "/query",
            files={"file": ("unknown_part.png", io.BytesIO(img_bytes), "image/png")},
            data={"question": "is there any defect?", "top_k": 3},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "insufficient reference data"
        assert body["predicted_defect"] == "unknown"
        assert body["predicted_severity"] == "none"
        assert body["confidence"] == 0.0
