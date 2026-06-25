"""FastAPI backend for the Vision-Language RAG defect detection pipeline.

This module wires together the CLIP encoder, Qdrant vector database, and
Qwen2.5-VL generator into a REST API with:

- **Typed Pydantic response models** (visible in ``/docs`` Swagger UI).
- **Request-ID middleware** for end-to-end tracing in structured logs.
- **File validation** (content-type whitelist, 10 MB size cap).
- **CORS middleware** for cross-origin frontend requests.
- **``/metrics``** endpoint with basic operational counters.
"""

import json
import logging
import os
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

from src.config import VERSION, settings, setup_logging
from src.embeddings.clip_encoder import CLIPEncoder
from src.utils.data_loader import download_mvtec_category, generate_synthetic_dataset
from src.utils.tracker import MLflowTracker
from src.vector_db.qdrant_client import QdrantDBClient
from src.vlm.qwen_generator import QwenVLGenerator

# ---------------------------------------------------------------------------
# Structured logging bootstrap
# ---------------------------------------------------------------------------
setup_logging()
logger = logging.getLogger("api")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TEMP_UPLOAD_DIR = settings.DATA_DIR / "uploads"
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_LOG_FILE = settings.BASE_DIR / "logs" / "predictions.jsonl"

#: Allowed MIME types for uploaded images.
ALLOWED_CONTENT_TYPES = frozenset({
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/bmp",
    "image/tiff",
})

#: Maximum upload file size in bytes (10 MB).
MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024

# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class QdrantStatusModel(BaseModel):
    """Nested model describing Qdrant connection state."""

    status: str = Field(..., description="Connection status string.")
    collection_info: Optional[Dict[str, Any]] = Field(
        None, description="Collection metadata (points count, status)."
    )


class HealthResponse(BaseModel):
    """Response schema for ``GET /health``."""

    status: str = Field("healthy", description="API health status.")
    version: str = Field(..., description="Application version.")
    device: str = Field(..., description="Active compute device (cuda / cpu).")
    clip_model: str = Field(..., description="CLIP model identifier.")
    qwen_model: str = Field(..., description="Qwen VLM model identifier.")
    qdrant: QdrantStatusModel


class RetrievedMatchPayload(BaseModel):
    """Payload associated with a single retrieved Qdrant vector."""

    image_path: str
    defect_label: str = "unknown"
    severity: str = "unknown"
    category: Optional[str] = None
    split: Optional[str] = None


class RetrievedMatch(BaseModel):
    """A single retrieved visual neighbour from the vector database."""

    id: Any
    score: float = Field(..., description="Cosine similarity score (0–1).")
    payload: RetrievedMatchPayload


class QueryResponse(BaseModel):
    """Response schema for ``POST /query``."""

    query: str = Field(..., description="The user's text question.")
    answer: str = Field(..., description="VLM-generated analysis narrative.")
    predicted_defect: str = Field(..., description="Predicted defect class label.")
    predicted_severity: str = Field(..., description="Predicted severity level.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="VLM confidence score.")
    retrieved_matches: List[Dict[str, Any]] = Field(
        default_factory=list, description="Top-K visual reference matches."
    )


class IndexResponse(BaseModel):
    """Response schema for ``POST /index/dataset``."""

    status: str
    message: str
    collection_info: Optional[Dict[str, Any]] = None


class MetricsResponse(BaseModel):
    """Response schema for ``GET /metrics``."""

    uptime_seconds: float
    total_queries: int
    total_index_requests: int
    average_query_latency_seconds: Optional[float]


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    error_code: str
    detail: str


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------
encoder: Optional[CLIPEncoder] = None
qdrant: Optional[QdrantDBClient] = None
vlm: Optional[QwenVLGenerator] = None
tracker: Optional[MLflowTracker] = None

# Operational counters
_start_time: float = 0.0
_total_queries: int = 0
_total_index_requests: int = 0
_query_latencies: List[float] = []


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise ML components at startup and tear down on shutdown."""
    global encoder, qdrant, vlm, tracker, _start_time
    _start_time = time.time()
    logger.info("Starting up Vision-Language RAG API...")

    try:
        encoder = CLIPEncoder()
        qdrant = QdrantDBClient()
        vlm = QwenVLGenerator()
        tracker = MLflowTracker()
    except Exception as e:
        logger.error(f"Error initializing RAG components: {e}")

    yield
    logger.info("Shutting down Vision-Language RAG API...")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Vision-Language RAG Assistant API",
    description=(
        "Production-grade API for industrial defect detection "
        "using CLIP retrieval and Qwen2.5-VL generation."
    ),
    version=VERSION,
    lifespan=lifespan,
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
)

# CORS — allow the Gradio frontend (and dev tools) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request-ID middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Inject a unique request ID into every request/response cycle.

    The ID is:
    - Stored on ``request.state.request_id`` for handler access.
    - Returned in the ``X-Request-ID`` response header.
    - Attached to all log records emitted during the request via a
      logging filter.
    """
    rid = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = rid

    # Attach request_id to the root logger so _JSONFormatter can pick it up
    _rid_filter = _RequestIDFilter(rid)
    logging.getLogger().addFilter(_rid_filter)

    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response
    finally:
        logging.getLogger().removeFilter(_rid_filter)


class _RequestIDFilter(logging.Filter):
    """Logging filter that injects ``request_id`` into every record."""

    def __init__(self, request_id: str) -> None:
        super().__init__()
        self.request_id = request_id

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        record.request_id = self.request_id  # type: ignore[attr-defined]
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_encoder() -> CLIPEncoder:
    if encoder is None:
        raise HTTPException(status_code=503, detail="CLIP Encoder model is not loaded.")
    return encoder


def _get_qdrant() -> QdrantDBClient:
    if qdrant is None:
        raise HTTPException(status_code=503, detail="Qdrant DB Client is not connected.")
    return qdrant


def _get_vlm() -> QwenVLGenerator:
    if vlm is None:
        raise HTTPException(status_code=503, detail="QwenVL Generator is not initialized.")
    return vlm


def _validate_upload(file: UploadFile) -> None:
    """Validate uploaded file content-type and size.

    Raises:
        HTTPException: If the file is not an image or exceeds 10 MB.
    """
    content_type = file.content_type or ""
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{content_type}'. Allowed: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}.",
        )

    # Read the file to check size (seek back afterwards)
    file.file.seek(0, 2)  # seek to end
    size = file.file.tell()
    file.file.seek(0)  # reset

    if size > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=422,
            detail=f"File size {size / (1024 * 1024):.1f} MB exceeds the 10 MB limit.",
        )


def _log_prediction(
    question: str,
    defect_label: str,
    severity: str,
    confidence: float,
    latency: float,
    matches: List[Dict[str, Any]],
) -> None:
    """Append a prediction record to the local JSONL audit log."""
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "predicted_defect": defect_label,
        "predicted_severity": severity,
        "confidence": confidence,
        "latency_sec": latency,
        "retrieved_matches": [
            {
                "score": m["score"],
                "label": m["payload"]["defect_label"],
                "severity": m["payload"]["severity"],
            }
            for m in matches
        ],
    }
    try:
        with open(PREDICTIONS_LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logger.error(f"Failed to write prediction log: {e}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Check the health of the API and Qdrant connection."""
    qdrant_status = "disconnected"
    qdrant_info: Optional[Dict[str, Any]] = None

    if qdrant is not None:
        try:
            qdrant_info = await qdrant.get_collection_info_async()
            qdrant_status = "connected" if qdrant_info else "connected (no collection)"
        except Exception as e:
            qdrant_status = f"error: {e!s}"

    return HealthResponse(
        status="healthy",
        version=VERSION,
        device=settings.DEVICE,
        clip_model=settings.CLIP_MODEL_NAME,
        qwen_model=settings.QWEN_MODEL_NAME,
        qdrant=QdrantStatusModel(status=qdrant_status, collection_info=qdrant_info),
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["System"])
async def get_metrics() -> MetricsResponse:
    """Return operational metrics for monitoring dashboards."""
    avg_latency: Optional[float] = None
    if _query_latencies:
        avg_latency = sum(_query_latencies) / len(_query_latencies)

    return MetricsResponse(
        uptime_seconds=round(time.time() - _start_time, 2),
        total_queries=_total_queries,
        total_index_requests=_total_index_requests,
        average_query_latency_seconds=round(avg_latency, 4) if avg_latency else None,
    )


@app.post("/index/dataset", response_model=IndexResponse, tags=["Indexing"])
async def index_dataset(
    category: str = Query("bottle", description="The MVTec category to index."),
    synthetic: bool = Query(False, description="Whether to force synthetic dataset generation."),
    recreate: bool = Query(False, description="Whether to recreate the collection, clearing previous vectors."),
) -> IndexResponse:
    """Download (or generate) and index a dataset category into Qdrant."""
    global _total_index_requests
    _total_index_requests += 1

    logger.info(f"Indexing request received for category: {category}")

    # 1. Download or generate data
    if synthetic:
        data_path = generate_synthetic_dataset(category=category)
    else:
        data_path = download_mvtec_category(category=category)

    db = _get_qdrant()
    enc = _get_encoder()

    # 2. Recreate collection if requested
    db.create_collection(
        collection_name=settings.QDRANT_COLLECTION,
        vector_size=512,
        recreate=recreate,
    )

    # 3. Read image paths and metadata
    image_paths: List[Path] = []
    metadata_list: List[Dict[str, Any]] = []

    for root, _dirs, files in os.walk(data_path):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                full_path = Path(root) / file
                parts = full_path.relative_to(data_path).parts
                if len(parts) >= 2:
                    split = parts[0]
                    defect_label = parts[1]
                    severity = (
                        "none"
                        if defect_label == "good"
                        else ("high" if defect_label in ("broken", "broken_large", "hole") else "medium")
                    )

                    image_paths.append(full_path)
                    metadata_list.append({
                        "image_path": str(full_path.resolve()),
                        "category": category,
                        "split": split,
                        "defect_label": defect_label,
                        "severity": severity,
                    })

    total = len(image_paths)
    if total == 0:
        return IndexResponse(
            status="error",
            message=f"No images found to index in {data_path}.",
        )

    # 4. Batch index images
    batch_size = 16
    for i in range(0, total, batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_meta = metadata_list[i : i + batch_size]

        images: List[Image.Image] = []
        valid_indices: List[int] = []
        for idx, path in enumerate(batch_paths):
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
                valid_indices.append(idx)
            except Exception as e:
                logger.error(f"Error reading image {path}: {e}")

        if not images:
            continue

        batch_meta_filtered = [batch_meta[idx] for idx in valid_indices]
        embeddings = enc.encode_images(images, batch_size=batch_size)
        db.upsert_images(embeddings=embeddings, metadata_list=batch_meta_filtered)

    info = db.get_collection_info()
    return IndexResponse(
        status="success",
        message=f"Indexed dataset category '{category}' successfully.",
        collection_info=info,
    )


@app.post("/search", tags=["Retrieval"])
async def search_similar(
    file: UploadFile = File(...),
    top_k: int = Query(settings.TOP_K, description="Number of visual matches to retrieve."),
) -> JSONResponse:
    """Retrieve top-K visually similar images from Qdrant for an uploaded query image."""
    _validate_upload(file)
    enc = _get_encoder()
    db = _get_qdrant()

    try:
        img = Image.open(file.file).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    query_vector = await enc.encode_image_async(img)
    results = await db.search_similar_async(query_vector=query_vector, top_k=top_k)
    return JSONResponse(content={"results": results})


@app.post("/query", response_model=QueryResponse, tags=["RAG Pipeline"])
async def run_rag_query(
    file: UploadFile = File(..., description="Query image of the manufactured part."),
    question: str = Form(..., description="Text question (e.g. 'what defect is this?')."),
    top_k: int = Form(settings.TOP_K, description="Number of visual examples to retrieve for context."),
) -> QueryResponse:
    """Run the full RAG pipeline: embed → retrieve → generate grounded answer."""
    global _total_queries
    _total_queries += 1
    start_time = time.time()

    _validate_upload(file)
    enc = _get_encoder()
    db = _get_qdrant()
    vlm_model = _get_vlm()

    # 1. Save uploaded image to temp file (needed by VLM pipelines)
    temp_path = TEMP_UPLOAD_DIR / f"{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save query file: {e}")

    try:
        # 2. Load query image and compute embedding
        img = Image.open(temp_path).convert("RGB")
        query_vector = await enc.encode_image_async(img)

        # 3. Retrieve top-K similar database images
        retrieved_examples = await db.search_similar_async(query_vector=query_vector, top_k=top_k)

        # Similarity threshold gate
        max_similarity = 0.0
        if retrieved_examples:
            max_similarity = max(ex.get("score", 0.0) for ex in retrieved_examples)

        if max_similarity < 0.75:
            logger.warning(
                f"Similarity score ({max_similarity:.4f}) below gate threshold (0.75). "
                "Returning insufficient reference data."
            )
            if temp_path.exists():
                os.remove(temp_path)
            latency = time.time() - start_time
            _query_latencies.append(latency)
            _log_prediction(
                question=question,
                defect_label="unknown",
                severity="none",
                confidence=0.0,
                latency=latency,
                matches=retrieved_examples,
            )
            if tracker is not None:
                tracker.log_inference_run(
                    question=question,
                    predicted_defect="unknown",
                    predicted_severity="none",
                    confidence=0.0,
                    latency_sec=latency,
                    retrieved_matches=retrieved_examples,
                )
            return QueryResponse(
                query=question,
                answer="insufficient reference data",
                predicted_defect="unknown",
                predicted_severity="none",
                confidence=0.0,
                retrieved_matches=retrieved_examples,
            )

        # 4. Generate answer using VLM with retrieved context
        logger.info(f"Generating VLM response using {len(retrieved_examples)} retrieved images...")
        vlm_response = vlm_model.generate_answer(
            query_image_path=str(temp_path.resolve()),
            question=question,
            retrieved_examples=retrieved_examples,
        )

        # Clean up query file
        if temp_path.exists():
            os.remove(temp_path)

        latency = time.time() - start_time
        _query_latencies.append(latency)

        # Log prediction locally
        _log_prediction(
            question=question,
            defect_label=vlm_response["predicted_defect"],
            severity=vlm_response["predicted_severity"],
            confidence=vlm_response["confidence"],
            latency=latency,
            matches=retrieved_examples,
        )

        # Log prediction to MLflow
        if tracker is not None:
            tracker.log_inference_run(
                question=question,
                predicted_defect=vlm_response["predicted_defect"],
                predicted_severity=vlm_response["predicted_severity"],
                confidence=vlm_response["confidence"],
                latency_sec=latency,
                retrieved_matches=retrieved_examples,
            )

        return QueryResponse(
            query=question,
            answer=vlm_response["answer"],
            predicted_defect=vlm_response["predicted_defect"],
            predicted_severity=vlm_response["predicted_severity"],
            confidence=vlm_response["confidence"],
            retrieved_matches=retrieved_examples,
        )

    except Exception as e:
        if temp_path.exists():
            os.remove(temp_path)
        logger.error(f"Error in RAG pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing RAG pipeline: {e!s}")
