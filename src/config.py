"""Application-wide configuration and structured logging setup.

This module handles:
- Redirecting HuggingFace / PyTorch caches to the workspace directory.
- Loading configuration from environment variables via Pydantic Settings.
- Configuring structured JSON logging with request-ID propagation.
"""

import logging
import logging.config
import os
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Cache redirection (must run before any HF / torch import)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(CACHE_DIR / "huggingface")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TORCH_HOME"] = str(CACHE_DIR / "torch")
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

# Add local packages directory to PYTHONPATH
PACKAGES_DIR = BASE_DIR / "packages"
if PACKAGES_DIR.exists() and str(PACKAGES_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGES_DIR))

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
from pydantic_settings import BaseSettings, SettingsConfigDict
import torch

#: Semantic version of the application, surfaced in /health and logs.
VERSION: str = "1.0.0"


class Settings(BaseSettings):
    """Global application settings loaded from environment variables and ``.env`` file.

    Attributes:
        QDRANT_HOST: Hostname of the Qdrant vector database server.
        QDRANT_PORT: HTTP port of the Qdrant server.
        QDRANT_COLLECTION: Default Qdrant collection name for defect images.
        CLIP_MODEL_NAME: HuggingFace identifier for the CLIP embedding model.
        QWEN_MODEL_NAME: HuggingFace identifier for the Qwen2.5-VL model.
        DEVICE: Compute device, auto-detected from CUDA availability.
        BASE_DIR: Project root directory (auto-resolved).
        DATA_DIR: Directory for datasets, uploads, and generated data.
        TOP_K: Default number of nearest neighbours to retrieve from Qdrant.
        BACKEND_URL: Base URL of the FastAPI backend (used by the Gradio frontend).
        LOG_LEVEL: Logging verbosity level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """

    # Vector DB settings
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "defect_collection"

    # Embedding settings
    CLIP_MODEL_NAME: str = "openai/clip-vit-base-patch32"

    # VLM settings
    QWEN_MODEL_NAME: str = "Qwen/Qwen2.5-VL-3B-Instruct"

    # Device configuration
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Directory Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"

    # RAG parameters
    TOP_K: int = 3

    # API Backend URL
    BACKEND_URL: str = "http://localhost:8000"

    # Logging
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.getcwd(), ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

# Ensure directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
(settings.BASE_DIR / "logs").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------
class _JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects.

    Each record includes *timestamp*, *level*, *logger*, *message*, and an
    optional *request_id* propagated via :class:`logging.LogRecord` extras.
    """

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        import json
        import datetime

        log_entry = {
            "timestamp": datetime.datetime.fromtimestamp(
                record.created, tz=datetime.timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Attach request-ID when set by the middleware
        request_id: Optional[str] = getattr(record, "request_id", None)
        if request_id:
            log_entry["request_id"] = request_id

        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging(level: Optional[str] = None) -> None:
    """Configure the root logger with structured JSON output.

    Args:
        level: Override log level. Falls back to ``settings.LOG_LEVEL`` if *None*.
    """
    resolved_level = (level or settings.LOG_LEVEL).upper()

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": _JSONFormatter,
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "json",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {
            "handlers": ["console"],
            "level": resolved_level,
        },
        # Silence noisy third-party loggers
        "loggers": {
            "uvicorn": {"level": "WARNING"},
            "uvicorn.access": {"level": "WARNING"},
            "httpx": {"level": "WARNING"},
            "httpcore": {"level": "WARNING"},
        },
    }
    logging.config.dictConfig(config)
