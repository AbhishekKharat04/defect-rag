"""Qdrant vector database sub-package."""

from src.vector_db.qdrant_client import QdrantDBClient

__all__: list[str] = ["QdrantDBClient"]
