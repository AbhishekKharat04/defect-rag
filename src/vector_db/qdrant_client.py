"""Qdrant vector database client wrapper.

Provides a high-level API over the ``qdrant-client`` SDK for:
- Collection lifecycle (create / delete / info)
- Payload-aware vector upsert and cosine-similarity search
- In-memory, local-disk, and remote-server connection modes
- Async counterparts for all read operations (via ``asyncio.to_thread``)
"""

import asyncio
import logging
import uuid
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from src.config import settings

logger = logging.getLogger(__name__)


class QdrantDBClient:
    """High-level wrapper around the Qdrant vector database SDK.

    Supports three connection modes selected at init time:

    1. **In-memory** — ``host=":memory:"`` or ``location=":memory:"``
       (ideal for unit tests).
    2. **Local disk** — ``path="/some/dir"`` (serverless fallback if
       Docker is unavailable).
    3. **Remote server** — ``host``/``port`` targeting a running Qdrant
       container or cluster.

    If a remote connection fails, the client automatically falls back
    to local-disk storage under ``settings.DATA_DIR / "qdrant_storage"``.

    Attributes:
        client: Underlying ``qdrant_client.QdrantClient`` instance.
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        location: str | None = None,
        path: str | None = None,
    ) -> None:
        """Initialise connection to Qdrant.

        Args:
            host: Server hostname. Use ``":memory:"`` for the in-memory
                backend.
            port: Server HTTP port (default ``6333``).
            location: Explicit Qdrant location string (e.g.
                ``":memory:"``).
            path: Local filesystem directory for serverless storage.
        """
        self.host: str = host or settings.QDRANT_HOST
        self.port: int = port or settings.QDRANT_PORT
        self.location: str | None = location
        self.path: str | None = path

        if self.location == ":memory:" or self.host == ":memory:":
            logger.info("Initializing in-memory Qdrant Client...")
            self.client = QdrantClient(location=":memory:")
        elif self.path:
            logger.info(f"Initializing local disk Qdrant Client at path: {self.path}")
            self.client = QdrantClient(path=self.path)
        else:
            logger.info(f"Connecting to Qdrant server at {self.host}:{self.port}...")
            try:
                self.client = QdrantClient(host=self.host, port=self.port, timeout=3.0)
                self.client.get_collections()
                logger.info("Successfully connected to Qdrant server.")
            except Exception as e:
                fallback_path = str(settings.DATA_DIR / "qdrant_storage")
                logger.warning(
                    f"Failed to connect to Qdrant server at {self.host}:{self.port}: {e}. "
                    f"Falling back to serverless local disk storage at: {fallback_path}"
                )
                self.client = QdrantClient(path=fallback_path)

    # ------------------------------------------------------------------
    # Collection lifecycle
    # ------------------------------------------------------------------

    def create_collection(
        self,
        collection_name: str | None = None,
        vector_size: int = 512,
        recreate: bool = False,
    ) -> bool:
        """Create a Qdrant collection if it does not already exist.

        Args:
            collection_name: Target collection. Falls back to
                ``settings.QDRANT_COLLECTION``.
            vector_size: Embedding dimensionality (``512`` for CLIP
                ViT-B/32).
            recreate: When *True*, drops and re-creates the collection.

        Returns:
            *True* if the collection exists after the call, *False* on
            error.
        """
        name = collection_name or settings.QDRANT_COLLECTION

        try:
            exists = self.client.collection_exists(collection_name=name)

            if exists and recreate:
                logger.info(f"Recreating collection '{name}'...")
                self.client.delete_collection(collection_name=name)
                exists = False

            if not exists:
                logger.info(f"Creating collection '{name}' with vector size {vector_size}...")
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    ),
                )
                logger.info(f"Collection '{name}' created successfully.")
            else:
                logger.info(f"Collection '{name}' already exists.")
            return True
        except Exception as e:
            logger.error(f"Error creating collection '{name}': {e}")
            return False

    def get_collection_info(self, collection_name: str | None = None) -> dict[str, Any] | None:
        """Return count and status metadata for a collection.

        Args:
            collection_name: Target collection.

        Returns:
            Dictionary with keys ``status``, ``points_count``,
            ``vectors_count``; or *None* if the collection does not
            exist.
        """
        name = collection_name or settings.QDRANT_COLLECTION
        try:
            info = self.client.get_collection(collection_name=name)
            return {
                "status": info.status.value if hasattr(info.status, "value") else str(info.status),
                "points_count": info.points_count,
                "vectors_count": getattr(info, "vectors_count", info.points_count),
            }
        except UnexpectedResponse as e:
            if "not found" in str(e).lower():
                return None
            raise
        except Exception as e:
            logger.error(f"Error getting info for collection '{name}': {e}")
            return None

    def delete_collection(self, collection_name: str | None = None) -> bool:
        """Delete a Qdrant collection.

        Args:
            collection_name: Name of the collection to delete.

        Returns:
            *True* if deletion succeeded, *False* on error.
        """
        name = collection_name or settings.QDRANT_COLLECTION
        try:
            self.client.delete_collection(collection_name=name)
            logger.info(f"Deleted collection '{name}'.")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection '{name}': {e}")
            return False

    # ------------------------------------------------------------------
    # Data operations
    # ------------------------------------------------------------------

    def upsert_images(
        self,
        embeddings: np.ndarray,
        metadata_list: list[dict[str, Any]],
        ids: list[str | int] | None = None,
        collection_name: str | None = None,
    ) -> bool:
        """Upsert image embeddings with payload metadata into Qdrant.

        Args:
            embeddings: Array of shape ``(N, vector_dim)``.
            metadata_list: List of *N* metadata dictionaries (image path,
                defect label, severity, etc.).
            ids: Optional list of point IDs. Random UUIDs are generated
                when *None*.
            collection_name: Target collection.

        Returns:
            *True* if the upsert succeeded, *False* otherwise.

        Raises:
            ValueError: If ``embeddings`` and ``metadata_list`` (or
                ``ids``) have mismatched lengths.
        """
        name = collection_name or settings.QDRANT_COLLECTION

        if len(embeddings) != len(metadata_list):
            raise ValueError("Embeddings and metadata lists must have the same length.")

        if ids is not None and len(ids) != len(embeddings):
            raise ValueError("IDs and embeddings lists must have the same length.")

        points: list[models.PointStruct] = []
        for i in range(len(embeddings)):
            point_id = ids[i] if ids is not None else str(uuid.uuid4())
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embeddings[i].tolist(),
                    payload=metadata_list[i],
                )
            )

        try:
            logger.info(f"Upserting {len(points)} vectors into collection '{name}'...")
            self.client.upsert(collection_name=name, points=points, wait=True)
            logger.info("Upsert completed successfully.")
            return True
        except Exception as e:
            logger.error(f"Error upserting vectors into collection '{name}': {e}")
            return False

    def search_similar(
        self,
        query_vector: np.ndarray,
        top_k: int | None = None,
        collection_name: str | None = None,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for the *top_k* most similar vectors by cosine distance.

        Args:
            query_vector: 1-D embedding array for the query image.
            top_k: Number of nearest neighbours to return.
            collection_name: Target collection.
            filter_dict: Optional payload-field equality filters
                (e.g. ``{"defect_label": "scratch"}``).

        Returns:
            List of result dicts, each with keys ``id``, ``score``, and
            ``payload``.
        """
        name = collection_name or settings.QDRANT_COLLECTION
        k = top_k or settings.TOP_K

        query_list = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector

        # Construct payload filters if provided
        query_filter: models.Filter | None = None
        if filter_dict:
            filter_conditions = [
                models.FieldCondition(key=key, match=models.MatchValue(value=val))
                for key, val in filter_dict.items()
            ]
            query_filter = models.Filter(must=filter_conditions)

        try:
            search_results = self.client.query_points(
                collection_name=name,
                query=query_list,
                limit=k,
                query_filter=query_filter,
                with_payload=True,
            ).points

            return [
                {"id": res.id, "score": res.score, "payload": res.payload}
                for res in search_results
            ]
        except Exception as e:
            logger.error(f"Error searching collection '{name}': {e}")
            return []

    # ------------------------------------------------------------------
    # Async wrappers (for FastAPI event-loop compatibility)
    # ------------------------------------------------------------------

    async def search_similar_async(
        self,
        query_vector: np.ndarray,
        top_k: int | None = None,
        collection_name: str | None = None,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Non-blocking version of :meth:`search_similar`.

        Offloads the synchronous Qdrant SDK call to a worker thread.
        """
        return await asyncio.to_thread(
            self.search_similar, query_vector, top_k, collection_name, filter_dict
        )

    async def get_collection_info_async(self, collection_name: str | None = None) -> dict[str, Any] | None:
        """Non-blocking version of :meth:`get_collection_info`."""
        return await asyncio.to_thread(self.get_collection_info, collection_name)

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def is_healthy(self) -> bool:
        """Check whether the Qdrant backend is reachable and responsive.

        Returns:
            *True* if the server responds to a ``get_collections`` call,
            *False* otherwise.
        """
        try:
            await asyncio.to_thread(self.client.get_collections)
            return True
        except Exception:
            return False
