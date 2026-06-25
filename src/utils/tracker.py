"""MLflow experiment tracking for RAG inference predictions.

Logs per-query parameters (model identifiers, question text, top-k) and
metrics (latency, confidence, similarity scores) to a local MLflow
file-based tracking store.
"""

import logging
from typing import Any

import mlflow

from src.config import settings

logger = logging.getLogger(__name__)


class MLflowTracker:
    """Helper class for tracking RAG inference experiments with MLflow.

    Each :meth:`log_inference_run` call creates a nested MLflow run with
    parameters, metrics, and tags that can be compared in the MLflow UI
    (``mlflow ui --port 5000``).

    Attributes:
        tracking_uri: Local file URI pointing to the ``mlruns/`` directory.
        experiment: The active MLflow experiment object.
    """

    def __init__(self, experiment_name: str = "Defect_RAG_Assistant") -> None:
        """Initialise the MLflow tracker and set the local tracking URI.

        Args:
            experiment_name: Name of the MLflow experiment to create or
                reuse.
        """
        self.tracking_uri: str = (settings.BASE_DIR / "mlruns").as_uri()
        mlflow.set_tracking_uri(self.tracking_uri)

        logger.info(f"MLflow local tracking URI set to: {self.tracking_uri}")

        try:
            self.experiment = mlflow.set_experiment(experiment_name)
            logger.info(
                f"MLflow experiment configured: '{experiment_name}' "
                f"(ID: {self.experiment.experiment_id})"
            )
        except Exception as e:
            logger.error(f"Failed to set MLflow experiment: {e}")

    def log_inference_run(
        self,
        question: str,
        predicted_defect: str,
        predicted_severity: str,
        confidence: float,
        latency_sec: float,
        retrieved_matches: list[dict[str, Any]],
        run_name: str | None = None,
    ) -> str | None:
        """Log a single inference prediction as an MLflow run.

        Args:
            question: The user's text query question.
            predicted_defect: VLM-predicted defect class label.
            predicted_severity: VLM-predicted severity level.
            confidence: VLM confidence score (0.0–1.0).
            latency_sec: End-to-end pipeline latency in seconds.
            retrieved_matches: List of Qdrant search result dicts.
            run_name: Optional human-readable run name.

        Returns:
            The MLflow run ID if logging succeeded, *None* otherwise.
        """
        try:
            scores = [m["score"] for m in retrieved_matches] if retrieved_matches else []
            max_score = max(scores) if scores else 0.0
            avg_score = sum(scores) / len(scores) if scores else 0.0

            with mlflow.start_run(run_name=run_name, nested=True) as run:
                mlflow.log_params({
                    "clip_model": settings.CLIP_MODEL_NAME,
                    "vlm_model": settings.QWEN_MODEL_NAME,
                    "device": settings.DEVICE,
                    "top_k": settings.TOP_K,
                    "question": question,
                })

                mlflow.log_metrics({
                    "latency_sec": latency_sec,
                    "confidence": confidence,
                    "max_similarity_score": max_score,
                    "avg_similarity_score": avg_score,
                })

                mlflow.set_tags({
                    "predicted_defect": predicted_defect,
                    "predicted_severity": predicted_severity,
                    "retrieved_count": len(retrieved_matches),
                })

                logger.info(f"Logged inference details to MLflow run: {run.info.run_id}")
                return run.info.run_id
        except Exception as e:
            logger.error(f"Error logging to MLflow: {e}")
            return None
