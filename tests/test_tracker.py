import mlflow
import pytest

from src.utils.tracker import MLflowTracker


@pytest.fixture(scope="module")
def tracker():
    """Fixture to initialize the MLflowTracker."""
    # Log to a separate test experiment to keep production logs clean
    return MLflowTracker(experiment_name="Test_Defect_Tracker")

def test_log_inference_run(tracker):
    """Tests logging a single prediction event, asserting params and metrics are saved."""
    question = "Is there a scratch on the bottle neck?"
    predicted_defect = "scratch"
    predicted_severity = "high"
    confidence = 0.88
    latency_sec = 0.45

    retrieved_matches = [
        {"score": 0.94, "payload": {"image_path": "ref1.png", "defect_label": "scratch", "severity": "medium"}},
        {"score": 0.82, "payload": {"image_path": "ref2.png", "defect_label": "scratch", "severity": "high"}}
    ]

    # Run tracker log
    run_id = tracker.log_inference_run(
        question=question,
        predicted_defect=predicted_defect,
        predicted_severity=predicted_severity,
        confidence=confidence,
        latency_sec=latency_sec,
        retrieved_matches=retrieved_matches,
        run_name="test-run-event"
    )

    assert run_id is not None
    assert isinstance(run_id, str)

    # Retrieve run from MLflow using the SDK to verify content
    run = mlflow.get_run(run_id)

    # Assert parameters
    assert run.data.params["top_k"] == "3"
    assert run.data.params["question"] == question

    # Assert metrics
    assert float(run.data.metrics["confidence"]) == pytest.approx(confidence)
    assert float(run.data.metrics["latency_sec"]) == pytest.approx(latency_sec)
    assert float(run.data.metrics["max_similarity_score"]) == pytest.approx(0.94)
    assert float(run.data.metrics["avg_similarity_score"]) == pytest.approx(0.88)

    # Assert tags
    assert run.data.tags["predicted_defect"] == predicted_defect
    assert run.data.tags["predicted_severity"] == predicted_severity
    assert run.data.tags["retrieved_count"] == "2"
