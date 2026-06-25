import pytest

from src.vlm.qwen_generator import QwenVLGenerator


@pytest.fixture
def vlm_generator():
    """Fixture to initialize QwenVLGenerator in mock mode."""
    return QwenVLGenerator(mode="mock")

def test_mock_generation_conforming(vlm_generator):
    """Tests that a conforming (good) bottle query image generates correct mock metadata."""
    query_image_path = "/tmp/data/bottle/train/good/005.png"
    question = "Is this bottle defective?"

    retrieved_examples = [
        {
            "id": "1",
            "score": 0.98,
            "payload": {"image_path": "/tmp/data/bottle/train/good/001.png", "defect_label": "good", "severity": "none"}
        },
        {
            "id": "2",
            "score": 0.92,
            "payload": {"image_path": "/tmp/data/bottle/train/good/002.png", "defect_label": "good", "severity": "none"}
        }
    ]

    res = vlm_generator.generate_answer(
        query_image_path=query_image_path,
        question=question,
        retrieved_examples=retrieved_examples
    )

    assert "answer" in res
    assert res["predicted_defect"] == "no defect detected"
    assert res["predicted_severity"] == "none"
    assert res["confidence"] > 0.90
    assert len(res["retrieved_sources"]) == 2

def test_mock_generation_defect(vlm_generator):
    """Tests that a broken_small defect query image generates correct mock metadata."""
    query_image_path = "/tmp/data/bottle/test/broken_small/002.png"
    question = "Does this bottle have a small break? Rate its severity."

    retrieved_examples = [
        {
            "id": "3",
            "score": 0.89,
            "payload": {"image_path": "/tmp/data/bottle/test/broken_small/001.png", "defect_label": "broken_small", "severity": "medium"}
        }
    ]

    res = vlm_generator.generate_answer(
        query_image_path=query_image_path,
        question=question,
        retrieved_examples=retrieved_examples
    )

    assert "answer" in res
    assert res["predicted_defect"] == "broken_small"
    assert res["predicted_severity"] == "medium"
    assert res["confidence"] == 0.85
    assert res["retrieved_sources"] == ["/tmp/data/bottle/test/broken_small/001.png"]

def test_structured_response_parsing(vlm_generator):
    """Tests that the custom parser successfully extracts metadata tags from Qwen-VL response text."""
    vlm_output_text = (
        "Based on inspection, the part has a scratch.\n\n"
        "```\n"
        "DEFECT_LABEL: scratch\n"
        "SEVERITY: high\n"
        "CONFIDENCE: 0.88\n"
        "```"
    )

    retrieved_examples = [
        {"payload": {"image_path": "path1"}}
    ]

    parsed = vlm_generator._parse_structured_response(vlm_output_text, retrieved_examples)

    assert parsed["predicted_defect"] == "scratch"
    assert parsed["predicted_severity"] == "high"
    assert parsed["confidence"] == 0.88
    assert parsed["retrieved_sources"] == ["path1"]
