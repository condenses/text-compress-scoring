import pytest
from text_compress_scoring.scoring_modeling import (
    ScoringPrometheusModel,
    RelativeDataPoint,
)
from text_compress_scoring.config import CONFIG
from prometheus_eval.prompts import HELPFULNESS_RUBRIC


@pytest.fixture
def prometheus_model():
    return ScoringPrometheusModel()


@pytest.fixture
def sample_data():
    return {
        "instruction": "Explain how photosynthesis works.",
        "reference_answer": """Photosynthesis is the process by which plants convert light energy into chemical energy. 
        The process occurs in the chloroplasts, where chlorophyll captures sunlight. This light energy is used to 
        convert water and carbon dioxide into glucose and oxygen. The glucose serves as food for the plant, while 
        oxygen is released as a by-product.""",
        "responses": [
            """Photosynthesis is how plants make their food using sunlight. They take in CO2 and water, 
            and produce glucose and oxygen.""",  # Should get a medium score
            """Plants do something with sun.""",  # Should get a low score
        ],
    }


def test_score_absolute(prometheus_model, sample_data):
    data_point = RelativeDataPoint(
        instruction=sample_data["instruction"],
        response=sample_data["responses"][0],
        reference_answer=sample_data["reference_answer"],
    )

    score = prometheus_model.score_absolute(data_point)

    assert isinstance(score, int)
    assert 1 <= score <= 5


def test_score_batch(prometheus_model, sample_data):
    scores = prometheus_model.score_batch(
        instruction=sample_data["instruction"],
        reference_answer=sample_data["reference_answer"],
        responses=sample_data["responses"],
    )

    assert isinstance(scores, list)
    assert len(scores) == len(sample_data["responses"])
    assert all(isinstance(score, float) for score in scores)
    assert all(0 <= score <= 1 for score in scores)
    # First response should score higher than second response
    assert scores[0] > scores[1]
