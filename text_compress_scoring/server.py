from fastapi import FastAPI
import uvicorn
from typing import List
from .scoring_modeling import ScoringModel, sigmoid
from .schemas import (
    ScoringResponse,
    BatchScoringRequest,
    BatchScoringResponse,
    Message,
)
from loguru import logger
from .config import CONFIG

app = FastAPI()
scoring_model = ScoringModel()


def score_single_messages(
    compressed_messages: List[Message], original_score: float
) -> ScoringResponse:
    """Score compressed and original messages"""
    try:
        # Check prompt guard for compressed messages
        for comp_msg in compressed_messages:
            if comp_msg.is_compressed:
                if scoring_model.guarding(comp_msg.content):
                    return ScoringResponse(score=0.0)
        compressed_score = scoring_model.score_messages(compressed_messages)
        compress_gain = sigmoid(compressed_score) / sigmoid(original_score)
        score = CONFIG.reward_model_config.reference_score * compress_gain
        return score
    except Exception as e:
        logger.error(f"Error scoring messages: {e}")
        return 0.0


@app.post("/api/scoring", response_model=BatchScoringResponse)
def scoring(request: BatchScoringRequest) -> BatchScoringResponse:
    scores = [0.0] * len(request.batch_compressed_messages)
    valid_indexes = []
    for i, single_compressed_messages in enumerate(request.batch_compressed_messages):
        for comp_msg in single_compressed_messages:
            if comp_msg.is_compressed and scoring_model.guarding(comp_msg.content):
                break
        valid_indexes.append(i)
    original_score = scoring_model.score_messages(request.original_messages)
    for i in valid_indexes:
        scores[i] = score_single_messages(
            request.batch_compressed_messages[i], original_score
        )
    return BatchScoringResponse(scores=scores)


def start_server():
    uvicorn.run(app, host=CONFIG.scoring_host, port=CONFIG.scoring_port)
