from pydantic import BaseModel
from typing import List


class ScoringResponse(BaseModel):
    score: float


class BatchScoringRequest(BaseModel):
    batch_compressed_user_messages: List[str]
    original_user_message: str


class BatchScoringResponse(BaseModel):
    scores: List[float]
