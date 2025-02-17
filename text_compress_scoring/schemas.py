from pydantic import BaseModel
from typing import List


class Message(BaseModel):
    role: str
    content: str
    is_compressed: bool = False


class ScoringResponse(BaseModel):
    score: float


class BatchScoringRequest(BaseModel):
    batch_compressed_messages: List[List[Message]]
    original_messages: List[Message]


class BatchScoringResponse(BaseModel):
    scores: List[float]
