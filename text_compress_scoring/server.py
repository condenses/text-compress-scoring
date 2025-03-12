from fastapi import FastAPI
import uvicorn
from typing import List
from loguru import logger
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware

from .scoring_modeling import (
    ParaphraseScorer,
    GuardingModel,
)
from .schemas import BatchScoringRequest, BatchScoringResponse
from .config import CONFIG
import bittensor as bt
import time


WALLET = bt.Wallet(
    name=CONFIG.wallet_name,
    path=CONFIG.wallet_path,
    hotkey=CONFIG.wallet_hotkey,
)


def get_signature_headers() -> dict:
    """
    Get the signature headers for the validator.
    """
    nonce = str(time.time_ns())
    signature = f"0x{WALLET.hotkey.sign(nonce).hex()}"
    return {
        "validator-hotkey": WALLET.hotkey.ss58_address,
        "signature": signature,
        "nonce": nonce,
        "netuid": "47",
        "Content-Type": "application/json",
    }


class NineteenAPI(OpenAI):
    @property
    def auth_headers(self) -> dict:
        return get_signature_headers()


# Initialize models and clients
app = FastAPI()
if CONFIG.use_nineteen_api:
    openai_client = NineteenAPI(base_url="https://api.nineteen.ai/v1", api_key="abc")
else:
    openai_client = OpenAI(
        base_url=CONFIG.vllm_config.base_url, api_key=CONFIG.vllm_config.api_key
    )
paraphrase_scorer = ParaphraseScorer(openai_client)
guarding_model = GuardingModel()
logger.info("Initialized FastAPI server with scoring model and OpenAI client")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_valid_messages(messages: List[str]) -> List[int]:
    """Filter messages through prompt guard and return valid indices."""
    valid_indices = []
    for i, message in enumerate(messages):
        logger.debug(f"Checking message {i} through prompt guard")
        if guarding_model.guard(message):
            logger.warning(f"Message {i} failed prompt guard check")
            break
        valid_indices.append(i)
        logger.debug(f"Message {i} passed prompt guard check")
    return valid_indices


def calculate_scores(
    original_message: str,
    compressed_messages: List[str],
) -> List[float]:
    """Calculate paraphrase scores between original and compressed messages."""
    scores = paraphrase_scorer.score_paraphrase_batch(
        original_message, compressed_messages
    )
    logger.info(f"Paraphrase scores: {scores}")
    return scores


@app.post("/api/scoring", response_model=BatchScoringResponse)
def scoring(request: BatchScoringRequest) -> BatchScoringResponse:
    """Handle batch scoring requests."""
    try:
        logger.info(
            f"Received scoring request with {len(request.batch_compressed_user_messages)} messages to score"
        )

        scores = [0.0] * len(request.batch_compressed_user_messages)
        valid_indices = get_valid_messages(request.batch_compressed_user_messages)

        if not valid_indices:
            logger.warning("No valid messages to score, returning zero scores")
            return BatchScoringResponse(scores=scores)

        # Get valid compressed messages
        valid_compressed_messages = [
            request.batch_compressed_user_messages[i] for i in valid_indices
        ]

        # Calculate paraphrase scores
        valid_scores = calculate_scores(
            request.original_user_message,
            valid_compressed_messages,
        )

        # Update scores for valid indices
        for idx, score in zip(valid_indices, valid_scores):
            scores[idx] = score

        logger.info(f"Completed scoring. Final scores: {scores}")
        return BatchScoringResponse(scores=scores)

    except Exception as e:
        logger.exception("Exception occurred during scoring")
        raise e


def start_server():
    """Start the FastAPI server."""
    logger.info(f"Starting server on {CONFIG.scoring_client_config.base_url}")
    uvicorn.run(
        app,
        host=CONFIG.scoring_client_config.host,
        port=CONFIG.scoring_client_config.port,
    )
