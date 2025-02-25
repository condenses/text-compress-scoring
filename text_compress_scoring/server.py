from fastapi import FastAPI
import uvicorn
from typing import List
from loguru import logger
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR
from .utils import retry

from .scoring_modeling import (
    LLMPreferenceModel,
    SYSTEM_PROMPT,
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
    openai_client = NineteenAPI(
        base_url="https://api.nineteen.ai/v1",
        api_key="abc"
    )
else:
    openai_client = OpenAI(
        base_url=CONFIG.vllm_config.base_url, api_key=CONFIG.vllm_config.api_key
    )
preference_score = LLMPreferenceModel(openai_client)
guarding_model = GuardingModel()
logger.info("Initialized FastAPI server with scoring model and OpenAI client")

# Add CORS middleware if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add middleware to handle exceptions
class ExceptionLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as exc:
            logger.error(f"Unhandled exception occurred: {str(exc)}")
            return JSONResponse(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Internal Server Error"},
            )

app.add_middleware(ExceptionLoggingMiddleware)


@retry(max_retries=3, retry_delay=5)
def generate_assistant_message(user_message: str, model: str) -> str:
    """Generate assistant response using OpenAI API."""
    logger.debug(f"Generating assistant message using model {model}")
    logger.debug(f"User message: {user_message[:100]}...")

    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=CONFIG.vllm_config.temperature,
        top_p=CONFIG.vllm_config.top_p,
        max_completion_tokens=CONFIG.vllm_config.max_new_tokens,
    )
    assistant_message = response.choices[0].message.content
    logger.debug(f"Generated assistant message: {assistant_message[:100]}...")
    return assistant_message


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
    instruction: str, reference_answer: str, responses: List[str]
) -> List[float]:
    """Calculate combined scores using enabled scoring models."""
    scores = preference_score.score_batch(instruction, reference_answer, responses)
    logger.info(f"Preference scores: {scores}")
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

        # Generate reference response
        logger.info(f"Using model for generation: {CONFIG.vllm_config.model_name}")

        original_assistant_message = generate_assistant_message(
            request.original_user_message, CONFIG.vllm_config.model_name
        )

        # Generate responses for valid compressed messages
        responses = []
        for i in valid_indices:
            response = generate_assistant_message(
                request.batch_compressed_user_messages[i], CONFIG.vllm_config.model_name
            )
            responses.append(response)

        # Calculate final scores
        valid_scores = calculate_scores(
            request.original_user_message, original_assistant_message, responses
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
