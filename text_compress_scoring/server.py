from fastapi import FastAPI
import uvicorn
from typing import List
from loguru import logger
from openai import OpenAI

from .scoring_modeling import ScoringModel, GuardingModel, ScoringPrometheusModel
from .schemas import BatchScoringRequest, BatchScoringResponse
from .config import CONFIG

# Initialize models and clients
app = FastAPI()
if CONFIG.reward_model_config.enabled:
    scoring_model = ScoringModel()
if CONFIG.prompt_guard_config.enabled:
    guarding_model = GuardingModel()
if CONFIG.prometheus_model_config.enabled:
    scoring_prometheus_model = ScoringPrometheusModel()
openai_client = OpenAI()

# Constants
GENERATE_MODELS = ["meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-128K"]

logger.info("Initialized FastAPI server with scoring model and OpenAI client")


def generate_assistant_message(user_message: str, model: str) -> str:
    """Generate assistant response using OpenAI API."""
    logger.debug(f"Generating assistant message using model {model}")
    logger.debug(f"User message: {user_message[:100]}...")

    response = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_message}],
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
    scores = []

    if CONFIG.prometheus_model_config.enabled:
        p_scores = scoring_prometheus_model.score_batch(
            instruction, reference_answer, responses
        )
        logger.info(f"Prometheus scores: {p_scores}")
        scores.append(p_scores)

    if CONFIG.reward_model_config.enabled:
        r_scores = scoring_model.score_batch(instruction, reference_answer, responses)
        logger.info(f"Reward scores: {r_scores}")
        scores.append(r_scores)

    if not scores:
        raise ValueError("No scoring models are enabled")

    # Multiply scores from all enabled models together
    final_scores = [1.0] * len(responses)
    for score_list in scores:
        for i, score in enumerate(score_list):
            final_scores[i] *= score

    return final_scores


@app.post("/api/scoring", response_model=BatchScoringResponse)
def scoring(request: BatchScoringRequest) -> BatchScoringResponse:
    """Handle batch scoring requests."""
    logger.info(
        f"Received scoring request with {len(request.batch_compressed_user_messages)} messages to score"
    )

    scores = [0.0] * len(request.batch_compressed_user_messages)
    valid_indices = get_valid_messages(request.batch_compressed_user_messages)

    if not valid_indices:
        logger.warning("No valid messages to score, returning zero scores")
        return BatchScoringResponse(scores=scores)

    # Generate reference response
    model = GENERATE_MODELS[0]  # Simplified from random choice since only one model
    logger.info(f"Using model for generation: {model}")

    original_assistant_message = generate_assistant_message(
        request.original_user_message, model
    )

    # Generate responses for valid compressed messages
    responses = [
        generate_assistant_message(request.batch_compressed_user_messages[i], model)
        for i in valid_indices
    ]

    # Calculate final scores
    valid_scores = calculate_scores(
        request.original_user_message, original_assistant_message, responses
    )

    # Update scores for valid indices
    for idx, score in zip(valid_indices, valid_scores):
        scores[idx] = score

    logger.info(f"Completed scoring. Final scores: {scores}")
    return BatchScoringResponse(scores=scores)


def start_server():
    """Start the FastAPI server."""
    logger.info(
        f"Starting server on {CONFIG.scoring_client_config.host}:{CONFIG.scoring_client_config.port}"
    )
    uvicorn.run(
        app,
        host=CONFIG.scoring_client_config.host,
        port=CONFIG.scoring_client_config.port,
    )
