import numpy as np
from loguru import logger
from pydantic import BaseModel
from openai import OpenAI
import re
from transformers import pipeline
from .config import CONFIG
import torch
from .utils import retry

SYSTEM_PROMPT = """You are direct and efficient. Follow these rules:

Core Rules:
- Answer immediately with key information
- Skip all pleasantries and context
- Use simple words and short sentences
- Never elaborate unless asked
- Don't ask follow-up questions
- Don't explain your process
- Don't offer alternatives
- Don't make suggestions

Format:
- One line answers when possible
- No greetings or signoffs
- Skip examples
- Code only without explanation
- Use active voice only

If confused, ask only what's needed to answer. Nothing more."""

SIMPLE_COMPARISON_PROMPT = """You are comparing a submitted answer to an expert answer on a given question. Here is the data:
[BEGIN DATA]
************
[Question]: {instruction}
************
[Expert]: {reference_answer}
************
[Submission]: {response}
************
[END DATA]

Compare the factual content of the submitted answer with the expert answer. Ignore any differences in style, grammar, or punctuation.
Rate the submission on a scale of 1 to 10.
Put the score in <score> tags.
Example:
<score>10</score>
"""


class RelativeDataPoint(BaseModel):
    instruction: str
    response: str
    reference_answer: str


class LLMPreferenceModel:
    def __init__(self, openai_client: OpenAI):
        self.llm_client = openai_client
        self.model = CONFIG.vllm_config.model_name
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    @retry(max_retries=3, retry_delay=5)
    def single_absolute_grade(self, data_point: RelativeDataPoint) -> int:
        """
        Compute the absolute grade for a single data point using the vLLM model.
        Returns an integer score between 1 and 5.
        """
        prompt = SIMPLE_COMPARISON_PROMPT.format(
            instruction=data_point.instruction,
            response=data_point.response,
            reference_answer=data_point.reference_answer,
        )
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=CONFIG.vllm_config.temperature,
            top_p=0.95,
        )
        completion = response.choices[0].message.content
        match = re.search(r"<score>(\d+)</score>", completion, re.DOTALL)
        if not match:
            logger.warning(f"Could not parse completion: {completion}")
            return "No score provided", 1
        self.total_input_tokens += response.usage.prompt_tokens
        self.total_output_tokens += response.usage.completion_tokens
        logger.info(
            f"Prometheus scoring: {data_point.instruction[:32]}... | {data_point.response[:32]}... | {data_point.reference_answer[:32]}... | {response.usage.prompt_tokens} input tokens | {response.usage.completion_tokens} output tokens"
        )
        score = match.groups()[0]
        return int(score)

    def score_absolute(self, data_point: RelativeDataPoint) -> int:
        """
        Compute the absolute grade for a single data point using the Prometheus model.
        Returns an integer score between 1 and 5.
        """
        score = self.single_absolute_grade(data_point)
        logger.info(f"Prometheus score: {score}")
        return score

    def score_batch(
        self, instruction: str, reference_answer: str, responses: list[str]
    ) -> list[float]:
        """
        Compute absolute scores for a batch of responses and normalize them by dividing by 5.
        """
        scores = [
            self.score_absolute(
                RelativeDataPoint(
                    instruction=instruction,
                    response=response,
                    reference_answer=reference_answer,
                )
            )
            for response in responses
        ]
        normalized_scores = [score / 10.0 for score in scores]
        return normalized_scores


class GuardingModel:
    def __init__(self):
        self.prompt_guard = pipeline(
            "text-classification",
            model=CONFIG.prompt_guard_config.model_name,
            device=CONFIG.prompt_guard_config.device,  # Use 'device' if supported
        )

    @torch.no_grad()
    def guard(self, prompt: str) -> bool:
        """
        Evaluate a prompt with the text-classification pipeline.
        Returns True if the prompt is classified as a "JAILBREAK".
        """
        try:
            result = self.prompt_guard(prompt)
            logger.info(f"Prompt guard result: {result} | prompt: {prompt[:32]}...")
            return result[0]["label"] == "JAILBREAK"
        except Exception as e:
            logger.error(f"Error in prompt guard: {e}")
            return True
