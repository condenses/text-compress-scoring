import numpy as np
from loguru import logger
from pydantic import BaseModel
from openai import OpenAI
import re
from transformers import pipeline
from .config import CONFIG
import torch
from .utils import retry

PARAPHRASE_SCORE_PROMPT = """
You are an impartial evaluator tasked with analyzing a paraphrase based on specific criteria. You will be provided with an original text and its paraphrase. Your goal is to evaluate the paraphrase and assign a score from 1 (worst) to 10 (best) based on the following criteria:

1. Meaning Retention (40% weight): Does the paraphrase preserve the core meaning and intent of the original?
2. Key Details (30% weight): Are all critical facts, entities, and nuances retained?
3. Wording & Structure (20% weight): Does it use distinct vocabulary/syntax while avoiding plagiarism?
4. Grammar & Fluency (10% weight): Is the paraphrase grammatically correct and natural-sounding?

Here is the original text:
<original_text>
{ORIGINAL_TEXT}
</original_text>

Here is the paraphrase:
<paraphrase>
{PARAPHRASE}
</paraphrase>

Carefully analyze the paraphrase in relation to the original text, considering each of the four criteria. Pay special attention to whether the core meaning is preserved, key details are retained, wording is sufficiently different, and the paraphrase is grammatically correct and fluent.

In your analysis, use the following scoring guidelines:
- 1-3: Major meaning distortion, missing key details, or nonsensical.
- 4-6: Partial meaning preserved but with omissions/errors.
- 7-9: Minor issues (e.g., awkward phrasing, slight inaccuracies).
- 10: Flawless: Equivalent meaning, no redundancy, and natural language.

First, provide a concise explanation of your evaluation, highlighting strengths and weaknesses according to each criterion. Then, based on your analysis, assign an overall score from 1 to 10.

Present your response in the following format:
<explaination>
[Your concise analysis here, addressing each criterion]
</explainataion>

<score>
[Your numerical score from 1 to 10]
</score>

Ensure that your explanation is thorough yet concise, and that your score accurately reflects your analysis based on the given criteria and scoring guidelines."""


class ParaphraseScorer:
    def __init__(self, openai_client: OpenAI):
        self.llm_client = openai_client
        self.model = CONFIG.vllm_config.model_name
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    @retry(max_retries=3, retry_delay=5)
    def single_paraphrase_grade(
        self, original_text: str, paraphrase: str
    ) -> tuple[str, int]:
        """
        Compute the paraphrase score between original text and its paraphrase.
        Returns a tuple of (feedback, score) where score is between 1 and 10.
        """
        prompt = PARAPHRASE_SCORE_PROMPT.format(
            ORIGINAL_TEXT=original_text,
            PARAPHRASE=paraphrase,
        )
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=CONFIG.vllm_config.temperature,
        )
        completion = response.choices[0].message.content

        # Fix regex to handle both spellings of explanation
        score_match = re.search(r"<score>\s*(\d+)\s*</score>", completion, re.DOTALL)
        explanation_match = re.search(
            r"<(explaination|explanation)>(.*?)</(explaination|explanation)>",
            completion,
            re.DOTALL,
        )

        if not score_match:
            logger.warning(f"Could not parse completion: {completion}")
            return "No feedback provided", 1

        self.total_input_tokens += response.usage.prompt_tokens
        self.total_output_tokens += response.usage.completion_tokens
        logger.info(
            f"Paraphrase scoring: Original: {original_text[:32]}... | Paraphrase: {paraphrase[:32]}... | {response.usage.prompt_tokens} input tokens | {response.usage.completion_tokens} output tokens"
        )

        score = score_match.group(1).strip()
        # Extract explanation if available
        explanation = (
            explanation_match.group(2).strip()
            if explanation_match
            else "No explanation provided"
        )

        return explanation, int(score)

    def score_paraphrase_batch(
        self, original_message: str, compressed_messages: list[str]
    ) -> list[float]:
        """
        Compute paraphrase scores for a batch of original texts and their paraphrases.
        Returns normalized scores between 0 and 1.
        """
        scores = []
        for compressed_message in compressed_messages:
            feedback, score = self.single_paraphrase_grade(
                original_message, compressed_message
            )
            logger.info(f"Paraphrase feedback: {feedback} | score: {score}")
            scores.append(score)

        # Normalize scores from 1-10 to 0-1
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
