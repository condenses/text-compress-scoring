from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
import numpy as np
from loguru import logger
from pydantic import BaseModel

from .config import CONFIG
from prometheus_eval.litellm import LiteLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import HELPFULNESS_RUBRIC

# Corrected prompt name and content.
ABSOLUTE_REFINE_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing an evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "(write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.
5. Don't get confused. You are conducting an absolute grading of another model's grading! For convenience, I will seperate the input and output of the other model's relative grading with "@@@"s.

@@@
###The instruction to evaluate:
{instruction}
@@@

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
{rubric}

###Feedback: """


def sigmoid(
    x: float, temperature: float = CONFIG.reward_model_config.temperature
) -> float:
    """Apply sigmoid function with temperature scaling."""
    return 1 / (1 + np.exp(-x / temperature))


class RelativeDataPoint(BaseModel):
    instruction: str
    response: str
    reference_answer: str
    rubric: str = HELPFULNESS_RUBRIC


class ScoringPrometheusModel:
    def __init__(self):
        self.model = LiteLLM(
            name="together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
            api_base="https://api.together.xyz/v1",
        )
        self.prometheus_judge = PrometheusEval(
            model=self.model,
            absolute_grade_template=ABSOLUTE_REFINE_PROMPT,
        )

    def score_absolute(self, data_point: RelativeDataPoint) -> int:
        """
        Compute the absolute grade for a single data point using the Prometheus model.
        Returns an integer score between 1 and 5.
        """
        feedback, score = self.prometheus_judge.single_absolute_grade(
            **data_point.model_dump()
        )
        logger.info(f"Prometheus feedback: {feedback} | score: {score}")
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
        normalized_scores = [score / 5.0 for score in scores]
        return normalized_scores


class GuardingModel:
    def __init__(self):
        self.prompt_guard = pipeline(
            "text-classification",
            model=CONFIG.prompt_guard_config.model_name,
            device=CONFIG.prompt_guard_config.device,  # Use 'device' if supported
        )

    def guard(self, prompt: str) -> bool:
        """
        Evaluate a prompt with the text-classification pipeline.
        Returns True if the prompt is classified as a "JAILBREAK".
        """
        result = self.prompt_guard(prompt)
        logger.info(f"Prompt guard result: {result} | prompt: {prompt[:32]}...")
        return result[0]["label"] == "JAILBREAK"


class ScoringModel:
    def __init__(self):
        model_name = CONFIG.reward_model_config.model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
            num_labels=1,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def _score_message_pair(self, instruction: str, message: str) -> float:
        """
        Helper method: Score a single (instruction, message) pair.
        Returns the raw logit score.
        """
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": message},
        ]
        tokenized = self.tokenizer.apply_chat_template(
            messages, tokenize=True, return_tensors="pt"
        )
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        logits = self.model(**tokenized).logits
        return logits[0][0].item()

    @torch.no_grad()
    def score_batch(
        self, instruction: str, reference_answer: str, responses: list[str]
    ) -> list[float]:
        """
        Score a batch of responses relative to the reference answer.
        The score is normalized using a sigmoid transformation.
        """
        # Compute the reference score once.
        ref_score = self._score_message_pair(instruction, reference_answer)

        # Prepare a batch of message pairs for all responses.
        batch_messages = [
            [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response},
            ]
            for response in responses
        ]

        # Batch tokenization: assumes the tokenizer supports batching.
        tokenized_batch = self.tokenizer.apply_chat_template(
            batch_messages,
            tokenize=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        tokenized_batch = {k: v.to(self.device) for k, v in tokenized_batch.items()}
        logits = self.model(**tokenized_batch).logits
        # Squeeze the logits to shape (batch_size,)
        response_scores = logits.squeeze(dim=1).tolist()

        normalized_scores = [
            sigmoid(score) / sigmoid(ref_score) for score in response_scores
        ]
        return normalized_scores
