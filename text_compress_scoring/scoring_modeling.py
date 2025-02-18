from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
import numpy as np
from loguru import logger
from pydantic import BaseModel
from openai import OpenAI
import tiktoken
import re

from .config import CONFIG

# Corrected prompt name and content.
ABSOLUTE_REFINE_PROMPT = """You are an expert evaluator tasked with assessing the quality of a response based on a specific scoring rubric. Your goal is to provide concise feedback and assign a score that accurately reflects the response's quality.

Please review the following information:

1. Instruction to evaluate:
<instruction_to_evaluate>
{INSTRUCTION}
</instruction_to_evaluate>

2. Reference answer (This would receive a score of 5):
<reference_answer>
{REFERENCE_ANSWER}
</reference_answer>

3. Scoring rubric:
<score_rubric>
{RUBRIC}
</score_rubric>

4. Response to evaluate:
<response_to_evaluate>
{RESPONSE}
</response_to_evaluate>

Follow these steps to complete your evaluation:

1. Write concise feedback:
   Based on your analysis, provide clear and concise feedback that directly addresses each criterion in the rubric. Be objective and specific, using examples from the response to support your evaluation.

2. Assign a score:
   Determine an integer score between 1 and 5, where 5 is the highest quality (equivalent to the reference answer) and 1 is the lowest. Ensure your score accurately reflects your feedback and aligns with the rubric.

3. Format your output as follows:
   <feedback>
   (Your concise feedback here)
   </feedback>
   <score>(Integer score between 1 and 5)</score>

Important guidelines:
- Focus solely on the criteria outlined in the rubric.
- Be objective and consistent in your evaluation.
- Provide feedback that is clear and concise.
- Do not include any opening statements, closing remarks, or additional explanations outside of the specified format.
- Do not generate or evaluate any content not provided in the input materials.

Example output structure (do not use this content, it's just to illustrate the format):

<feedback>
The response demonstrates understanding of concepts A and B with clear explanations. However, it lacks discussion on crucial concept C. Writing is clear but could use more supporting examples. Minor error in concept B explanation slightly impacts quality.
</feedback>
<score>4</score>

Please proceed with your evaluation based on these instructions."""

HELPFULNESS_RUBRIC = """
[Does the model provide relevant and useful responses to the user's needs or questions?]
Score 1: The model's responses are irrelevant or unhelpful to the user's needs or queries.
Score 2: The model sometimes provides helpful information, but often fails to address the user's actual needs or questions.
Score 3: The model generally provides helpful responses that address the user's needs, though it may occasionally miss the mark.
Score 4: The model regularly provides helpful responses that are well-aligned with the user's inquiries, with only rare inaccuracies.
Score 5: The model consistently offers highly relevant and useful responses that perfectly cater to the user's needs and inquiries.
""".strip()


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
        self.llm_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
        )
        self.model = "mistralai/mistral-small-24b-instruct-2501"
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def single_absolute_grade(self, data_point: RelativeDataPoint) -> int:
        """
        Compute the absolute grade for a single data point using the Prometheus model.
        Returns an integer score between 1 and 5.
        """
        prompt = ABSOLUTE_REFINE_PROMPT.format(
            instruction=data_point.instruction,
            response=data_point.response,
            reference_answer=data_point.reference_answer,
            rubric=data_point.rubric,
        )
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=CONFIG.reward_model_config.temperature,
        )
        completion = response.choices[0].message.content
        match = re.search(
            r"<feedback>(.*?)</feedback>\s*<score>(\d+)</score>", completion, re.DOTALL
        )
        if not match:
            logger.warning(f"Could not parse completion: {completion}")
            return "No feedback provided", 1
        self.total_input_tokens += response.usage.prompt_tokens
        self.total_output_tokens += response.usage.completion_tokens
        logger.info(
            f"Prometheus scoring: {data_point.instruction[:32]}... | {data_point.response[:32]}... | {data_point.reference_answer[:32]}... | {response.usage.prompt_tokens} input tokens | {response.usage.completion_tokens} output tokens"
        )
        feedback, score = match.groups()
        return feedback.strip(), int(score)

    def score_absolute(self, data_point: RelativeDataPoint) -> int:
        """
        Compute the absolute grade for a single data point using the Prometheus model.
        Returns an integer score between 1 and 5.
        """
        feedback, score = self.single_absolute_grade(data_point)
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
