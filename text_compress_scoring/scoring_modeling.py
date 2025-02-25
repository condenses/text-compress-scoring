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

Score 1:
- Responses are completely off-topic or irrelevant
- Fails to understand or address the user's basic query
- Provides incorrect or misleading information
- May be harmful or counterproductive to user's needs
- Shows no evidence of understanding the context

Score 2:
- Responses are partially relevant but mostly miss the mark
- Addresses surface-level aspects while missing core needs
- Contains significant gaps or inaccuracies
- Requires substantial follow-up questions for clarity
- Shows limited understanding of user context

Score 3:
- Responses are generally on-topic and helpful
- Addresses main points but may miss some details
- Information is mostly accurate with minor gaps
- May need occasional clarification
- Demonstrates basic understanding of context
- Solutions are workable but not optimal

Score 4:
- Responses are well-aligned with user needs
- Addresses both main points and important details
- Information is accurate and well-structured
- Requires minimal clarification
- Shows good understanding of context
- Provides effective, practical solutions

Score 5:
- Responses perfectly match user needs and context
- Addresses all aspects comprehensively
- Information is completely accurate and thorough
- Requires no clarification or follow-up
- Demonstrates deep understanding of context
- Provides optimal, actionable solutions
- Anticipates potential issues or edge cases
""".strip()


class RelativeDataPoint(BaseModel):
    instruction: str
    response: str
    reference_answer: str
    rubric: str = HELPFULNESS_RUBRIC


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
        prompt = ABSOLUTE_REFINE_PROMPT.format(
            INSTRUCTION=data_point.instruction,
            RESPONSE=data_point.response,
            REFERENCE_ANSWER=data_point.reference_answer,
            RUBRIC=data_point.rubric,
        )
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=CONFIG.vllm_config.temperature,
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
