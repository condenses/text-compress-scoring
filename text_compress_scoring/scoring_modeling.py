from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from loguru import logger
import numpy as np
from transformers import pipeline
from .config import CONFIG


def sigmoid(x: float) -> float:
    """Apply sigmoid function with temperature scaling."""
    return 1 / (1 + np.exp(-x / CONFIG.reward_model_config.temperature))


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

        self.prompt_guard = pipeline(
            "text-classification",
            model=CONFIG.prompt_guard_config.model_name,
            device_map=CONFIG.prompt_guard_config.device,
        )

    @torch.no_grad()
    def guarding(self, prompt: str) -> bool:
        """
        Guard a prompt from being scored.
        Return True if the prompt is jailbroken.
        """
        result = self.prompt_guard(prompt)
        logger.info(f"Prompt guard result: {result} | prompt: {prompt[:32]}")
        return result[0]["label"] == "JAILBREAK"

    @torch.no_grad()
    def score_messages(self, messages) -> float:
        """Score a set of messages using the reward model."""
        tokenized = self.tokenizer.apply_chat_template(
            messages, tokenize=True, return_tensors="pt"
        ).to(self.device)
        return self.model(tokenized).logits[0][0].item()
