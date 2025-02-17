from pydantic_settings import BaseSettings
import torch


class RewardModelConfig(BaseSettings):
    model_name: str = "Skywork/Skywork-Reward-Gemma-2-27B-v0.2"
    temperature: float = 1.5
    compression_scale: float = 0.2
    tiktoken_model: str = "gpt-4o"
    reference_score: float = 0.5


class PromptGuardConfig(BaseSettings):
    model_name: str = "katanemo/Arch-Guard"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ScoringClientConfig(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 9102
    timeout: float = 32.0


class Config(BaseSettings):
    reward_model_config: RewardModelConfig = RewardModelConfig()
    prompt_guard_config: PromptGuardConfig = PromptGuardConfig()
    scoring_client_config: ScoringClientConfig = ScoringClientConfig()


CONFIG = Config()
