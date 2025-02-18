from pydantic_settings import BaseSettings
import torch


class RewardModelConfig(BaseSettings):
    enabled: bool = False
    model_name: str = "Skywork/Skywork-Reward-Gemma-2-27B-v0.2"
    temperature: float = 0.01
    compression_scale: float = 0.2
    tiktoken_model: str = "gpt-4o"
    reference_score: float = 0.5


class PromptGuardConfig(BaseSettings):
    enabled: bool = True
    model_name: str = "katanemo/Arch-Guard"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ScoringClientConfig(BaseSettings):
    host: str = "localhost"
    port: int = 9102
    timeout: float = 64.0


class PrometheusModelConfig(BaseSettings):
    enabled: bool = True
    model_name: str = "mistralai/Mistral-Small-24B-Instruct-2501"
    temperature: float = 0.01


class GenerateModelConfig(BaseSettings):
    model_name: str = "mistralai/Mistral-Small-24B-Instruct-2501"
    temperature: float = 0.01


class Config(BaseSettings):
    reward_model_config: RewardModelConfig = RewardModelConfig()
    prompt_guard_config: PromptGuardConfig = PromptGuardConfig()
    scoring_client_config: ScoringClientConfig = ScoringClientConfig()
    prometheus_model_config: PrometheusModelConfig = PrometheusModelConfig()
    generate_model_config: GenerateModelConfig = GenerateModelConfig()

    class Config:
        env_nested_delimiter = "__"


CONFIG = Config()
