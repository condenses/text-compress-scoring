from pydantic_settings import BaseSettings
import torch


class PromptGuardConfig(BaseSettings):
    model_name: str = "katanemo/Arch-Guard"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ScoringClientConfig(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 9102
    timeout: float = 64.0

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class vLLMConfig(BaseSettings):
    base_url: str = "http://localhost:8080"
    timeout: float = 128.0
    model_name: str = "mistralai/Mistral-Small-24B-Instruct-2501"
    temperature: float = 0.01
    max_tokens: int = 1024
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.0
    max_new_tokens: int = 1024
    api_key: str = "sk-proj-1234567890"


class Config(BaseSettings):
    vllm_config: vLLMConfig = vLLMConfig()
    scoring_client_config: ScoringClientConfig = ScoringClientConfig()
    prompt_guard_config: PromptGuardConfig = PromptGuardConfig()

    class Config:
        env_nested_delimiter = "__"


CONFIG = Config()
