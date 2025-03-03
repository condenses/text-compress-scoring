from pydantic_settings import BaseSettings
import torch
from pydantic import model_validator


class PromptGuardConfig(BaseSettings):
    model_name: str = "katanemo/Arch-Guard"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    class Config:
        extra = "ignore"
        env_prefix = "PROMPT_GUARD_CONFIG__"


class ScoringClientConfig(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 9102
    timeout: float = 64.0

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    class Config:
        extra = "ignore"
        env_prefix = "SCORING_CLIENT_CONFIG__"


class vLLMConfig(BaseSettings):
    base_url: str = "http://localhost:8080"
    timeout: float = 128.0
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    temperature: float = 0.01
    max_tokens: int = 1024
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.0
    max_new_tokens: int = 1024
    api_key: str = "sk-proj-1234567890"

    class Config:
        extra = "ignore"
        env_prefix = "VLLM_CONFIG__"


class Config(BaseSettings):
    vllm_config: vLLMConfig = vLLMConfig()
    scoring_client_config: ScoringClientConfig = ScoringClientConfig()
    prompt_guard_config: PromptGuardConfig = PromptGuardConfig()
    wallet_name: str = "default"
    wallet_path: str = "~/.bittensor/wallets"
    wallet_hotkey: str = "default"
    use_nineteen_api: bool = False

    @model_validator(mode="after")
    def update_model_name(self) -> "Config":
        if self.use_nineteen_api:
            self.vllm_config.model_name = "chat-llama-3-1-70b"
        return self

    class Config:
        env_nested_delimiter = "__"
        env_file = ".env"
        extra = "ignore"
        env_ignore = [".*"]


CONFIG = Config()

from rich.console import Console
from rich.panel import Panel

console = Console()
settings_dict = CONFIG.model_dump()

for section, values in settings_dict.items():
    console.print(
        Panel.fit(
            str(values),
            title=f"[bold blue]{section}[/bold blue]",
            border_style="green",
        )
    )
