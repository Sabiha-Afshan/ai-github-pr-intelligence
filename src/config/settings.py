"""Central application configuration."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    app_name: str = "AI GitHub PR Intelligence"
    app_env: str = "development"
    log_level: str = "INFO"

    github_token: str | None = Field(
        default=None,
        repr=False,
    )

    github_repository_owner: str = "pallets"
    github_repository_name: str = "flask"

    github_api_base_url: str = "https://api.github.com"
    github_api_version: str = "2022-11-28"
    github_request_timeout_seconds: float = 30.0
    github_max_retries: int = 3

    github_retry_backoff_seconds: float = 1.0
    github_rate_limit_buffer_seconds: int = 5
    github_rate_limit_wait_enabled: bool = True
    github_default_page_size: int = 100
    github_max_pages: int = 100

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5-coder:3b"

    data_directory: Path = Path("data")
    model_directory: Path = Path("models")
    knowledge_base_directory: Path = Path("knowledge_base")
    evaluation_directory: Path = Path("evaluation")
    log_directory: Path = Path("logs")

    default_delay_threshold_hours: int = 48
    random_seed: int = 42

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def github_repository_full_name(self) -> str:
        """Return the configured GitHub owner and repository name."""

        return f"{self.github_repository_owner}/{self.github_repository_name}"


@lru_cache
def get_settings() -> Settings:
    """Return one cached application-settings object."""

    return Settings()
