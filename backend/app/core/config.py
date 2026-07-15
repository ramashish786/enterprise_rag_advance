"""Centralized settings, loaded from env with sensible dev defaults.

Everything reads settings via the module-level `settings` instance rather than
os.environ directly, so tests can override cleanly.
"""
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- App ---
    app_env: Literal["development", "test", "staging", "production"] = "development"
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:5173"

    # --- Auth ---
    jwt_secret_key: str = "dev-only-secret-do-not-use-in-prod"
    jwt_algorithm: str = "HS256"
    access_token_ttl_minutes: int = 15
    refresh_token_ttl_days: int = 7

    # --- Database ---
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "insurance_agent"
    postgres_user: str = "insurance_app"
    postgres_password: str = "insurance_app"
    postgres_readonly_user: str = "insurance_ro"
    postgres_readonly_password: str = "insurance_ro"

    # In-memory sqlite is used for the fast unit-test path.
    test_database_url: str = "sqlite+aiosqlite:///:memory:"

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"

    # --- LLM ---
    openai_api_key: str = ""
    primary_llm_model: str = "gpt-4o"
    router_llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 1536

    # --- Guardrails ---
    enable_pii_redaction: bool = True
    enable_prompt_injection_check: bool = True
    enable_output_validation: bool = True
    max_tokens_per_user_per_day: int = 100_000

    # --- Retrieval ---
    retrieval_top_k_dense: int = 20
    retrieval_top_k_bm25: int = 20
    rerank_top_k: int = 5
    reranker_enabled: bool = True

    # --- Observability ---
    langsmith_api_key: str = ""
    langsmith_project: str = "insurance-agent"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def readonly_database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_readonly_user}:"
            f"{self.postgres_readonly_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_test(self) -> bool:
        return self.app_env == "test"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
