"""Env-driven configuration (pydantic-settings, .env file supported)."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from .embeddings import Embedder, build_embedder
from .llm import LLMClient, MockLLM, OpenAICompatibleLLM

PROVIDER_PRESETS: dict[str, dict[str, str | None]] = {
    "openai": {"base_url": None},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1"},
    "custom": {"base_url": None},
    "mock": {"base_url": None},
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: str = "openai"
    openai_api_key: str | None = None
    openrouter_api_key: str | None = None
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o-2024-08-06"

    embedding_backend: str = "auto"  # auto | sentence-transformers | tfidf
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    hydra_db_api_key: str | None = None

    data_dir: Path = Path("data")
    store_path: Path = Path("data/hydra_mem.sqlite3")
    cache_dir: Path = Path("cache")
    max_concurrency: int = 8
    top_k: int = 12
    request_timeout: float = 60.0
    max_retries: int = 5
    log_level: str = "INFO"

    def resolve_credentials(self) -> tuple[str | None, str]:
        """Return (base_url, api_key) for the configured provider.

        Raises ValueError naming the exact env var to set when misconfigured.
        """
        provider = self.llm_provider.lower()
        if provider == "mock":
            return None, "mock"
        if provider not in PROVIDER_PRESETS:
            raise ValueError(
                f"Unknown LLM_PROVIDER={provider!r} (expected openai|openrouter|custom|mock)"
            )
        base_url = self.llm_base_url or PROVIDER_PRESETS[provider]["base_url"]
        if provider == "openai":
            key, var = self.openai_api_key or self.llm_api_key, "OPENAI_API_KEY"
        elif provider == "openrouter":
            key, var = self.openrouter_api_key or self.llm_api_key, "OPENROUTER_API_KEY"
        else:  # custom
            key, var = self.llm_api_key or self.openai_api_key, "LLM_API_KEY"
            if not base_url:
                raise ValueError("LLM_PROVIDER=custom requires LLM_BASE_URL to be set")
        if not key:
            raise ValueError(f"Missing API key: set {var} in .env (or use LLM_PROVIDER=mock)")
        return base_url, key


def build_llm(settings: Settings, *, force_mock: bool = False) -> LLMClient:
    """The single factory for LLM clients. `force_mock` powers --mock CLI runs."""
    if force_mock or settings.llm_provider.lower() == "mock":
        return MockLLM()
    base_url, api_key = settings.resolve_credentials()
    return OpenAICompatibleLLM(
        api_key=api_key,
        base_url=base_url,
        model=settings.llm_model,
        timeout=settings.request_timeout,
        max_retries=settings.max_retries,
    )


def build_default_embedder(settings: Settings) -> Embedder:
    return build_embedder(settings.embedding_backend, settings.embedding_model)
