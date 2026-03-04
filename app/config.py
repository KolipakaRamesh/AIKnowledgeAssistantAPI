import os
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    # OpenRouter (OpenAI-compatible)
    openrouter_api_key: str = Field(default="", env="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1", env="OPENROUTER_BASE_URL"
    )

    # Internal API security
    api_key: str = Field(default="", env="API_KEY")

    # OpenAI / OpenRouter key for Embeddings (defaults to openrouter_api_key if empty)
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")

    # Vector store (use /tmp on Vercel for writable filesystem)
    chroma_path: str = Field(
        default="/tmp/chroma_db" if os.getenv("VERCEL") == "1" else "./chroma_db",
        env="CHROMA_PATH"
    )

    # LLM settings
    llm_model: str = Field(default="openai/gpt-4o-mini", env="LLM_MODEL")
    # Cloud-based embeddings (text-embedding-3-small is cheap and fast)
    embedding_model: str = Field(default="text-embedding-3-small", env="EMBEDDING_MODEL")
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")

    # Rate limiting
    rate_limit_global: str = Field(default="60/minute", env="RATE_LIMIT_GLOBAL")
    rate_limit_upload: str = Field(default="10/minute", env="RATE_LIMIT_UPLOAD")

    # CORS
    allowed_origins: str = Field(default="*", env="ALLOWED_ORIGINS")

    # LangSmith tracing (newer env var names)
    langsmith_tracing: str = Field(default="false", env="LANGSMITH_TRACING")
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com", env="LANGSMITH_ENDPOINT"
    )
    langsmith_api_key: str = Field(default="", env="LANGSMITH_API_KEY")
    langsmith_project: str = Field(
        default="AIKnowledgeAssistantAPI", env="LANGSMITH_PROJECT"
    )

    def validate_keys(self):
        """Check for missing keys without crashing on startup."""
        missing = []
        if not self.openrouter_api_key: missing.append("OPENROUTER_API_KEY")
        if not self.api_key: missing.append("API_KEY")
        return missing

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
