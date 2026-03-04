from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    # OpenRouter (OpenAI-compatible)
    openrouter_api_key: str = Field(..., env="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1", env="OPENROUTER_BASE_URL"
    )

    # Internal API security
    api_key: str = Field(..., env="API_KEY")

    # Vector store
    chroma_path: str = Field(default="./chroma_db", env="CHROMA_PATH")

    # LLM settings
    llm_model: str = Field(default="openai/gpt-4o-mini", env="LLM_MODEL")
    # HuggingFace sentence-transformers model for local embeddings (no API key needed)
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
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

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
