import logging

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """System level Configuration."""

    model_config = SettingsConfigDict(env_file=".env")  # Load env vars from .env file

    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    MILVUS_TOKEN: str

    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "bear"
    POSTGRES_URL: str = "postgresql://postgres:postgres@localhost:5432/bear"
    OPENALEX_MAILTO_EMAIL: str | None = None

    # API keys for external services
    OPENAI_API_KEY: str | None = None

    # This is the default embedding settings for all documents, which can be overridden in the per-document settings.
    DEFAULT_EMBEDDING_PROVIDER: str = "openai"
    DEFAULT_EMBEDDING_SERVER_URL: str = ""
    DEFAULT_EMBEDDING_MODEL: str = "text-embedding-3-large"
    DEFAULT_EMBEDDING_DIMS: int = 3072
    DEFAULT_EMBEDDING_MAX_TOKENS: int = 8191

    DEFAULT_HNSW_M: int = 32
    DEFAULT_HNSW_EF_CONSTRUCTION: int = 512


def make_logger() -> logging.Logger:
    """Make a logger."""

    logger = logging.getLogger(name="BEAR")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


LOGGER = make_logger()
CONFIG = Config()
