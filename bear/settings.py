import logging

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """System level Configuration."""

    model_config = SettingsConfigDict(env_file=".env")  # Load env vars from .env file

    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "bear"
    POSTGRES_URL: str = "postgresql://postgres:postgres@localhost:5432/bear"
    OPENAI_API_KEY: str = "your-openai-api-key"
    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    EMBEDDING_DIMS: int = 1536
    HNSW_M: int = 32
    HNSW_EF_CONSTRUCTION: int = 512
    CONTACT_EMAIL: str | None = None


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
