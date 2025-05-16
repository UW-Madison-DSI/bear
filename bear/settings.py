import logging

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """System level Configuration."""

    model_config = SettingsConfigDict(env_file=".env")  # Load env vars from .env file

    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    OPENAI_API_KEY: str
    POSTGRES_URL: str
    POSTGRES_LOCAL_URL: str
    EMBEDDING_MODEL: str
    EMBEDDING_DIMS: int
    HNSW_M: int
    HNSW_EF_CONSTRUCTION: int
    CONTACT_EMAIL: str


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
CONFIG = Config()  # type: ignore
