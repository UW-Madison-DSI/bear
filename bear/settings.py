import logging

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """System level Configuration."""

    model_config = SettingsConfigDict(env_file=".env")  # Load env vars from .env file

    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    MILVUS_TOKEN: str = ""

    # Milvus configuration
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_DB_NAME: str = "bear"
    MILVUS_COLLECTION_NAME: str = "academic_works"

    OPENALEX_MAILTO_EMAIL: str | None = None

    # API keys for external services
    OPENAI_API_KEY: str | None = None

    # This is the default embedding settings for all documents, which can be overridden in the per-document settings.
    DEFAULT_EMBEDDING_PROVIDER: str = "text-embedding-inference"
    DEFAULT_EMBEDDING_SERVER_URL: str = "http://olvi-1:8000"
    DEFAULT_EMBEDDING_MODEL: str = (
        "intfloat/multilingual-e5-large-instruct"  # https://huggingface.co/intfloat/multilingual-e5-large-instruct
    )
    DEFAULT_EMBEDDING_DIMS: int = 1024
    DEFAULT_EMBEDDING_MAX_TOKENS: int = 512

    # Milvus index parameters
    DEFAULT_INDEX_TYPE: str = "HNSW"
    DEFAULT_METRIC_TYPE: str = "IP"  # Inner Product
    DEFAULT_HNSW_M: int = 32
    DEFAULT_HNSW_EF_CONSTRUCTION: int = 512


def make_logger() -> logging.Logger:
    """Make a logger."""

    logger = logging.getLogger(name="BEAR")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


LOGGER = make_logger()
CONFIG = Config()
