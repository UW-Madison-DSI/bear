import logging

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """System level Configuration."""

    model_config = SettingsConfigDict(env_file=".env")

    # Database
    POSTGRES_USER: str = ""
    POSTGRES_PASSWORD: str = ""
    POSTGRES_URL: str = ""

    # Milvus
    MINIO_ACCESS_KEY: str = ""
    MINIO_SECRET_KEY: str = ""
    MILVUS_TOKEN: str = ""
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_DB_NAME: str = "dev"
    MILVUS_COLLECTION_NAME: str = "academic_works"

    # External APIs
    OPENALEX_MAILTO_EMAIL: str | None = None
    OPENAI_API_KEY: str | None = None

    # Embeddings
    DEFAULT_EMBEDDING_PROVIDER: str = "text-embedding-inference"
    DEFAULT_EMBEDDING_SERVER_URL: str = "http://localhost:8000"
    DEFAULT_EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large-instruct"
    DEFAULT_EMBEDDING_DIMS: int = 1024
    DEFAULT_EMBEDDING_MAX_TOKENS: int = 512

    # Index
    DEFAULT_INDEX_TYPE: str = "HNSW"
    DEFAULT_METRIC_TYPE: str = "IP"
    DEFAULT_HNSW_M: int = 32
    DEFAULT_HNSW_EF_CONSTRUCTION: int = 512


# Global instances
logger = logging.getLogger("BEAR")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

config = Config()
