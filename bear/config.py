import logging

from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingConfig(BaseModel):
    provider: str
    server_url: str
    model: str
    dimensions: int
    max_tokens: int
    doc_prefix: str = ""
    query_prefix: str = ""
    api_key: SecretStr | None = None

    # Index settings
    index_type: str = "HNSW"
    metric_type: str = "IP"
    hnsw_m: int = 64
    hnsw_ef_construction: int = 256

    @property
    def index_config(self) -> dict:
        """Return the index configuration dict for Milvus. Note. Missing `field_name` should be injected from the model definition."""
        assert self.index_type == "HNSW", "Only HNSW index type is supported in BEAR for now. Send a PR if you need other index types."

        return {
            "index_type": self.index_type,
            "metric_type": self.metric_type,
            "params": {
                "M": self.hnsw_m,
                "efConstruction": self.hnsw_ef_construction,
            },
        }


class Config(BaseSettings):
    """System configuration. Settings are defined in `.env`. Refer to `example.env` for details."""

    model_config = SettingsConfigDict(env_file=".env")

    # (Optional) OpenAlex data dump database
    POSTGRES_USER: SecretStr | None = None
    POSTGRES_PASSWORD: SecretStr | None = None
    POSTGRES_URL: SecretStr | None = None

    # Milvus
    MINIO_ACCESS_KEY: SecretStr | None = None
    MINIO_SECRET_KEY: SecretStr | None = None
    MILVUS_TOKEN: SecretStr | None = None
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_DB_NAME: str = "dev"

    # External APIs
    OPENALEX_MAILTO_EMAIL: str = ""
    OPENAI_API_KEY: SecretStr | None = None
    TEI_API_KEY: SecretStr | None = None

    # Embeddings
    DEFAULT_EMBEDDING_PROVIDER: str = "openai"
    DEFAULT_EMBEDDING_SERVER_URL: str = "https://api.openai.com/v1"
    DEFAULT_EMBEDDING_MODEL: str = "text-embedding-3-large"
    DEFAULT_EMBEDDING_DIMS: int = 3072
    DEFAULT_EMBEDDING_MAX_TOKENS: int = 512
    DEFAULT_EMBEDDING_DOC_PREFIX: str = ""
    DEFAULT_EMBEDDING_QUERY_PREFIX: str = ""

    # Embeddings Index
    DEFAULT_INDEX_TYPE: str = "HNSW"
    DEFAULT_METRIC_TYPE: str = "IP"
    DEFAULT_HNSW_M: int = 32
    DEFAULT_HNSW_EF_CONSTRUCTION: int = 512

    # Logging
    LOG_LEVEL: str = "DEBUG"

    # Other integrations
    TQDM_SLACK_TOKEN: SecretStr | None = None
    TQDM_SLACK_CHANNEL: str = "general"

    @property
    def DEFAULT_EMBEDDING_API_KEY(self) -> SecretStr | None:
        """Return the default embedding API key based on the provider."""
        if self.DEFAULT_EMBEDDING_PROVIDER == "openai":
            logger.debug(f"Using OpenAI API key for default key: {self.OPENAI_API_KEY}")
            return self.OPENAI_API_KEY
        elif self.DEFAULT_EMBEDDING_PROVIDER == "tei":
            logger.debug(f"Using Text Embedding Inference API key for default key: {self.TEI_API_KEY}")
            return self.TEI_API_KEY
        return None

    @property
    def default_embedding_config(self) -> EmbeddingConfig:
        """Return the default embedding configuration."""
        return EmbeddingConfig(
            provider=self.DEFAULT_EMBEDDING_PROVIDER,
            server_url=self.DEFAULT_EMBEDDING_SERVER_URL,
            model=self.DEFAULT_EMBEDDING_MODEL,
            dimensions=self.DEFAULT_EMBEDDING_DIMS,
            max_tokens=self.DEFAULT_EMBEDDING_MAX_TOKENS,
            doc_prefix=self.DEFAULT_EMBEDDING_DOC_PREFIX,
            query_prefix=self.DEFAULT_EMBEDDING_QUERY_PREFIX,
            api_key=self.DEFAULT_EMBEDDING_API_KEY,
            index_type=self.DEFAULT_INDEX_TYPE,
            metric_type=self.DEFAULT_METRIC_TYPE,
            hnsw_m=self.DEFAULT_HNSW_M,
            hnsw_ef_construction=self.DEFAULT_HNSW_EF_CONSTRUCTION,
        )


# Global instances
config = Config()
logging.basicConfig(level=config.LOG_LEVEL, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("BEAR")

logger.debug(f"Configuration loaded: {config}")
