import os
from unittest.mock import patch

import pytest
from pydantic import SecretStr, ValidationError

from bear.config import Config, EmbeddingConfig


class TestEmbeddingConfig:
    """Test cases for EmbeddingConfig class."""

    def test_embedding_config_creation(self):
        """Test creating an EmbeddingConfig with all required fields."""
        config = EmbeddingConfig(
            provider="openai",
            server_url="https://api.openai.com/v1",
            model="text-embedding-3-large",
            dimensions=3072,
            max_tokens=512,
            doc_prefix="doc: ",
            query_prefix="query: ",
            api_key=SecretStr("test-api-key"),
            index_type="HNSW",
            metric_type="IP",
            hnsw_m=32,
            hnsw_ef_construction=512,
        )

        assert config.provider == "openai"
        assert config.server_url == "https://api.openai.com/v1"
        assert config.model == "text-embedding-3-large"
        assert config.dimensions == 3072
        assert config.max_tokens == 512
        assert config.doc_prefix == "doc: "
        assert config.query_prefix == "query: "
        assert config.api_key is not None
        assert config.api_key.get_secret_value() == "test-api-key"
        assert config.index_type == "HNSW"
        assert config.metric_type == "IP"
        assert config.hnsw_m == 32
        assert config.hnsw_ef_construction == 512

    def test_embedding_config_optional_api_key(self):
        """Test creating an EmbeddingConfig without api_key."""
        config = EmbeddingConfig(
            provider="tei",
            server_url="http://localhost:8080",
            model="all-MiniLM-L6-v2",
            dimensions=384,
            max_tokens=256,
            doc_prefix="",
            query_prefix="",
            index_type="HNSW",
            metric_type="COSINE",
            hnsw_m=16,
            hnsw_ef_construction=256,
        )

        assert config.provider == "tei"
        assert config.api_key is None

    def test_index_config_property(self):
        """Test the index_config property returns correct dictionary."""
        config = EmbeddingConfig(
            provider="openai",
            server_url="https://api.openai.com/v1",
            model="text-embedding-3-large",
            dimensions=3072,
            max_tokens=512,
            doc_prefix="",
            query_prefix="",
            index_type="HNSW",
            metric_type="IP",
            hnsw_m=32,
            hnsw_ef_construction=512,
        )

        index_config = config.index_config

        expected_config = {
            "index_type": "HNSW",
            "metric_type": "IP",
            "params": {
                "M": 32,
                "efConstruction": 512,
            },
        }

        assert index_config == expected_config

    def test_index_config_with_different_values(self):
        """Test index_config with different HNSW parameters."""
        config = EmbeddingConfig(
            provider="tei",
            server_url="http://localhost:8080",
            model="all-MiniLM-L6-v2",
            dimensions=384,
            max_tokens=256,
            doc_prefix="",
            query_prefix="",
            index_type="IVF_FLAT",
            metric_type="L2",
            hnsw_m=64,
            hnsw_ef_construction=1024,
        )

        index_config = config.index_config

        expected_config = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {
                "M": 64,
                "efConstruction": 1024,
            },
        }

        assert index_config == expected_config

    def test_embedding_config_validation(self):
        """Test that EmbeddingConfig validates required fields."""
        with pytest.raises(ValidationError):
            # Missing required fields should raise validation error
            EmbeddingConfig()


class TestConfig:
    """Test cases for Config class."""

    def setup_method(self):
        """Setup method to clear environment variables before each test."""
        # Store original values
        self.original_env = {}
        env_vars = [
            "POSTGRES_USER",
            "POSTGRES_PASSWORD",
            "POSTGRES_URL",
            "MINIO_ACCESS_KEY",
            "MINIO_SECRET_KEY",
            "MILVUS_TOKEN",
            "MILVUS_HOST",
            "MILVUS_PORT",
            "MILVUS_DB_NAME",
            "MILVUS_COLLECTION_NAME",
            "OPENALEX_MAILTO_EMAIL",
            "OPENAI_API_KEY",
            "TEI_API_KEY",
            "DEFAULT_EMBEDDING_PROVIDER",
            "DEFAULT_EMBEDDING_SERVER_URL",
            "DEFAULT_EMBEDDING_MODEL",
            "DEFAULT_EMBEDDING_DIMS",
            "DEFAULT_EMBEDDING_MAX_TOKENS",
            "DEFAULT_EMBEDDING_DOC_PREFIX",
            "DEFAULT_EMBEDDING_QUERY_PREFIX",
            "DEFAULT_INDEX_TYPE",
            "DEFAULT_METRIC_TYPE",
            "DEFAULT_HNSW_M",
            "DEFAULT_HNSW_EF_CONSTRUCTION",
            "LOG_LEVEL",
        ]

        for var in env_vars:
            self.original_env[var] = os.environ.get(var)
            if var in os.environ:
                del os.environ[var]

    def teardown_method(self):
        """Restore original environment variables after each test."""
        for var, value in self.original_env.items():
            if value is not None:
                os.environ[var] = value
            elif var in os.environ:
                del os.environ[var]

    def test_config_default_values(self, isolated_config):
        """Test Config with default values."""
        config = isolated_config()

        # Test database defaults
        assert config.POSTGRES_USER is None
        assert config.POSTGRES_PASSWORD is None
        assert config.POSTGRES_URL is None

        # Test Milvus defaults
        assert config.MINIO_ACCESS_KEY is None
        assert config.MINIO_SECRET_KEY is None
        assert config.MILVUS_TOKEN is None
        assert config.MILVUS_HOST == "localhost"
        assert config.MILVUS_PORT == 19530
        assert config.MILVUS_DB_NAME == "dev"
        assert config.MILVUS_COLLECTION_NAME == "academic_works"

        # Test API defaults
        assert config.OPENALEX_MAILTO_EMAIL == ""
        assert config.OPENAI_API_KEY is None
        assert config.TEI_API_KEY is None

        # Test embedding defaults
        assert config.DEFAULT_EMBEDDING_PROVIDER == "openai"
        assert config.DEFAULT_EMBEDDING_SERVER_URL == "https://api.openai.com/v1"
        assert config.DEFAULT_EMBEDDING_MODEL == "text-embedding-3-large"
        assert config.DEFAULT_EMBEDDING_DIMS == 3072
        assert config.DEFAULT_EMBEDDING_MAX_TOKENS == 512
        assert config.DEFAULT_EMBEDDING_DOC_PREFIX == ""
        assert config.DEFAULT_EMBEDDING_QUERY_PREFIX == ""

        # Test index defaults
        assert config.DEFAULT_INDEX_TYPE == "HNSW"
        assert config.DEFAULT_METRIC_TYPE == "IP"
        assert config.DEFAULT_HNSW_M == 32
        assert config.DEFAULT_HNSW_EF_CONSTRUCTION == 512

        # Test logging default
        assert config.LOG_LEVEL == "DEBUG"

    def test_config_from_environment_variables(self, clean_environment):
        """Test Config loading from environment variables."""
        # Set environment variables
        os.environ["POSTGRES_USER"] = "test_user"
        os.environ["POSTGRES_PASSWORD"] = "test_password"
        os.environ["POSTGRES_URL"] = "postgresql://localhost:5432/test"
        os.environ["MILVUS_HOST"] = "test-milvus"
        os.environ["MILVUS_PORT"] = "19531"
        os.environ["MILVUS_DB_NAME"] = "test_db"
        os.environ["OPENAI_API_KEY"] = "test-openai-key"
        os.environ["DEFAULT_EMBEDDING_PROVIDER"] = "tei"
        os.environ["DEFAULT_EMBEDDING_MODEL"] = "custom-model"
        os.environ["DEFAULT_EMBEDDING_DIMS"] = "768"
        os.environ["LOG_LEVEL"] = "INFO"

        config = Config()

        # Verify values are loaded from environment
        assert config.POSTGRES_USER is not None
        assert config.POSTGRES_USER.get_secret_value() == "test_user"
        assert config.POSTGRES_PASSWORD is not None
        assert config.POSTGRES_PASSWORD.get_secret_value() == "test_password"
        assert config.POSTGRES_URL is not None
        assert config.POSTGRES_URL.get_secret_value() == "postgresql://localhost:5432/test"
        assert config.MILVUS_HOST == "test-milvus"
        assert config.MILVUS_PORT == 19531
        assert config.MILVUS_DB_NAME == "test_db"
        assert config.OPENAI_API_KEY is not None
        assert config.OPENAI_API_KEY.get_secret_value() == "test-openai-key"
        assert config.DEFAULT_EMBEDDING_PROVIDER == "tei"
        assert config.DEFAULT_EMBEDDING_MODEL == "custom-model"
        assert config.DEFAULT_EMBEDDING_DIMS == 768
        assert config.LOG_LEVEL == "INFO"

    @patch("bear.config.logger")
    def test_default_embedding_api_key_openai(self, mock_logger, clean_environment):
        """Test DEFAULT_EMBEDDING_API_KEY property with OpenAI provider."""
        os.environ["OPENAI_API_KEY"] = "test-openai-key"
        os.environ["DEFAULT_EMBEDDING_PROVIDER"] = "openai"
        config = Config()

        api_key = config.DEFAULT_EMBEDDING_API_KEY

        assert api_key is not None
        assert api_key.get_secret_value() == "test-openai-key"
        mock_logger.info.assert_called()

    @patch("bear.config.logger")
    def test_default_embedding_api_key_tei(self, mock_logger, clean_environment):
        """Test DEFAULT_EMBEDDING_API_KEY property with TEI provider."""
        os.environ["DEFAULT_EMBEDDING_PROVIDER"] = "tei"
        os.environ["TEI_API_KEY"] = "test-tei-key"
        config = Config()

        api_key = config.DEFAULT_EMBEDDING_API_KEY

        assert api_key is not None
        assert api_key.get_secret_value() == "test-tei-key"
        mock_logger.info.assert_called()

    def test_default_embedding_api_key_unknown_provider(self, clean_environment):
        """Test DEFAULT_EMBEDDING_API_KEY property with unknown provider."""
        os.environ["DEFAULT_EMBEDDING_PROVIDER"] = "unknown"
        config = Config()

        api_key = config.DEFAULT_EMBEDDING_API_KEY

        assert api_key is None

    def test_default_embedding_api_key_no_key_set(self, isolated_config):
        """Test DEFAULT_EMBEDDING_API_KEY property when no API key is set."""
        config = isolated_config()

        api_key = config.DEFAULT_EMBEDDING_API_KEY

        assert api_key is None

    @patch("bear.config.logger")
    def test_embedding_config_property(self, mock_logger):
        """Test the embedding_config property returns correct EmbeddingConfig."""
        os.environ["OPENAI_API_KEY"] = "test-openai-key"
        os.environ["DEFAULT_EMBEDDING_PROVIDER"] = "openai"
        os.environ["DEFAULT_EMBEDDING_SERVER_URL"] = "https://custom.api.com/v1"
        os.environ["DEFAULT_EMBEDDING_MODEL"] = "custom-model"
        os.environ["DEFAULT_EMBEDDING_DIMS"] = "1536"
        os.environ["DEFAULT_EMBEDDING_MAX_TOKENS"] = "1024"
        os.environ["DEFAULT_EMBEDDING_DOC_PREFIX"] = "document: "
        os.environ["DEFAULT_EMBEDDING_QUERY_PREFIX"] = "query: "
        os.environ["DEFAULT_INDEX_TYPE"] = "IVF_FLAT"
        os.environ["DEFAULT_METRIC_TYPE"] = "L2"
        os.environ["DEFAULT_HNSW_M"] = "64"
        os.environ["DEFAULT_HNSW_EF_CONSTRUCTION"] = "1024"

        config = Config()
        embedding_config = config.embedding_config

        assert isinstance(embedding_config, EmbeddingConfig)
        assert embedding_config.provider == "openai"
        assert embedding_config.server_url == "https://custom.api.com/v1"
        assert embedding_config.model == "custom-model"
        assert embedding_config.dimensions == 1536
        assert embedding_config.max_tokens == 1024
        assert embedding_config.doc_prefix == "document: "
        assert embedding_config.query_prefix == "query: "
        assert embedding_config.api_key is not None
        assert embedding_config.api_key.get_secret_value() == "test-openai-key"
        assert embedding_config.index_type == "IVF_FLAT"
        assert embedding_config.metric_type == "L2"
        assert embedding_config.hnsw_m == 64
        assert embedding_config.hnsw_ef_construction == 1024

    def test_embedding_config_index_config_integration(self):
        """Test that embedding_config.index_config works correctly."""
        os.environ["DEFAULT_INDEX_TYPE"] = "HNSW"
        os.environ["DEFAULT_METRIC_TYPE"] = "IP"
        os.environ["DEFAULT_HNSW_M"] = "16"
        os.environ["DEFAULT_HNSW_EF_CONSTRUCTION"] = "256"

        config = Config()
        embedding_config = config.embedding_config
        index_config = embedding_config.index_config

        expected_config = {
            "index_type": "HNSW",
            "metric_type": "IP",
            "params": {
                "M": 16,
                "efConstruction": 256,
            },
        }

        assert index_config == expected_config

    def test_secret_fields_are_secret_str(self):
        """Test that secret fields are properly handled as SecretStr."""
        os.environ["POSTGRES_USER"] = "secret_user"
        os.environ["POSTGRES_PASSWORD"] = "secret_pass"
        os.environ["OPENAI_API_KEY"] = "secret_key"

        config = Config()

        assert isinstance(config.POSTGRES_USER, SecretStr)
        assert isinstance(config.POSTGRES_PASSWORD, SecretStr)
        assert isinstance(config.OPENAI_API_KEY, SecretStr)

        # Test that the actual values are accessible via get_secret_value()
        assert config.POSTGRES_USER.get_secret_value() == "secret_user"
        assert config.POSTGRES_PASSWORD.get_secret_value() == "secret_pass"
        assert config.OPENAI_API_KEY.get_secret_value() == "secret_key"

    def test_integer_field_conversion(self):
        """Test that string environment variables are properly converted to integers."""
        os.environ["MILVUS_PORT"] = "19532"
        os.environ["DEFAULT_EMBEDDING_DIMS"] = "2048"
        os.environ["DEFAULT_EMBEDDING_MAX_TOKENS"] = "1024"
        os.environ["DEFAULT_HNSW_M"] = "48"
        os.environ["DEFAULT_HNSW_EF_CONSTRUCTION"] = "768"

        config = Config()

        assert config.MILVUS_PORT == 19532
        assert config.DEFAULT_EMBEDDING_DIMS == 2048
        assert config.DEFAULT_EMBEDDING_MAX_TOKENS == 1024
        assert config.DEFAULT_HNSW_M == 48
        assert config.DEFAULT_HNSW_EF_CONSTRUCTION == 768
        assert isinstance(config.MILVUS_PORT, int)
        assert isinstance(config.DEFAULT_EMBEDDING_DIMS, int)


class TestConfigIntegration:
    """Integration tests for config module."""

    def test_config_module_imports(self):
        """Test that config module can be imported and instantiated."""
        from bear.config import Config, config, logger

        # Test that global config instance exists
        assert isinstance(config, Config)

        # Test that logger is configured
        assert logger.name == "BEAR"

    @patch.dict(
        os.environ,
        {
            "DEFAULT_EMBEDDING_PROVIDER": "openai",
            "OPENAI_API_KEY": "test-key",
            "DEFAULT_EMBEDDING_MODEL": "text-embedding-3-small",
            "DEFAULT_EMBEDDING_DIMS": "1536",
        },
    )
    def test_full_config_workflow(self):
        """Test a complete workflow using the config."""
        config = Config()

        # Get embedding config
        embedding_config = config.embedding_config

        # Verify it's properly configured
        assert embedding_config.provider == "openai"
        assert embedding_config.model == "text-embedding-3-small"
        assert embedding_config.dimensions == 1536
        assert embedding_config.api_key is not None
        assert embedding_config.api_key.get_secret_value() == "test-key"

        # Get index config
        index_config = embedding_config.index_config
        assert "index_type" in index_config
        assert "metric_type" in index_config
        assert "params" in index_config
        assert "M" in index_config["params"]
        assert "efConstruction" in index_config["params"]
