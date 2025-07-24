"""Pytest configuration for bear tests."""

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def clean_environment():
    """Clean environment variables before each test."""
    # Store original environment
    original_env = dict(os.environ)

    # Clear environment variables that could affect config
    env_vars_to_clear = [
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_URL",
        "MINIO_ACCESS_KEY",
        "MINIO_SECRET_KEY",
        "MILVUS_TOKEN",
        "MILVUS_HOST",
        "MILVUS_PORT",
        "MILVUS_DB_NAME",
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

    for var in env_vars_to_clear:
        if var in os.environ:
            del os.environ[var]

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def temp_env_file():
    """Create a temporary .env file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
        f.write("# Test environment file\n")
        f.write("TEST_VAR=test_value\n")
        temp_path = f.name

    yield temp_path

    # Clean up
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def isolated_config():
    """Fixture to create a Config class that doesn't load .env files."""
    from pydantic_settings import SettingsConfigDict

    from bear.config import Config

    class IsolatedConfig(Config):
        model_config = SettingsConfigDict(env_file=None)

    return IsolatedConfig
