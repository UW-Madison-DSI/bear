"""Tests for bear.db module."""

from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel
from pymilvus import MilvusClient

from bear.db import create_milvus_collection, get_milvus_client, init
from bear.model import ALL_RESOURCES, Work


class TestGetMilvusClient:
    """Test cases for get_milvus_client function."""

    @patch("bear.db.config")
    @patch("bear.db.MilvusClient")
    def test_get_milvus_client_with_token(self, mock_milvus_client, mock_config):
        """Test getting Milvus client with token."""
        mock_config.MILVUS_HOST = "localhost"
        mock_config.MILVUS_PORT = "19530"
        mock_config.MILVUS_TOKEN = "test-token"

        mock_client_instance = Mock()
        mock_milvus_client.return_value = mock_client_instance

        result = get_milvus_client()

        mock_milvus_client.assert_called_once_with(uri="http://localhost:19530", token="test-token")
        assert result == mock_client_instance

    @patch("bear.db.config")
    @patch("bear.db.MilvusClient")
    def test_get_milvus_client_without_token(self, mock_milvus_client, mock_config):
        """Test getting Milvus client without token."""
        mock_config.MILVUS_HOST = "localhost"
        mock_config.MILVUS_PORT = "19530"
        mock_config.MILVUS_TOKEN = None

        mock_client_instance = Mock()
        mock_milvus_client.return_value = mock_client_instance

        result = get_milvus_client()

        mock_milvus_client.assert_called_once_with(uri="http://localhost:19530", token="")
        assert result == mock_client_instance

    @patch("bear.db.config")
    @patch("bear.db.MilvusClient")
    def test_get_milvus_client_empty_token(self, mock_milvus_client, mock_config):
        """Test getting Milvus client with empty token."""
        mock_config.MILVUS_HOST = "localhost"
        mock_config.MILVUS_PORT = "19530"
        mock_config.MILVUS_TOKEN = ""

        mock_client_instance = Mock()
        mock_milvus_client.return_value = mock_client_instance

        result = get_milvus_client()

        mock_milvus_client.assert_called_once_with(uri="http://localhost:19530", token="")
        assert result == mock_client_instance


class TestCreateMilvusCollection:
    """Test cases for create_milvus_collection function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = Mock(spec=MilvusClient)
        self.mock_schema = Mock()
        self.mock_index_params = Mock()

        self.mock_client.create_schema.return_value = self.mock_schema
        self.mock_client.prepare_index_params.return_value = self.mock_index_params

    def test_create_milvus_collection_success(self):
        """Test successful creation of Milvus collection."""
        self.mock_client.has_collection.return_value = False

        create_milvus_collection(self.mock_client, Work)

        # Verify collection creation flow
        self.mock_client.has_collection.assert_called_once_with("work")
        self.mock_client.create_schema.assert_called_once_with(auto_id=True, enable_dynamic_field=True)
        self.mock_client.prepare_index_params.assert_called_once()
        self.mock_client.create_collection.assert_called_once()

        # Verify schema fields were added
        assert self.mock_schema.add_field.call_count > 0

        # Get the create_collection call arguments
        create_collection_call = self.mock_client.create_collection.call_args
        assert create_collection_call[1]["collection_name"] == "work"
        assert create_collection_call[1]["schema"] == self.mock_schema
        assert create_collection_call[1]["index_params"] == self.mock_index_params

    def test_create_milvus_collection_already_exists(self):
        """Test creating collection when it already exists."""
        self.mock_client.has_collection.return_value = True

        create_milvus_collection(self.mock_client, Work)

        # Verify only existence check was made
        self.mock_client.has_collection.assert_called_once_with("work")
        self.mock_client.create_schema.assert_not_called()
        self.mock_client.create_collection.assert_not_called()

    def test_create_milvus_collection_unregistered_model(self):
        """Test creating collection with unregistered model."""

        # Create a dummy model not in ALL_MODELS
        class UnregisteredModel(BaseModel):
            name: str

        with pytest.raises(ValueError, match="Model .* is not registered in bear.model.ALL_MODELS"):
            create_milvus_collection(self.mock_client, UnregisteredModel)


class TestInit:
    """Test cases for init function."""

    @patch("bear.db.get_milvus_client")
    @patch("bear.db.create_milvus_collection")
    @patch("bear.db.config")
    def test_init_creates_database_and_collections(self, mock_config, mock_create_collection, mock_get_client):
        """Test init function creates database and collections."""
        # Set up the mock config to return our test value
        mock_config.MILVUS_DB_NAME = "test_db"

        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.list_databases.return_value = ["default"]  # test_db not in list

        # Call init with the mocked db_name parameter
        init(db_name="test_db")

        # Verify database creation
        mock_client.create_database.assert_called_once_with(db_name="test_db")
        mock_client.use_database.assert_called_once_with("test_db")

        # Verify collections creation for all models
        assert mock_create_collection.call_count == len(ALL_RESOURCES)
        for model in ALL_RESOURCES:
            mock_create_collection.assert_any_call(client=mock_client, model=model)

    @patch("bear.db.get_milvus_client")
    @patch("bear.db.create_milvus_collection")
    @patch("bear.db.config")
    def test_init_database_already_exists(self, mock_config, mock_create_collection, mock_get_client):
        """Test init function when database already exists."""
        mock_config.MILVUS_DB_NAME = "test_db"

        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.list_databases.return_value = ["default", "test_db"]  # test_db in list

        # Call init with the db_name parameter
        init(db_name="test_db")

        # Verify database creation was not called
        mock_client.create_database.assert_not_called()
        mock_client.use_database.assert_called_once_with("test_db")

        # Verify collections creation still happens
        assert mock_create_collection.call_count == len(ALL_RESOURCES)

    @patch("bear.db.get_milvus_client")
    @patch("bear.db.create_milvus_collection")
    def test_init_with_custom_db_name(self, mock_create_collection, mock_get_client):
        """Test init function with custom database name."""
        custom_db_name = "custom_test_db"

        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.list_databases.return_value = ["default"]

        init(db_name=custom_db_name)

        # Verify custom database name is used
        mock_client.create_database.assert_called_once_with(db_name=custom_db_name)
        mock_client.use_database.assert_called_once_with(custom_db_name)

        # Verify collections creation
        assert mock_create_collection.call_count == len(ALL_RESOURCES)
        for model in ALL_RESOURCES:
            mock_create_collection.assert_any_call(client=mock_client, model=model)

    @patch("bear.db.get_milvus_client")
    @patch("bear.db.logger")
    def test_init_logs_database_creation(self, mock_logger, mock_get_client):
        """Test that init function logs database creation."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.list_databases.return_value = ["default"]

        with patch("bear.db.create_milvus_collection"):
            init("new_db")

        # Verify logging
        mock_logger.info.assert_any_call("Creating database: new_db")


class TestIntegration:
    """Integration tests for db module."""

    @patch("bear.db.MilvusClient")
    @patch("bear.db.config")
    def test_full_workflow_integration(self, mock_config, mock_milvus_client):
        """Test the full workflow from client creation to collection initialization."""
        # Setup mocks
        mock_config.MILVUS_HOST = "localhost"
        mock_config.MILVUS_PORT = "19530"
        mock_config.MILVUS_TOKEN = "test-token"
        mock_config.MILVUS_DB_NAME = "test_db"

        mock_client = Mock()
        mock_milvus_client.return_value = mock_client
        mock_client.list_databases.return_value = ["default"]
        mock_client.has_collection.return_value = False

        # Mock schema and index params
        mock_schema = Mock()
        mock_index_params = Mock()
        mock_client.create_schema.return_value = mock_schema
        mock_client.prepare_index_params.return_value = mock_index_params

        # Run init with explicit db_name
        init(db_name="test_db")

        # Verify the full workflow
        mock_milvus_client.assert_called_once_with(uri="http://localhost:19530", token="test-token")
        mock_client.create_database.assert_called_once_with(db_name="test_db")
        mock_client.use_database.assert_called_once_with("test_db")

        # Verify collection creation for each model
        expected_calls = len(ALL_RESOURCES)
        assert mock_client.create_collection.call_count == expected_calls
