"""Tests for the FastAPI endpoints."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from bear.api.main import app


@pytest.fixture
def client():
    """FastAPI test client fixture."""
    return TestClient(app)


@pytest.fixture
def mock_search_engine():
    """Mock SearchEngine for testing."""
    with patch("bear.api.app_state") as mock_app_state:
        mock_engine = patch("bear.search.SearchEngine").start()
        mock_app_state.__getitem__.return_value = mock_engine
        yield mock_engine
        patch.stopall()


class TestAPI:
    """Test cases for API endpoints."""

    def test_read_root(self, client):
        """Test the root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        assert "Instruction" in response.json()

    def test_search_resource_success(self, client, mock_search_engine):
        """Test successful resource search."""
        # Mock search results
        mock_results = [
            {
                "entity": {
                    "id": "test_id_1",
                    "doi": "10.1000/test1",
                    "title": "Test Paper 1",
                    "display_name": "Test Paper 1",
                    "publication_year": 2023,
                    "publication_date": "2023-01-01",
                    "type": "article",
                    "cited_by_count": 10,
                    "source_display_name": "Test Journal",
                    "topics": ["machine learning", "AI"],
                    "author_ids": ["author1", "author2"],
                },
                "distance": 0.85,
            }
        ]
        mock_search_engine.search_resource.return_value = mock_results

        # Mock the Work._recover_abstract method
        with patch("bear.api.Work._recover_abstract", return_value="Test abstract"):
            response = client.get("/search_resource?query=machine learning&top_k=1")

        assert response.status_code == 200
        results = response.json()
        assert len(results) == 1
        assert results[0]["id"] == "test_id_1"
        assert results[0]["title"] == "Test Paper 1"
        assert results[0]["distance"] == 0.85

    def test_search_resource_no_results(self, client, mock_search_engine):
        """Test resource search with no results."""
        mock_search_engine.search_resource.return_value = []

        response = client.get("/search_resource?query=nonexistent")
        assert response.status_code == 404
        assert "No results found" in response.json()["detail"]

    def test_search_resource_with_parameters(self, client, mock_search_engine):
        """Test resource search with various parameters."""
        mock_search_engine.search_resource.return_value = [
            {
                "entity": {
                    "id": "test_id",
                    "title": "Test",
                },
                "distance": 0.9,
            }
        ]

        response = client.get("/search_resource?query=test&top_k=5&resource_name=work&min_distance=0.8&since_year=2020")

        assert response.status_code == 200
        # Verify the search_resource was called with correct parameters
        mock_search_engine.search_resource.assert_called_once_with(
            resource_name="work",
            query="test",
            top_k=5,
            min_distance=0.8,
            since_year=2020,
        )

    def test_search_resource_error(self, client, mock_search_engine):
        """Test resource search with search engine error."""
        mock_search_engine.search_resource.side_effect = Exception("Search engine error")

        response = client.get("/search_resource?query=test")
        assert response.status_code == 500
        assert "Search failed" in response.json()["detail"]

    def test_search_author_success(self, client, mock_search_engine):
        """Test successful author search."""
        mock_results = [
            {"author_id": "author1", "score": 0.95},
            {"author_id": "author2", "score": 0.88},
        ]
        mock_search_engine.search_author.return_value = mock_results

        response = client.get("/search_author?query=machine learning researcher")
        assert response.status_code == 200
        results = response.json()
        assert len(results) == 2
        assert results[0]["author_id"] == "author1"
        assert results[0]["score"] == 0.95

    def test_search_author_no_results(self, client, mock_search_engine):
        """Test author search with no results."""
        mock_search_engine.search_author.return_value = []

        response = client.get("/search_author?query=nonexistent author")
        assert response.status_code == 404
        assert "No results found" in response.json()["detail"]

    def test_search_author_with_parameters(self, client, mock_search_engine):
        """Test author search with various parameters."""
        mock_search_engine.search_author.return_value = [{"author_id": "author1", "score": 0.9}]

        response = client.get("/search_author?query=researcher&top_k=10&institutions=uw-madison&min_distance=0.7&since_year=2021")

        assert response.status_code == 200
        # Verify the search_author was called with correct parameters
        mock_search_engine.search_author.assert_called_once_with(
            query="researcher",
            top_k=10,
            institutions=["uw-madison"],
            min_distance=0.7,
            since_year=2021,
        )

    def test_search_author_error(self, client, mock_search_engine):
        """Test author search with search engine error."""
        mock_search_engine.search_author.side_effect = Exception("Author search error")

        response = client.get("/search_author?query=test")
        assert response.status_code == 500
        assert "Author search failed" in response.json()["detail"]

    def test_search_resource_abstract_recovery(self, client, mock_search_engine):
        """Test abstract recovery from inverted index."""
        mock_results = [
            {
                "entity": {
                    "id": "test_id",
                    "title": "Test Paper",
                    "abstract_inverted_index": {"test": [0], "abstract": [1]},
                },
                "distance": 0.9,
            }
        ]
        mock_search_engine.search_resource.return_value = mock_results

        with patch("bear.api.Work._recover_abstract", return_value="Recovered abstract") as mock_recover:
            response = client.get("/search_resource?query=test")

        assert response.status_code == 200
        results = response.json()
        assert results[0]["abstract"] == "Recovered abstract"
        mock_recover.assert_called_once_with({"test": [0], "abstract": [1]})

    def test_search_resource_no_abstract(self, client, mock_search_engine):
        """Test resource search when no abstract is available."""
        mock_results = [
            {
                "entity": {
                    "id": "test_id",
                    "title": "Test Paper",
                    # No abstract_inverted_index
                },
                "distance": 0.9,
            }
        ]
        mock_search_engine.search_resource.return_value = mock_results

        response = client.get("/search_resource?query=test")
        assert response.status_code == 200
        results = response.json()
        assert results[0]["abstract"] is None


class TestEmbedEndpoint:
    """Test cases for the /embed endpoint."""

    def test_embed_success_query(self, client):
        """Test successful embedding generation with query type."""
        # Mock the embedder
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        
        with patch("bear.api.main.get_embedder", return_value=mock_embedder):
            response = client.post(
                "/embed",
                json={"texts": ["test query 1", "test query 2"], "type": "query"}
            )
        
        assert response.status_code == 200
        result = response.json()
        assert "embeddings" in result
        assert len(result["embeddings"]) == 2
        assert result["embeddings"][0] == [0.1, 0.2, 0.3]
        assert result["embeddings"][1] == [0.4, 0.5, 0.6]
        mock_embedder.embed.assert_called_once_with(text=["test query 1", "test query 2"], text_type="query")

    def test_embed_success_doc(self, client):
        """Test successful embedding generation with doc type."""
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [[0.7, 0.8, 0.9]]
        
        with patch("bear.api.main.get_embedder", return_value=mock_embedder):
            response = client.post(
                "/embed",
                json={"texts": ["test document"], "type": "doc"}
            )
        
        assert response.status_code == 200
        result = response.json()
        assert len(result["embeddings"]) == 1
        assert result["embeddings"][0] == [0.7, 0.8, 0.9]
        mock_embedder.embed.assert_called_once_with(text=["test document"], text_type="doc")

    def test_embed_success_raw(self, client):
        """Test successful embedding generation with raw type."""
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [[0.1, 0.2, 0.3]]
        
        with patch("bear.api.main.get_embedder", return_value=mock_embedder):
            response = client.post(
                "/embed",
                json={"texts": ["query: test"], "type": "raw"}
            )
        
        assert response.status_code == 200
        result = response.json()
        assert len(result["embeddings"]) == 1
        assert result["embeddings"][0] == [0.1, 0.2, 0.3]
        mock_embedder.embed.assert_called_once_with(text=["query: test"], text_type="raw")

    def test_embed_default_type(self, client):
        """Test that default type is 'query' when not specified."""
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [[0.1, 0.2, 0.3]]
        
        with patch("bear.api.main.get_embedder", return_value=mock_embedder):
            response = client.post(
                "/embed",
                json={"texts": ["test"]}
            )
        
        assert response.status_code == 200
        mock_embedder.embed.assert_called_once_with(text=["test"], text_type="query")

    def test_embed_multiple_texts(self, client):
        """Test embedding multiple texts at once."""
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
        ]
        
        with patch("bear.api.main.get_embedder", return_value=mock_embedder):
            response = client.post(
                "/embed",
                json={"texts": ["text1", "text2", "text3"], "type": "doc"}
            )
        
        assert response.status_code == 200
        result = response.json()
        assert len(result["embeddings"]) == 3

    def test_embed_invalid_type(self, client):
        """Test that invalid type returns 422 validation error."""
        response = client.post(
            "/embed",
            json={"texts": ["test"], "type": "invalid"}
        )
        
        assert response.status_code == 422

    def test_embed_empty_texts(self, client):
        """Test embedding with empty texts list."""
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = []
        
        with patch("bear.api.main.get_embedder", return_value=mock_embedder):
            response = client.post(
                "/embed",
                json={"texts": [], "type": "query"}
            )
        
        assert response.status_code == 200
        result = response.json()
        assert result["embeddings"] == []

    def test_embed_error_handling(self, client):
        """Test error handling when embedding fails."""
        mock_embedder = MagicMock()
        mock_embedder.embed.side_effect = Exception("Embedding error")
        
        with patch("bear.api.main.get_embedder", return_value=mock_embedder):
            response = client.post(
                "/embed",
                json={"texts": ["test"], "type": "query"}
            )
        
        assert response.status_code == 500
        assert "Embedding generation failed" in response.json()["detail"]


class TestEmbedInfoEndpoint:
    """Test cases for the /embed/info endpoint."""

    def test_embed_info_success(self, client):
        """Test successful retrieval of embedding info."""
        # Mock the embedder
        mock_embedder = MagicMock()
        mock_embedder.info = {
            "provider": "openai",
            "model": "text-embedding-3-large",
            "dimensions": 3072,
            "doc_prefix": "passage: ",
            "query_prefix": "query: "
        }
        
        # Mock the config
        mock_config = MagicMock()
        mock_config.default_embedding_config.max_tokens = 512
        
        with patch("bear.api.main.get_embedder", return_value=mock_embedder), \
             patch("bear.config.config", mock_config):
            response = client.get("/embed/info")
        
        assert response.status_code == 200
        result = response.json()
        assert result["provider"] == "openai"
        assert result["model"] == "text-embedding-3-large"
        assert result["dimensions"] == 3072
        assert result["max_tokens"] == 512
        assert result["doc_prefix"] == "passage: "
        assert result["query_prefix"] == "query: "

    def test_embed_info_tei_provider(self, client):
        """Test embedding info with TEI provider."""
        mock_embedder = MagicMock()
        mock_embedder.info = {
            "provider": "tei",
            "model": "BAAI/bge-large-en-v1.5",
            "dimensions": 1024,
            "doc_prefix": "",
            "query_prefix": ""
        }
        
        mock_config = MagicMock()
        mock_config.default_embedding_config.max_tokens = 256
        
        with patch("bear.api.main.get_embedder", return_value=mock_embedder), \
             patch("bear.config.config", mock_config):
            response = client.get("/embed/info")
        
        assert response.status_code == 200
        result = response.json()
        assert result["provider"] == "tei"
        assert result["model"] == "BAAI/bge-large-en-v1.5"
        assert result["dimensions"] == 1024
        assert result["max_tokens"] == 256
        assert result["doc_prefix"] == ""
        assert result["query_prefix"] == ""

    def test_embed_info_error_handling(self, client):
        """Test error handling when retrieving embedding info fails."""
        with patch("bear.api.main.get_embedder", side_effect=Exception("Config error")):
            response = client.get("/embed/info")
        
        assert response.status_code == 500
        assert "Failed to retrieve embedding info" in response.json()["detail"]
