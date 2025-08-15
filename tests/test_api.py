"""Tests for the FastAPI endpoints."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.main import app


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
