"""Tests for the crawler module."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
import pandas as pd
import pytest

from bear.crawler import (
    _dump,
    _get_page_results,
    get_openalex_id,
    query_openalex,
)
from bear.utils import strip_oa_prefix


class TestStripOAPrefix:
    """Test the strip_oa_prefix function."""

    def test_strips_openalex_prefix(self):
        """Test that OpenAlex prefix is correctly stripped."""
        assert strip_oa_prefix("https://openalex.org/A123456789") == "A123456789"
        assert strip_oa_prefix("https://openalex.org/I135310074") == "I135310074"
        assert strip_oa_prefix("https://openalex.org/W2755950973") == "W2755950973"

    def test_handles_already_stripped(self):
        """Test that already stripped IDs are returned unchanged."""
        assert strip_oa_prefix("A123456789") == "A123456789"
        assert strip_oa_prefix("I135310074") == "I135310074"

    def test_handles_empty_string(self):
        """Test that empty string is handled correctly."""
        assert strip_oa_prefix("") == ""


class TestGetOpenAlexId:
    """Test the get_openalex_id function."""

    @patch("bear.crawler.httpx.get")
    def test_successful_author_search(self, mock_get):
        """Test successful author ID retrieval."""
        mock_response = Mock()
        mock_response.json.return_value = {"results": [{"id": "https://openalex.org/A123456789", "display_name": "Jason Chor Ming Lo"}]}
        mock_get.return_value = mock_response

        result = get_openalex_id("authors", "Jason Chor Ming Lo")

        assert result == "A123456789"
        mock_get.assert_called_once()

    @patch("bear.crawler.httpx.get")
    def test_successful_institution_search(self, mock_get):
        """Test successful institution ID retrieval."""
        mock_response = Mock()
        mock_response.json.return_value = {"results": [{"id": "https://openalex.org/I135310074", "display_name": "University of Wisconsin-Madison"}]}
        mock_get.return_value = mock_response

        result = get_openalex_id("institutions", "University of Wisconsin-Madison")

        assert result == "I135310074"

    @patch("bear.crawler.httpx.get")
    def test_no_results_found(self, mock_get):
        """Test when no results are found."""
        mock_response = Mock()
        mock_response.json.return_value = {"results": []}
        mock_get.return_value = mock_response

        with pytest.raises(ValueError, match="No author found for query"):
            get_openalex_id("authors", "NonExistent Author")

    def test_invalid_entity_type(self):
        """Test invalid entity type raises ValueError."""
        with pytest.raises(ValueError, match="entity_type must be 'authors' or 'institutions'"):
            get_openalex_id("invalid", "test")

    @patch("bear.crawler.httpx.get")
    def test_http_error_retry_exhaustion(self, mock_get):
        """Test that HTTP errors cause retries and eventually raise."""
        mock_get.side_effect = httpx.HTTPError("Connection failed")

        with pytest.raises(httpx.HTTPError):
            get_openalex_id("authors", "Test Author")

    @patch("bear.crawler.config")
    @patch("bear.crawler.httpx.get")
    def test_includes_mailto_when_configured(self, mock_get, mock_config):
        """Test that mailto parameter is included when configured."""
        mock_config.OPENALEX_MAILTO_EMAIL = "test@example.com"
        mock_response = Mock()
        mock_response.json.return_value = {"results": [{"id": "https://openalex.org/A123456789", "display_name": "Test Author"}]}
        mock_get.return_value = mock_response

        get_openalex_id("authors", "Test Author")

        # Check that the URL includes the mailto parameter
        called_url = mock_get.call_args[0][0]
        assert "mailto=test@example.com" in called_url


class TestGetPageResults:
    """Test the _get_page_results function."""

    @patch("bear.crawler.httpx.get")
    def test_successful_page_retrieval(self, mock_get):
        """Test successful page retrieval."""
        mock_response = Mock()
        mock_response.json.return_value = {"meta": {"next_cursor": "cursor123"}, "results": [{"id": "A123"}, {"id": "A456"}]}
        mock_get.return_value = mock_response

        cursor, results = _get_page_results("authors", "test_query", "*")

        assert cursor == "cursor123"
        assert len(results) == 2
        assert results[0]["id"] == "A123"

    @patch("bear.crawler.httpx.get")
    def test_http_error_raises(self, mock_get):
        """Test that HTTP errors are raised for retry logic."""
        mock_get.side_effect = httpx.HTTPError("Connection failed")

        with pytest.raises(httpx.HTTPError):
            _get_page_results("authors", "test_query", "*")


class TestQueryOpenAlex:
    """Test the query_openalex function."""

    @patch("bear.crawler._get_page_results")
    def test_query_all_pages(self, mock_get_page):
        """Test querying all pages until exhausted."""
        # Mock multiple pages of results
        mock_get_page.side_effect = [
            ("cursor1", [{"id": "A1"}, {"id": "A2"}]),
            ("cursor2", [{"id": "A3"}, {"id": "A4"}]),
            (None, []),  # No more results
        ]

        results = query_openalex("authors", "test_query")

        assert len(results) == 4
        assert results[0]["id"] == "A1"
        assert results[-1]["id"] == "A4"

    @patch("bear.crawler._get_page_results")
    def test_query_with_limit(self, mock_get_page):
        """Test querying with API call limit."""
        mock_get_page.side_effect = [
            ("cursor1", [{"id": "A1"}, {"id": "A2"}]),
            ("cursor2", [{"id": "A3"}, {"id": "A4"}]),
        ]

        results = query_openalex("authors", "test_query", limit=2)

        assert len(results) == 4
        assert mock_get_page.call_count == 2


class TestDump:
    """Test the _dump function."""

    def test_dumps_data_to_parquet(self):
        """Test that data is correctly dumped to parquet file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_data = [{"id": "A123", "name": "Author 1"}, {"id": "A456", "name": "Author 2"}]

            _dump(test_data, temp_path / "test.parquet")

            # Verify file was created and contains correct data
            assert (temp_path / "test.parquet").exists()
            df = pd.read_parquet(temp_path / "test.parquet")
            assert len(df) == 2
            assert df.iloc[0]["id"] == "A123"

    def test_creates_parent_directories(self):
        """Test that parent directories are created if they don't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            nested_path = temp_path / "nested" / "deep" / "test.parquet"
            test_data = [{"id": "A123", "name": "Test"}]

            _dump(test_data, nested_path)

            assert nested_path.exists()
            assert nested_path.parent.exists()
