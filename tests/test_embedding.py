import openai
import pytest

from bear.config import EmbeddingConfig
from bear.embedding import OpenAIEmbedder, Provider, TEIEmbedder, TextType, append_prefix, embed, get_embedder
from bear.model import Work


class DummyOpenAIClient:
    def __init__(self, *args, **kwargs):
        pass

    class embeddings:
        @staticmethod
        def create(model, input):
            class Data:
                embedding = [0.1, 0.2, 0.3]

            class Response:
                data = [Data() for _ in input]

            return Response()


class DummyTEIClient(DummyOpenAIClient):
    pass


class MockHttpxClient:
    def __init__(self, base_url):
        self.base_url = base_url

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def get(self, url):
        class MockResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"model_id": "test-model", "max_input_length": 1000}

        return MockResponse()


def test_append_prefix():
    assert append_prefix("hello", "prefix") == ["prefix hello"]
    assert append_prefix(["a", "b"], "p") == ["p a", "p b"]


def test_openai_embedder(monkeypatch):
    monkeypatch.setattr("bear.embedding.OpenAI", lambda *a, **kw: DummyOpenAIClient())
    cfg = EmbeddingConfig(
        provider="openai",
        server_url="https://api.openai.com/v1",
        model="test-model",
        dimensions=3,
        max_tokens=10,
        doc_prefix="doc",
        query_prefix="query",
        api_key=None,
        index_type="HNSW",
        metric_type="IP",
        hnsw_m=32,
        hnsw_ef_construction=512,
    )
    embedder = OpenAIEmbedder.from_config(cfg)
    assert embedder.model == "test-model"
    assert embedder.max_tokens == 10
    assert embedder.doc_prefix == "doc"
    assert embedder.query_prefix == "query"
    assert embedder.info["provider"] == Provider.OPENAI
    dims = embedder.get_dimensions()
    assert isinstance(dims, int)
    out = embedder.embed(["text1", "text2"], TextType.DOC)
    assert isinstance(out, list)
    assert all(isinstance(v, list) for v in out)


def test_tei_embedder(monkeypatch):
    monkeypatch.setattr("bear.embedding.OpenAI", lambda *a, **kw: DummyTEIClient())
    monkeypatch.setattr("bear.embedding.httpx.Client", MockHttpxClient)
    cfg = EmbeddingConfig(
        provider="tei",
        server_url="http://localhost",
        model="test-model",
        dimensions=3,
        max_tokens=10,
        doc_prefix="doc",
        query_prefix="query",
        api_key=None,
        index_type="HNSW",
        metric_type="IP",
        hnsw_m=32,
        hnsw_ef_construction=512,
    )
    embedder = TEIEmbedder.from_config(cfg)
    assert embedder.model == "test-model"
    assert embedder.max_tokens == 10
    assert embedder.info["provider"] == Provider.TEXT_EMBEDDING_INFERENCE
    dims = embedder.get_dimensions()
    assert isinstance(dims, int)
    out = embedder.embed(["text1", "text2"], TextType.QUERY)
    assert isinstance(out, list)
    assert all(isinstance(v, list) for v in out)


def test_get_embedder(monkeypatch):
    monkeypatch.setattr("bear.embedding.OpenAI", lambda *a, **kw: DummyOpenAIClient())
    cfg = EmbeddingConfig(
        provider="openai",
        server_url="https://api.openai.com/v1",
        model="test-model",
        dimensions=3,
        max_tokens=10,
        doc_prefix="",
        query_prefix="",
        api_key=None,
        index_type="HNSW",
        metric_type="IP",
        hnsw_m=32,
        hnsw_ef_construction=512,
    )
    embedder = get_embedder(cfg)
    assert isinstance(embedder, OpenAIEmbedder)
    cfg2 = EmbeddingConfig(
        provider="tei",
        server_url="http://localhost",
        model="test-model",
        dimensions=3,
        max_tokens=10,
        doc_prefix="",
        query_prefix="",
        api_key=None,
        index_type="HNSW",
        metric_type="IP",
        hnsw_m=32,
        hnsw_ef_construction=512,
    )
    monkeypatch.setattr("bear.embedding.httpx.Client", MockHttpxClient)
    embedder2 = get_embedder(cfg2)
    assert isinstance(embedder2, TEIEmbedder)
    with pytest.raises(ValueError):
        get_embedder(
            EmbeddingConfig(
                provider="unknown",
                server_url="https://api.openai.com/v1",
                model="test-model",
                dimensions=3,
                max_tokens=10,
                doc_prefix="",
                query_prefix="",
                api_key=None,
                index_type="HNSW",
                metric_type="IP",
                hnsw_m=32,
                hnsw_ef_construction=512,
            )
        )


def test_embed_works(monkeypatch):
    monkeypatch.setattr("bear.embedding.OpenAI", lambda *a, **kw: DummyOpenAIClient())
    works = [
        Work(
            primary_key=None,
            id=str(i),
            doi=None,
            title=f"title{i}",
            display_name=None,
            publication_year=None,
            publication_date=None,
            type=None,
            cited_by_count=None,
            is_retracted=None,
            is_paratext=None,
            cited_by_api_url=None,
            abstract_inverted_index={},
            source_id=None,
            source_display_name=None,
            topics=[],
            is_oa=None,
            pdf_url=None,
            landing_page_url=None,
            embedding=[],
        )
        for i in range(5)
    ]
    cfg = EmbeddingConfig(
        provider="openai",
        server_url="https://api.openai.com/v1",
        model="test-model",
        dimensions=3,
        max_tokens=10,
        doc_prefix="",
        query_prefix="",
        api_key=None,
        index_type="HNSW",
        metric_type="IP",
        hnsw_m=32,
        hnsw_ef_construction=512,
    )
    result = embed(works, batch_size=2, embedding_config=cfg)
    assert all(hasattr(w, "embedding") and isinstance(w.embedding, list) for w in result)


def test_text_type_enum():
    """Test TextType enum values."""
    assert TextType.DOC == "doc"
    assert TextType.QUERY == "query"


def test_provider_enum():
    """Test Provider enum values."""
    assert Provider.OPENAI == "openai"
    assert Provider.TEXT_EMBEDDING_INFERENCE == "tei"


def test_openai_embedder_with_prefixes(monkeypatch):
    """Test OpenAI embedder applies prefixes correctly."""
    monkeypatch.setattr("bear.embedding.OpenAI", lambda *a, **kw: DummyOpenAIClient())

    embedder = OpenAIEmbedder(model="test-model", max_tokens=100, doc_prefix="doc:", query_prefix="query:")

    # Test single string with doc prefix
    result = embedder.embed("test text", TextType.DOC)
    assert isinstance(result, list)
    assert len(result) == 1

    # Test single string with query prefix
    result = embedder.embed("test query", TextType.QUERY)
    assert isinstance(result, list)
    assert len(result) == 1


def test_tei_embedder_server_validation_error(monkeypatch):
    """Test TEI embedder server validation fails with mismatched model."""
    monkeypatch.setattr("bear.embedding.OpenAI", lambda *a, **kw: DummyTEIClient())

    class MockHttpxClientBadModel:
        def __init__(self, base_url):
            self.base_url = base_url

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def get(self, url):
            class MockResponse:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"model_id": "different-model", "max_input_length": 1000}

            return MockResponse()

    monkeypatch.setattr("bear.embedding.httpx.Client", MockHttpxClientBadModel)

    with pytest.raises(ValueError, match="Model ID test-model does not match server's model ID different-model"):
        TEIEmbedder(model="test-model", max_tokens=100, base_url="http://localhost")


def test_embed_model_not_found_error():
    """Test that embed methods raise not found error for invalid model."""
    embedder = OpenAIEmbedder(model="test", max_tokens=100)
    with pytest.raises(openai.NotFoundError):
        embedder.embed("test", "doc")


def test_embed_text_type_value_error():
    """Test that embed methods raise value error for invalid text_type."""
    embedder = OpenAIEmbedder(model="test", max_tokens=100)
    with pytest.raises(ValueError):
        embedder.embed("test", "invalid_type")
