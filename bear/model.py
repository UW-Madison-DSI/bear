from typing import Annotated, Any

import httpx
from pydantic import BaseModel, Field, WithJsonSchema
from pymilvus import DataType

from bear.config import config


def _clean_inverted_index(inverted_index: dict[str, Any]) -> dict[str, list[int]]:
    """Cleans the abstract inverted index by converting values to lists of int."""
    return {k: list(map(int, v)) for k, v in inverted_index.items() if v is not None}


class EmbeddingConfig(BaseModel):
    provider: str = config.DEFAULT_EMBEDDING_PROVIDER
    model_id: str = config.DEFAULT_EMBEDDING_MODEL
    dimensions: int = config.DEFAULT_EMBEDDING_DIMS

    # Index settings
    index_type: str = config.DEFAULT_INDEX_TYPE
    metric_type: str = config.DEFAULT_METRIC_TYPE
    hnsw_m: int = config.DEFAULT_HNSW_M
    hnsw_ef_construction: int = config.DEFAULT_HNSW_EF_CONSTRUCTION

    @property
    def index_config(self) -> dict:
        """Return the index configuration dict for Milvus. Note. Missing `field_name` should be injected from the model definition."""
        return {
            "index_type": self.index_type,
            "metric_type": self.metric_type,
            "params": {
                "M": self.hnsw_m,
                "efConstruction": self.hnsw_ef_construction,
            },
        }


class Work(BaseModel):
    """Work model ORM.

    `pymilvus` don't have any validation build-in, so this model use `pydantic` to validate the data before inserting into Milvus. It also store the Milvus collection schema for collection instantiation. Each the dictionary inside `WithJsonSchema` is a `pymilvus.FieldSchema.to_dict()` used for instantiating the Milvus collection.

    Example:
        ```python
        from bear.models import Work
        from pymilvus import FieldSchema, DataType

        title_schema = FieldSchema(name="title", datatype=DataType.VARCHAR, max_length=2048)
        class NewCollectionName(BaseModel):
            id: Annotated[int, WithJsonSchema({"datatype": DataType.INT64, "is_primary": True})]  # Easier to use
            title: Annotated[str, WithJsonSchema(title_schema.to_dict())]  # Safer to use
            ...
        ```

    """

    # OpenAlex Works fields
    primary_key: Annotated[int | None, Field(default=None), WithJsonSchema({"datatype": DataType.INT64, "is_primary": True})]
    id: Annotated[str | None, WithJsonSchema({"datatype": DataType.VARCHAR, "max_length": 512, "index_configs": {"index_type": "AUTOINDEX"}, "nullable": True})]
    doi: Annotated[
        str | None, WithJsonSchema({"datatype": DataType.VARCHAR, "max_length": 512, "index_configs": {"index_type": "AUTOINDEX"}, "nullable": True})
    ]
    title: Annotated[str | None, WithJsonSchema({"datatype": DataType.VARCHAR, "max_length": 2048, "nullable": True})]
    display_name: Annotated[str | None, WithJsonSchema({"datatype": DataType.VARCHAR, "max_length": 2048, "nullable": True})]
    publication_year: Annotated[int | None, WithJsonSchema({"datatype": DataType.INT64, "nullable": True})]
    publication_date: Annotated[str | None, WithJsonSchema({"datatype": DataType.VARCHAR, "max_length": 32, "nullable": True})]
    type: Annotated[str | None, WithJsonSchema({"datatype": DataType.VARCHAR, "max_length": 256, "nullable": True})]
    cited_by_count: Annotated[int | None, WithJsonSchema({"datatype": DataType.INT64, "nullable": True})]
    is_retracted: Annotated[bool | None, WithJsonSchema({"datatype": DataType.BOOL, "nullable": True})]
    is_paratext: Annotated[bool | None, WithJsonSchema({"datatype": DataType.BOOL, "nullable": True})]
    cited_by_api_url: Annotated[str | None, WithJsonSchema({"datatype": DataType.VARCHAR, "max_length": 2048, "nullable": True})]
    abstract_inverted_index: Annotated[dict[str, list[int]], Field(default_factory=dict), WithJsonSchema({"datatype": DataType.JSON})]

    # Additional field via default works API
    source_id: Annotated[str | None, WithJsonSchema({"datatype": DataType.VARCHAR, "max_length": 512, "nullable": True})]
    source_display_name: Annotated[str | None, WithJsonSchema({"datatype": DataType.VARCHAR, "max_length": 512, "nullable": True})]
    topics: Annotated[list[str], Field(default_factory=list), WithJsonSchema({"datatype": DataType.JSON})]
    is_oa: Annotated[bool | None, Field(default=None), WithJsonSchema({"datatype": DataType.BOOL, "nullable": True})]
    pdf_url: Annotated[str | None, Field(default=None), WithJsonSchema({"datatype": DataType.VARCHAR, "max_length": 2048, "nullable": True})]
    landing_page_url: Annotated[str | None, Field(default=None), WithJsonSchema({"datatype": DataType.VARCHAR, "max_length": 2048, "nullable": True})]

    embedding: Annotated[
        list[float],
        Field(default_factory=list),
        WithJsonSchema({"datatype": DataType.FLOAT_VECTOR, "dim": EmbeddingConfig().dimensions, "index_configs": EmbeddingConfig().index_config}),
    ]

    @property
    def abstract(self) -> str:
        """Recover the abstract from the inverted index."""
        return self._recover_abstract(self.abstract_inverted_index)

    @staticmethod
    def _recover_abstract(inverted_index: dict[str, list[int]]) -> str:
        """Recover the abstract from the inverted index."""
        if not inverted_index:
            return ""
        word_positions = [(pos, word) for word, positions in inverted_index.items() if positions for pos in positions]
        word_positions.sort()
        return " ".join(word for _, word in word_positions)

    @staticmethod
    def parse(data: dict) -> dict:
        """Parse a work from OpenAlex raw data to local Work format."""

        primary_location = data.get("primary_location") or {}
        source = primary_location.get("source") or {}
        best_oa_location = data.get("best_oa_location") or {}

        return {
            "id": data.get("id"),
            "doi": data.get("doi"),
            "title": data.get("title"),
            "display_name": data.get("display_name"),
            "publication_year": data.get("publication_year"),
            "publication_date": data.get("publication_date"),
            "type": data.get("type"),
            "cited_by_count": data.get("cited_by_count"),
            "is_retracted": data.get("is_retracted"),
            "is_paratext": data.get("is_paratext"),
            "cited_by_api_url": data.get("cited_by_api_url"),
            "abstract_inverted_index": _clean_inverted_index(data.get("abstract_inverted_index") or {}),
            "source_id": source.get("id"),
            "source_display_name": source.get("display_name"),
            "topics": [topic.get("display_name") for topic in data.get("topics", [])],
            "is_oa": best_oa_location.get("is_oa"),
            "pdf_url": best_oa_location.get("pdf_url"),
            "landing_page_url": best_oa_location.get("landing_page_url"),
        }

    @classmethod
    def pull(cls, doi: str) -> "Work":
        """Pull a work from the OpenAlex by DOI."""
        response = httpx.get(f"https://api.openalex.org/works/doi:{doi}")
        response.raise_for_status()
        data = response.json()
        return cls(**cls.parse(data))

    @classmethod
    def from_raw(cls, data: dict) -> "Work":
        """Create a Work from raw data."""
        return cls(**cls.parse(data))

    def __str__(self) -> str:
        """Return a string representation of the work."""
        text = ""
        if self.title:
            text += f"title: {self.title}"
        if self.source_display_name:
            text += f"\njournal:{self.source_display_name}"
        if self.topics:
            text += f"\ntopics: {', '.join(self.topics)}"
        if self.abstract:
            text += f"\nabstract: {self.abstract}"
        return text

    def to_milvus(self) -> dict:
        """Convert to dictionary for Milvus insertion."""
        return self.model_dump()


ALL_MODELS = [Work]
