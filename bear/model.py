from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Protocol, Self

import httpx
from pydantic import BaseModel, Field, WithJsonSchema
from pymilvus import DataType

from bear.config import EmbeddingConfig, config
from bear.crawler import strip_oa_prefix


def _clean_inverted_index(inverted_index: dict[str, Any]) -> dict[str, list[int]]:
    """Cleans the abstract inverted index by converting values to lists of int."""
    if not inverted_index:
        return {}
    return {k: list(map(int, v)) for k, v in inverted_index.items() if v is not None}


DUMMY_EMBEDDING_CONFIG = {
    "datatype": DataType.FLOAT_VECTOR,
    "dim": 2,
    "index_configs": {"index_type": "HNSW", "metric_type": "IP", "params": {"M": 2, "efConstruction": 2}},
}


class CollectionProtocol(Protocol):
    """Protocol for resources or clusters that can be stored in Milvus."""

    @property
    def _name(self) -> str: ...  # Name of the resource for Milvus collection
    @classmethod
    def embedding_config(cls) -> EmbeddingConfig | None: ...  # Embedding configuration for the resource
    @staticmethod
    def parse(raw_data: dict, *args, **kwargs) -> dict: ...  # Parse raw data to a dictionary suitable for the resource
    @classmethod
    def from_raw(cls, raw_data: dict, *args, **kwargs) -> Self: ...  # Create an instance from raw data
    def model_dump(self) -> dict: ...  # Convert the resource to a dictionary for Milvus insertion
    def __str__(self) -> str: ...  # Return a string representation of the resource, used for embeddings.


class Person(BaseModel):
    """Represents a person entity."""

    id: Annotated[str, WithJsonSchema({"datatype": DataType.VARCHAR, "max_length": 64, "is_primary": True})]
    display_name: Annotated[str, WithJsonSchema({"datatype": DataType.VARCHAR, "max_length": 128})]
    institution_id: Annotated[str, WithJsonSchema({"datatype": DataType.VARCHAR, "max_length": 64})]
    embedding: Annotated[list[float], Field(default_factory=lambda: [0, 0]), WithJsonSchema(DUMMY_EMBEDDING_CONFIG)]

    @property
    def _name(self) -> str:
        """Return the name of the model for Milvus collection."""
        return self.__class__.__name__.lower()

    @staticmethod
    def parse(raw_data: dict, institution_id: str) -> dict:
        """User institution_id and raw_data to create a Person instance."""

        # Verify institution_id is in the raw data
        institutions = raw_data.get("last_known_institutions", [])
        ids = [strip_oa_prefix(inst["id"]) for inst in institutions]
        assert institution_id in ids, f"Expected institution_id {institution_id} not found in {ids}."
        return {
            "id": raw_data.get("id"),
            "display_name": raw_data.get("display_name"),
            "institution_id": institution_id,
        }

    @classmethod
    def from_raw(cls, raw_data: dict, institution_id: str) -> Self:
        return cls(**cls.parse(raw_data, institution_id))

    @classmethod
    def embedding_config(cls) -> EmbeddingConfig | None:
        return None


class Work(BaseModel):
    """Work from OpenAlex.

    This model use `pydantic` to validate the data before inserting into Milvus. It also store the Milvus collection schema for collection instantiation. Each the dictionary inside `WithJsonSchema` is a `pymilvus.FieldSchema.to_dict()` used for instantiating the Milvus collection.

    Example:
        ```python
        from bear.model import Work
        from pymilvus import FieldSchema, DataType

        title_schema = FieldSchema(name="title", datatype=DataType.VARCHAR, max_length=2048)
        class NewCollectionName(BaseModel):

            # Put Milvus `FieldSchema` inside `WithJsonSchema`.
            # WithJsonSchema data can be access with `.model_fields["field_name"].metadata[0].json_schema`
            id: Annotated[int, WithJsonSchema({"datatype": DataType.INT64, "is_primary": True})]  # Easier to use
            title: Annotated[str, WithJsonSchema(title_schema.to_dict())]  # Safer to use
            ...
        ```

    """

    # OpenAlex Works fields
    id: Annotated[str, WithJsonSchema({"datatype": DataType.VARCHAR, "is_primary": True, "max_length": 64, "index_configs": {"index_type": "AUTOINDEX"}})]
    doi: Annotated[
        str | None, WithJsonSchema({"datatype": DataType.VARCHAR, "max_length": 256, "index_configs": {"index_type": "AUTOINDEX"}, "nullable": True})
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

    # Denormalized authors (Milvus does not support nested objects)
    author_ids: Annotated[
        list[str] | None,
        Field(default_factory=list),
        WithJsonSchema({"datatype": DataType.ARRAY, "element_type": DataType.VARCHAR, "max_capacity": 2048, "nullable": True, "max_length": 64}),
    ]

    embedding: Annotated[
        list[float] | None,
        Field(default_factory=list),
        WithJsonSchema(
            {
                "datatype": DataType.FLOAT_VECTOR,
                "dim": config.default_embedding_config.dimensions,
                "index_configs": config.default_embedding_config.index_config,
            }
        ),
    ]

    # Misc
    ignore: Annotated[bool, Field(default=False), WithJsonSchema({"datatype": DataType.BOOL})]
    last_modified: Annotated[
        str, Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d")), WithJsonSchema({"datatype": DataType.VARCHAR, "max_length": 32})
    ]

    @property
    def _name(self) -> str:
        """Return the name of the model for Milvus collection."""
        return self.__class__.__name__.lower()

    @classmethod
    def embedding_config(cls) -> EmbeddingConfig:
        """Return the embedding configuration for the model."""
        return config.default_embedding_config

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
    def parse(raw_data: dict) -> dict:
        """Parse a work from OpenAlex raw data to local Work format."""

        primary_location = raw_data.get("primary_location", {}) or {}
        source = primary_location.get("source", {}) or {}
        best_oa_location = raw_data.get("best_oa_location", {}) or {}
        authorships = raw_data.get("authorships", [])

        return {
            "id": raw_data.get("id"),
            "doi": raw_data.get("doi"),
            "title": raw_data.get("title"),
            "display_name": raw_data.get("display_name"),
            "publication_year": raw_data.get("publication_year"),
            "publication_date": raw_data.get("publication_date"),
            "type": raw_data.get("type"),
            "cited_by_count": raw_data.get("cited_by_count"),
            "is_retracted": raw_data.get("is_retracted"),
            "is_paratext": raw_data.get("is_paratext"),
            "cited_by_api_url": raw_data.get("cited_by_api_url"),
            "abstract_inverted_index": _clean_inverted_index(raw_data.get("abstract_inverted_index", {})),
            "source_id": source.get("id"),
            "source_display_name": source.get("display_name"),
            "topics": [topic.get("display_name") for topic in raw_data.get("topics", [])],
            "is_oa": best_oa_location.get("is_oa", False),
            "pdf_url": best_oa_location.get("pdf_url"),
            "landing_page_url": best_oa_location.get("landing_page_url"),
            "author_ids": [authorship.get("author", {}).get("id") for authorship in authorships],
        }

    @classmethod
    def pull(cls, doi: str) -> Self:
        """Pull a work from the OpenAlex by DOI."""
        response = httpx.get(f"https://api.openalex.org/works/doi:{doi}")
        response.raise_for_status()
        data = response.json()
        return cls(**cls.parse(data))

    @classmethod
    def from_raw(cls, raw_data: dict) -> Self:
        """Create a Work from raw data."""
        return cls(**cls.parse(raw_data))

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


# Resources are the base-level documents we embed and search.
ALL_RESOURCES = [Work]
ALL_RESOURCES_NAMES = [resource.__name__.lower() for resource in ALL_RESOURCES]
Resource = StrEnum("Resource", ALL_RESOURCES_NAMES)

# Clusters are the higher-level entities that group related resources.
ALL_CLUSTERS = [Person]
ALL_CLUSTERS_NAMES = [cluster.__name__.lower() for cluster in ALL_CLUSTERS]
Cluster = StrEnum("Cluster", ALL_CLUSTERS_NAMES)
