from typing import Any

import httpx
from pydantic import BaseModel, Field


def clean_inverted_index(inverted_index: dict[str, Any]) -> dict[str, list[int]]:
    """Cleans the abstract inverted index by converting values to lists of int."""
    return {k: list(map(int, v)) for k, v in inverted_index.items() if v is not None}


def recover_abstract(inverted_index: dict[str, list[int]]) -> str:
    """Recover the abstract from the inverted index."""
    if not inverted_index:
        return ""

    word_positions = [(pos, word) for word, positions in inverted_index.items() if positions for pos in positions]

    # Sort by position (the first element of the tuple)
    word_positions.sort()

    return " ".join(word for _, word in word_positions)


class Work(BaseModel):
    """Work model for Milvus storage."""

    # OA Works fields
    id: str
    doi: str
    title: str
    display_name: str
    publication_year: int
    publication_date: str
    type: str
    cited_by_count: int
    is_retracted: bool | None = None
    is_paratext: bool | None = None
    cited_by_api_url: str
    abstract_inverted_index: dict[str, list[int]] = Field(default_factory=dict)

    # Additional field via default works API
    source_id: str
    source_display_name: str
    topics: list[str] = Field(default_factory=list)
    is_oa: bool | None = None
    pdf_url: str | None = None
    landing_page_url: str | None = None

    embedding: list[float] = Field(default_factory=list)

    @property
    def abstract(self) -> str:
        """Recover the abstract from the inverted index."""
        return recover_abstract(self.abstract_inverted_index)

    @property
    def journal(self) -> str:
        """Return the journal name."""
        return self.source_display_name

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
            "abstract_inverted_index": clean_inverted_index(data.get("abstract_inverted_index") or {}),
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
        if self.journal:
            text += f"\njournal:{self.journal}"
        if self.topics:
            text += f"\ntopics: {', '.join(self.topics)}"
        if self.abstract:
            text += f"\nabstract: {self.abstract}"
        return text

    def to_milvus_dict(self) -> dict:
        """Convert to dictionary for Milvus insertion."""
        return {
            "id": self.id,
            "doi": self.doi or "",
            "title": self.title or "",
            "display_name": self.display_name or "",
            "journal": self.journal or "",
            "publication_year": self.publication_year,
            "publication_date": self.publication_date,
            "type": self.type,
            "cited_by_count": self.cited_by_count,
            "is_retracted": self.is_retracted,
            "is_paratext": self.is_paratext,
            "is_oa": self.is_oa if self.is_oa is not None else False,
            "pdf_url": self.pdf_url or "",
            "landing_page_url": self.landing_page_url or "",
            "abstract": self.abstract or "",
            "embedding": self.embedding or [],
        }
