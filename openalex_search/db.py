from datetime import datetime
from typing import Any, TypeVar

import requests
from pgvector.sqlalchemy import Vector
from sqlmodel import (
    Column,
    Field,
    Index,
    Session,
    SQLModel,
    create_engine,
    select,
    text,
)

from openalex_search.common import CONFIG

ENGINE = create_engine("postgresql://postgres:postgres@localhost/dev")


def recover_abstract(abstract_inverted_index: dict[str, list[int]]) -> str:
    """Recover the abstract from the inverted index."""
    word_positions = []
    for word, positions in abstract_inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))

    # Sort by positions
    word_positions.sort(key=lambda x: x[0])
    abstract = " ".join(word for _, word in word_positions)
    return abstract


class Work(SQLModel, table=True):
    id: str = Field(primary_key=True)
    doi: str | None = None
    title: str | None = None
    display_name: str | None = None
    journal: str | None = None
    publication_year: int
    publication_date: str
    type: str
    cited_by_count: int
    is_retracted: bool
    is_paratext: bool
    is_oa: bool | None
    pdf_url: str | None
    landing_page_url: str | None
    abstract: str | None = Field(default=None)
    embedding: Any = Field(
        default=None, sa_column=Column(Vector(CONFIG.EMBEDDING_DIMS))
    )

    @staticmethod
    def parse(data: dict) -> dict:
        """Parse a work from OpenAlex raw data to local Work format."""

        if data["best_oa_location"] is None:
            data["best_oa_location"] = {
                "is_oa": None,
                "pdf_url": None,
                "landing_page_url": None,
            }  # Best OA location has quite a few missing values

        try:
            journal = data["primary_location"]["source"]["display_name"]
        except (KeyError, TypeError):
            journal = None

        abstract = (
            recover_abstract(data["abstract_inverted_index"])
            if data.get("abstract_inverted_index")
            else None
        )

        return dict(
            id=data["id"],
            doi=data["doi"],
            title=data["title"],
            display_name=data["display_name"],
            journal=journal,
            publication_year=data["publication_year"],
            publication_date=data["publication_date"],
            type=data["type"],
            cited_by_count=data["cited_by_count"],
            is_retracted=data["is_retracted"],
            is_paratext=data["is_paratext"],
            is_oa=data["best_oa_location"]["is_oa"],
            pdf_url=data["best_oa_location"]["pdf_url"],
            landing_page_url=data["best_oa_location"]["landing_page_url"],
            abstract=abstract,
        )

    @classmethod
    def pull(cls, doi: str) -> "Work":
        """Pull a work from the OpenAlex by DOI."""
        response = requests.get(f"https://api.openalex.org/works/doi:{doi}")
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
        if self.abstract:
            text += f"\nabstract: {self.abstract}"
        return text


class WorkAuthorship(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    work_id: str | None = Field(default=None, foreign_key="work.id")
    author_position: str
    author_id: str | None = Field(default=None, foreign_key="author.id")
    institution_id: str


class Author(SQLModel, table=True):
    id: str = Field(primary_key=True)
    orcid: str | None = None
    display_name: str | None = None
    display_name_alternatives: str | None = None
    works_count: int = Field(default=0)
    cited_by_count: int = Field(default=0)
    last_known_institution: str | None = None
    works_api_url: str | None = None
    updated_date: datetime = Field(default_factory=datetime.now)


def init(wipe: bool = False) -> None:
    """Initialize the database."""

    with Session(ENGINE) as session:
        # create the PG vector extension
        session.connection().execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        session.commit()

        # create the table
        if wipe:
            session.connection().execute(text("DROP TABLE IF EXISTS work"))
            session.commit()
        SQLModel.metadata.create_all(ENGINE)

        # create the index
        index = Index(
            "work_index",
            Work.embedding,
            postgresql_using="hnsw",
            postgresql_with={
                "m": CONFIG.HNSW_M,
                "ef_construction": CONFIG.HNSW_EF_CONSTRUCTION,
            },
            postgresql_ops={"embedding": "vector_l2_ops"},
        )
        index.create(bind=ENGINE, checkfirst=True)


Record = TypeVar("Record", Work, WorkAuthorship, Author)


def push(records: list[Record]) -> None:
    """Push works to the database."""
    with Session(ENGINE) as session:
        for record in records:
            session.merge(record)
        session.commit()


def get_author(id: int) -> Author:
    """Get an author by id."""

    with Session(ENGINE) as session:
        if author := session.exec(select(Author).where(Author.id == id)).first():
            return author
        raise ValueError(f"Author with id {id} not found")
