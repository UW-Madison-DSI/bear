from typing import Any

import requests
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Index
from sqlalchemy.orm import Session
from sqlalchemy.sql import text
from sqlmodel import Field, SQLModel, create_engine

from openalex_search.common import CONFIG


class Work(SQLModel, table=True):
    id: str = Field(primary_key=True)
    doi: str
    title: str
    display_name: str
    journal: str | None  # TODO: Check if all articles has journal.
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

    @property
    def text(self) -> str:
        """Return a string representation of the work."""
        if not self.journal:
            return f"title: {self.title}"
        else:
            return f"title: {self.title} \njournal:{self.journal}"


# create a session
engine = create_engine("postgresql://postgres:postgres@localhost/dev")


def init(wipe: bool = False) -> None:
    """Initialize the database."""

    with Session(engine) as session:
        # create the PG vector extension
        session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        session.commit()

        # create the table
        if wipe:
            session.execute(text("DROP TABLE IF EXISTS work"))
            session.commit()
        SQLModel.metadata.create_all(engine)

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
        index.create(bind=engine)


def push_works(works: list[Work]) -> None:
    """Push works to the database."""
    with Session(engine) as session:
        for work in works:
            session.add(work)
        session.commit()
