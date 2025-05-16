from pathlib import Path

import pandas as pd

from openalex_search.settings import LOGGER
from openalex_search.db import Author, Work, WorkAuthorship, init, push
from openalex_search.embedding import embed_works


def unroll(row: pd.Series) -> tuple[list[WorkAuthorship], list[Author]]:
    """Unroll a row into a list of WorkAuthorship and Author objects."""

    work_authorships, authors = [], []
    for authorship in row.authorships:
        if authorship["author"]["id"] in [author.id for author in authors]:
            # Skip authors that have already been added
            continue

        author = authorship["author"]
        authors.append(
            Author(
                id=author["id"],
                orcid=author["orcid"],
                display_name=author["display_name"],
            )
        )
        for institution in authorship["institutions"]:
            work_authorships.append(
                WorkAuthorship(
                    work_id=row.id,
                    author_position=authorship["author_position"],
                    author_id=authorship["author"]["id"],
                    institution_id=institution["id"],
                )
            )
    return work_authorships, authors


def ingest(path: Path) -> None:
    """Ingest staging file into DB."""

    # Load the data
    LOGGER.info(f"Loading data from {path}")
    df = pd.read_parquet(path)
    works = [Work.from_raw(row.to_dict()) for _, row in df.iterrows()]

    LOGGER.info(f"Embedding {len(works)} works...")
    works = embed_works(works)  # TODO: Deduplicate, this is an expansive operation

    # Insert Works
    LOGGER.info("Inserting works...")
    push(works)

    # Insert WorkAuthorships and Authors
    # Unroll the data (There's no need to use the call author route to retrieve the entire object for this experiment; missing author columns are not a concern at this stage.)
    LOGGER.info("Inserting authorships and authors...")
    work_authorships = []
    authors = []
    for _, row in df.iterrows():
        wa, a = unroll(row)
        work_authorships.extend(wa)
        authors.extend(a)

    # Push to the DB
    push(authors)  # This must be done first to avoid foreign key constraint violations
    push(work_authorships)
    LOGGER.info(
        f"Data ingested into DB: {len(works)} works, {len(authors)} authors, {len(work_authorships)} authorships"
    )


if __name__ == "__main__":
    init()
    STAGING_DIR = Path("local_data")
    for file in STAGING_DIR.glob("*.parquet"):
        ingest(file)
        LOGGER.info(f"File {file} ingested.")
