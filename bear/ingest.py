from pathlib import Path

import pandas as pd

from bear.settings import LOGGER
from bear.db import Work, init, push
from bear.embedding import embed_works


def ingest(path: Path) -> None:
    """Ingest staging file into Milvus."""

    # Load the data
    LOGGER.info(f"Loading data from {path}")
    df = pd.read_parquet(path)
    works = [Work.from_raw(row.to_dict()) for _, row in df.iterrows()]

    LOGGER.info(f"Embedding {len(works)} works...")
    works = embed_works(works)

    # Insert Works into Milvus
    LOGGER.info("Inserting works into Milvus...")
    push(works)

    LOGGER.info(f"Data ingested into Milvus: {len(works)} works")


if __name__ == "__main__":
    init()
    STAGING_DIR = Path("tmp")
    for file in STAGING_DIR.glob("*.parquet"):
        ingest(file)
        LOGGER.info(f"File {file} ingested.")