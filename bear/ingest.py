from pathlib import Path

import pandas as pd

from bear.config import logger
from bear.db import push
from bear.embedding import embed
from bear.model import Work


def ingest(path: Path) -> None:
    """Ingest staging file into Milvus."""

    logger.info(f"Loading data from {path}")
    df = pd.read_parquet(path)

    logger.info(f"Data loaded with {len(df)} rows.")
    works = [Work.from_raw(row.to_dict()) for _, row in df.iterrows()]

    works = embed(works)
    push(works)
    logger.info(f"Ingested {len(works)} works from {path} into Milvus.")


if __name__ == "__main__":
    STAGING_DIR = Path("tmp/openalex_data/works")
    for file in STAGING_DIR.glob("*.parquet"):
        ingest(file)
    logger.info(f"Ingestion: {STAGING_DIR} complete.")
