import logging
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Configuration for the openalex-search package."""

    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    EMBEDDING_DIMS: int = 1536
    HNSW_M: int = 32
    HNSW_EF_CONSTRUCTION: int = 512


def make_logger() -> logging.Logger:
    """Make a logger for the openalex-search package."""

    logger = logging.getLogger(name="openalex-search")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


LOGGER = make_logger()
CONFIG = Config()
