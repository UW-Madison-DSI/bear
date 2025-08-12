from typing import TypeVar

from .model import ALL_CLUSTERS, ALL_RESOURCES, CollectionProtocol

CollectionType = TypeVar("CollectionType", bound=CollectionProtocol)
__all__ = ["CollectionType", "CollectionProtocol", "ALL_RESOURCES", "ALL_CLUSTERS"]
