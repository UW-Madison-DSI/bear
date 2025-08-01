from typing import TypeVar

from .model import ALL_RESOURCES, ResourceProtocol

ResourceType = TypeVar("ResourceType", bound=ResourceProtocol)

__all__ = ["ResourceType", "ResourceProtocol", "ALL_RESOURCES"]
