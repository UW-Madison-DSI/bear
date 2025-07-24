from typing import TypeVar

from .model import ALL_RESOURCES, Resource

ResourceType = TypeVar("ResourceType", bound=Resource)

__all__ = ["ResourceType", "Resource", "ALL_RESOURCES"]
