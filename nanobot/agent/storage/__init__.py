"""Storage module for nanobot memory backends.

This module provides a unified interface for memory storage,
supporting both file-based and MongoDB-based backends.
"""

from nanobot.agent.storage.base import MemoryStorageBackend
from nanobot.agent.storage.file_store import FileMemoryStore

__all__ = ["MemoryStorageBackend", "FileMemoryStore"]
