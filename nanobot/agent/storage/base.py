"""Base classes for memory storage backends.

This module defines the abstract interface that all storage
backends must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class MemoryStorageBackend(ABC):
    """Abstract base class for memory storage backends.

    All storage backends (file-based, MongoDB, etc.) must
    implement this interface to ensure consistent behavior.
    """

    # ------------------------------------------------------------------
    # History storage methods
    # ------------------------------------------------------------------

    @abstractmethod
    async def append_history(self, entry: Dict[str, Any]) -> int:
        """Append a history entry and return its cursor position.

        Args:
            entry: A dictionary containing the history entry.
                   Should include at least a 'content' field.

        Returns:
            The cursor value assigned to this entry (int).
        """
        ...

    @abstractmethod
    async def read_history(
        self, since_cursor: int = 0, limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Read history entries starting from a given cursor.

        Args:
            since_cursor: Return entries with cursor > since_cursor.
            limit: Maximum number of entries to return.

        Returns:
            A list of history entry dictionaries, ordered by cursor.
        """
        ...

    @abstractmethod
    async def get_last_cursor(self) -> int:
        """Return the last used cursor value (0 if no entries).

        Returns:
            The maximum cursor value in the history store, or 0.
        """
        ...

    @abstractmethod
    async def set_last_cursor(self, cursor: int) -> None:
        """Persist the last processed cursor value.

        This is used by the Consolidator to track which history
        entries have already been processed.

        Args:
            cursor: The cursor value to persist.
        """
        ...

    # ------------------------------------------------------------------
    # Semantic memory methods
    # ------------------------------------------------------------------

    @abstractmethod
    async def add_semantic_documents(
        self, docs: List[Dict[str, Any]]
    ) -> List[str]:
        """Add documents to the semantic memory store.

        Args:
            docs: A list of document dictionaries. Each document
                  should have at least a 'vector' (List[float]) and
                  'text' (str) field.

        Returns:
            A list of document IDs that were inserted.
        """
        ...

    @abstractmethod
    async def query_semantic(
        self, query_vector: List[float], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Query the semantic memory store by vector similarity.

        Args:
            query_vector: The query embedding vector.
            top_k: Maximum number of results to return.

        Returns:
            A list of document dictionaries, sorted by similarity
            (most similar first). Each document includes a
            '_score' field with the similarity score.
        """
        ...

    @abstractmethod
    async def delete_semantic_documents(
        self, doc_ids: List[str]
    ) -> int:
        """Delete documents from the semantic memory store.

        Args:
            doc_ids: A list of document IDs to delete.

        Returns:
            The number of documents deleted.
        """
        ...

    # ------------------------------------------------------------------
    # Lifecycle methods
    # ------------------------------------------------------------------

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the storage backend (create indexes, etc.).

        Called once when the backend is first used.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the storage backend and release resources.

        Called when shutting down gracefully.
        """
        ...
