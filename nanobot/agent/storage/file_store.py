"""File-based storage backend for nanobot memory.

This module implements the MemoryStorageBackend interface
using the existing file-based approach (JSONL + cursor files).
"""

import json
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from nanobot.agent.storage.base import MemoryStorageBackend


class FileMemoryStore(MemoryStorageBackend):
    """File-based memory storage using JSONL format.

    This is the traditional nanobot storage backend,
    now refactored to implement the unified interface.

    Files used:
      - memory/history.jsonl  : append-only history entries
      - memory/.cursor         : last consolidator cursor
      - memory/.dream_cursor   : last dream cursor
      - memory/semantic_index.jsonl : semantic memory documents
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.memory_dir = workspace / "memory"
        self.history_file = self.memory_dir / "history.jsonl"
        self.cursor_file = self.memory_dir / ".cursor"
        self.dream_cursor_file = self.memory_dir / ".dream_cursor"
        self.semantic_file = self.memory_dir / "semantic_index.jsonl"

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Ensure the memory directory exists."""
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    async def close(self) -> None:
        """No-op for file storage."""
        pass

    # ------------------------------------------------------------------
    # History methods
    # ------------------------------------------------------------------

    async def append_history(self, entry: Dict[str, Any]) -> int:
        """Append a history entry to history.jsonl.

        Assigns a cursor value automatically.

        Args:
            entry: The history entry to append.

        Returns:
            The assigned cursor value.
        """
        await self.initialize()

        # Read existing entries to determine next cursor
        entries = await self._read_all_history()
        next_cursor = entries[-1]["cursor"] + 1 if entries else 1
        entry["cursor"] = next_cursor

        # Append to file
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        return next_cursor

    async def read_history(
        self, since_cursor: int = 0, limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Read history entries since a given cursor.

        Args:
            since_cursor: Return entries with cursor > since_cursor.
            limit: Maximum number of entries to return.

        Returns:
            List of history entries ordered by cursor.
        """
        entries = await self._read_all_history()
        filtered = [e for e in entries if e.get("cursor", 0) > since_cursor]
        return filtered[:limit]

    async def get_last_cursor(self) -> int:
        """Get the last cursor value from history.

        Returns:
            The maximum cursor value, or 0 if no entries.
        """
        entries = await self._read_all_history()
        if not entries:
            return 0
        return max(e.get("cursor", 0) for e in entries)

    async def set_last_cursor(self, cursor: int) -> None:
        """Persist the last processed cursor.

        For file storage, we write to .cursor file.

        Args:
            cursor: The cursor value to persist.
        """
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        with open(self.cursor_file, "w", encoding="utf-8") as f:
            f.write(str(cursor))

    async def get_last_processed_cursor(self) -> int:
        """Read the last processed cursor from .cursor file.

        Returns:
            The stored cursor value, or 0 if not found.
        """
        if not self.cursor_file.exists():
            return 0
        with open(self.cursor_file, "r", encoding="utf-8") as f:
            try:
                return int(f.read().strip())
            except ValueError:
                return 0

    async def set_dream_cursor(self, cursor: int) -> None:
        """Persist the dream processing cursor.

        Args:
            cursor: The cursor value to persist.
        """
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        with open(self.dream_cursor_file, "w", encoding="utf-8") as f:
            f.write(str(cursor))

    async def get_dream_cursor(self) -> int:
        """Read the dream cursor from .dream_cursor file.

        Returns:
            The stored cursor value, or 0 if not found.
        """
        if not self.dream_cursor_file.exists():
            return 0
        with open(self.dream_cursor_file, "r", encoding="utf-8") as f:
            try:
                return int(f.read().strip())
            except ValueError:
                return 0

    # ------------------------------------------------------------------
    # Semantic memory methods
    # ------------------------------------------------------------------

    async def add_semantic_documents(
        self, docs: List[Dict[str, Any]]
    ) -> List[str]:
        """Add documents to semantic_index.jsonl.

        Args:
            docs: List of document dictionaries.

        Returns:
            List of document IDs (generated as sequential integers).
        """
        await self.initialize()

        # Read existing to determine next ID
        existing = await self._read_all_semantic()
        next_id = max((int(d.get("id", 0)) for d in existing), default=0) + 1

        ids = []
        with open(self.semantic_file, "a", encoding="utf-8") as f:
            for doc in docs:
                doc["id"] = str(next_id)
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                ids.append(str(next_id))
                next_id += 1

        return ids

    async def query_semantic(
        self, query_vector: List[float], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Query semantic memory using cosine similarity.

        Args:
            query_vector: The query embedding vector.
            top_k: Maximum number of results.

        Returns:
            List of documents sorted by similarity (highest first).
        """
        import math

        docs = await self._read_all_semantic()

        # Compute cosine similarity
        results = []
        for doc in docs:
            vector = doc.get("vector", [])
            if not vector:
                continue
            score = self._cosine_similarity(query_vector, vector)
            results.append({**doc, "_score": score})

        # Sort by score descending
        results.sort(key=lambda x: x["_score"], reverse=True)
        return results[:top_k]

    async def delete_semantic_documents(
        self, doc_ids: List[str]
    ) -> int:
        """Delete documents from semantic_index.jsonl.

        Args:
            doc_ids: List of document IDs to delete.

        Returns:
            Number of documents deleted.
        """
        docs = await self._read_all_semantic()
        id_set = set(doc_ids)

        new_docs = [d for d in docs if str(d.get("id", "")) not in id_set]
        deleted_count = len(docs) - len(new_docs)

        # Rewrite file
        with open(self.semantic_file, "w", encoding="utf-8") as f:
            for doc in new_docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")

        return deleted_count

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    async def _read_all_history(self) -> List[Dict[str, Any]]:
        """Read all history entries from history.jsonl.

        Returns:
            List of history entry dictionaries.
        """
        if not self.history_file.exists():
            return []

        entries = []
        with open(self.history_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return entries

    async def _read_all_semantic(self) -> List[Dict[str, Any]]:
        """Read all semantic documents from semantic_index.jsonl.

        Returns:
            List of semantic document dictionaries.
        """
        if not self.semantic_file.exists():
            return []

        docs = []
        with open(self.semantic_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        docs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return docs

    @staticmethod
    def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            v1: First vector.
            v2: Second vector.

        Returns:
            Cosine similarity score (0.0 to 1.0).
        """
        import math

        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))

        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)
