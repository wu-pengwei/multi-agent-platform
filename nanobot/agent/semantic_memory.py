from __future__ import annotations

import asyncio
import inspect
import json
import math
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional

from loguru import logger

from nanobot.agent.storage.base import MemoryStorageBackend


class SemanticMemory:
    """Pluggable semantic memory with pluggable backends.

    Supports both file-based (JSONL) and MongoDB backends via the
    `storage_backend` parameter.
    """

    def __init__(
        self,
        path: Path,
        embed_func: Callable[[List[str]], List[List[float]]],
        dim: int | None = None,
        storage_backend: Optional[MemoryStorageBackend] = None,
    ):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embed_func = embed_func
        self.dim = dim
        self._index: List[dict[str, Any]] = []
        self._storage = storage_backend
        self._use_storage = storage_backend is not None

        # Load index only if using file-based storage
        if not self._use_storage:
            self._load_index()

    def _load_index(self) -> None:
        if not self.path.exists():
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    # ensure vector is list[float]
                    if "vector" in obj and isinstance(obj["vector"], list):
                        self._index.append(obj)
        except Exception:
            logger.exception("Failed to load semantic index {}", self.path)

    def _append_entry(self, entry: dict[str, Any]) -> None:
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._index.append(entry)
        except Exception:
            logger.exception("Failed to append semantic index entry")

    async def add_documents(self, docs: Iterable[dict[str, Any]]) -> List[str]:
        """Add documents to the vector store.

        Each doc should be a dict with at least a `text` key. Optional keys:
        `id`, `metadata`.

        Returns list of ids added.
        """
        docs = list(docs)
        texts = [d.get("text", "") for d in docs]
        if not texts:
            return []
        try:
            maybe = self.embed_func(texts)
            # support async embed functions
            if hasattr(maybe, "__await__"):
                vectors = await maybe
            else:
                vectors = maybe
        except Exception:
            logger.exception("Embedding call failed")
            return []

        ids: List[str] = []

        # Use storage backend if available
        if self._use_storage and self._storage:
            entries = []
            for d, vec in zip(docs, vectors):
                _id = d.get("id") or uuid.uuid4().hex
                entry = {
                    "id": _id,
                    "text": d.get("text", ""),
                    "metadata": d.get("metadata", {}),
                    "vector": vec,
                }
                if self.dim is None:
                    self.dim = len(vec) if isinstance(vec, list) else None
                entries.append(entry)
                ids.append(_id)

            # Batch insert to storage backend
            result_ids = await self._storage.add_semantic_documents(entries)
            return result_ids

        # Fallback to file-based storage
        for d, vec in zip(docs, vectors):
            _id = d.get("id") or uuid.uuid4().hex
            entry = {
                "id": _id,
                "text": d.get("text", ""),
                "metadata": d.get("metadata", {}),
                "vector": vec,
            }
            if self.dim is None:
                self.dim = len(vec) if isinstance(vec, list) else None
            self._append_entry(entry)
            ids.append(_id)
        return ids

    @staticmethod
    def _dot(a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    @staticmethod
    def _norm(a: List[float]) -> float:
        return math.sqrt(sum(x * x for x in a))

    def _cosine(self, a: List[float], b: List[float]) -> float:
        na = self._norm(a)
        nb = self._norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return self._dot(a, b) / (na * nb)

    async def query(self, query_text: str, top_k: int = 5) -> List[dict[str, Any]]:
        """Embed the query and return top_k nearest entries (sorted by score)."""
        try:
            maybe = self.embed_func([query_text])
            if hasattr(maybe, "__await__"):
                qv = (await maybe)[0]
            else:
                qv = maybe[0]
        except Exception:
            logger.exception("Embedding failed for query")
            return []

        # Use storage backend if available
        if self._use_storage and self._storage:
            try:
                results = await self._storage.query_semantic(qv, top_k=top_k)
                return results
            except Exception as e:
                logger.warning(f"Storage backend query failed, falling back to in-memory: {e}")

        # Fallback to in-memory cosine similarity
        scores = []
        for entry in self._index:
            vec = entry.get("vector")
            if not isinstance(vec, list):
                continue
            score = self._cosine(qv, vec)
            scores.append((score, entry))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [e for s, e in scores[:top_k]]

    # -- Sync variants (for sync embed funcs, e.g. local sentence-transformers) --

    def _embed_sync(self, texts: List[str]) -> List[List[float]] | None:
        """Run the embed function synchronously. Returns None for async-only embedders."""
        maybe = self.embed_func(texts)
        if inspect.isawaitable(maybe):
            # Close the coroutine because the synchronous caller cannot await it.
            close = getattr(maybe, "close", None)
            if close is not None:
                close()
            logger.warning(
                "Semantic memory: embed function is async; sync indexing/recall skipped"
            )
            return None
        return maybe

    def add_documents_sync(self, docs: Iterable[dict[str, Any]]) -> List[str]:
        """Synchronous add_documents for sync embed funcs.

        Only supports the file-based store; storage backends are async-only
        and are skipped with a debug log.
        """
        docs = list(docs)
        texts = [d.get("text", "") for d in docs]
        if not texts:
            return []
        try:
            vectors = self._embed_sync(texts)
        except Exception:
            logger.exception("Embedding call failed (sync path)")
            return []
        if vectors is None:
            return []
        if self._use_storage and self._storage:
            logger.debug(
                "Semantic memory: storage backend configured; sync indexing skipped"
            )
            return []

        ids: List[str] = []
        for d, vec in zip(docs, vectors):
            _id = d.get("id") or uuid.uuid4().hex
            entry = {
                "id": _id,
                "text": d.get("text", ""),
                "metadata": d.get("metadata", {}),
                "vector": vec,
            }
            if self.dim is None:
                self.dim = len(vec) if isinstance(vec, list) else None
            self._append_entry(entry)
            ids.append(_id)
        return ids

    def query_sync(self, query_text: str, top_k: int = 5) -> List[dict[str, Any]]:
        """Synchronous query for sync embed funcs. Returns [] when unavailable."""
        try:
            vectors = self._embed_sync([query_text])
        except Exception:
            logger.exception("Embedding failed for query (sync path)")
            return []
        if vectors is None:
            return []
        if self._use_storage and self._storage:
            logger.debug(
                "Semantic memory: storage backend configured; sync query skipped"
            )
            return []
        qv = vectors[0]

        scores = []
        for entry in self._index:
            vec = entry.get("vector")
            if not isinstance(vec, list):
                continue
            score = self._cosine(qv, vec)
            scores.append((score, entry))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [e for s, e in scores[:top_k]]

    def clear(self) -> None:
        """Clear all documents from the semantic store."""
        # Use storage backend if available
        if self._use_storage and self._storage:
            try:
                asyncio.get_event_loop().run_until_complete(
                    self._storage.delete_semantic_documents([])  # Delete all
                )
                logger.info("Cleared semantic store via storage backend")
            except Exception:
                logger.exception("Failed to clear semantic store via storage backend")
            return

        # Fallback to file-based storage
        try:
            self.path.unlink(missing_ok=True)
        except Exception:
            logger.exception("Failed to clear semantic store file")
        self._index = []
