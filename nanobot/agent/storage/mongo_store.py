"""MongoDB-based storage backend for nanobot memory.

This module implements the MemoryStorageBackend interface
using MongoDB (with async support via motor).

Designed for use with MongoDB Atlas, including Vector Search support.
"""

import logging
from typing import Any, Dict, List, Optional

from pymongo import ASCENDING, DESCENDING

from nanobot.agent.storage.base import MemoryStorageBackend

logger = logging.getLogger(__name__)


class MongoDBMemoryStore(MemoryStorageBackend):
    """MongoDB-backed memory storage.

    Uses Motor (async MongoDB driver) for non-blocking operations.
    Supports MongoDB Atlas Vector Search for semantic memory.

    Collections used:
      - history         : conversation history entries
      - semantic_memory : semantic documents with vectors
      - counters        : auto-increment counters (for cursor)
    """

    def __init__(self, config: "MongoDBConfig"):
        """Initialize the MongoDB storage backend.

        Args:
            config: MongoDB configuration object with fields:
                   - uri: MongoDB connection URI
                   - database: database name
                   - history_collection: collection name for history
                   - semantic_collection: collection name for semantic memory
                   - vector_index: Atlas Vector Search index name
                   - fallback_to_file: whether to fallback to file store
        """
        from motor.motor_asyncio import AsyncIOMotorClient

        self.config = config
        self.client = AsyncIOMotorClient(config.uri)
        self.db = self.client[config.database]
        self.history = self.db[config.history_collection]
        self.semantic = self.db[config.semantic_collection]
        self.counters = self.db["counters"]
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize indexes for the collections.

        Creates necessary indexes if they don't exist.
        Note: Atlas Vector Search indexes must be created via Atlas UI.
        """
        if self._initialized:
            return

        try:
            # History collection indexes
            await self.history.create_index([("cursor", ASCENDING)], unique=True)
            await self.history.create_index([("timestamp", DESCENDING)])

            # Semantic collection indexes
            await self.semantic.create_index([("id", ASCENDING)], unique=True)
            await self.semantic.create_index([("timestamp", DESCENDING)])

            self._initialized = True
            logger.info("MongoDB storage initialized with indexes")

        except Exception as e:
            logger.error(f"Failed to initialize MongoDB indexes: {e}")
            raise

    async def close(self) -> None:
        """Close the MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")

    # ------------------------------------------------------------------
    # History methods
    # ------------------------------------------------------------------

    async def append_history(self, entry: Dict[str, Any]) -> int:
        """Append a history entry and return its cursor.

        Uses an atomic counter increment to generate unique cursors.

        Args:
            entry: The history entry to append.

        Returns:
            The assigned cursor value.
        """
        await self.initialize()

        # Atomically increment the counter
        result = await self.counters.find_one_and_update(
            {"_id": "history_cursor"},
            {"$inc": {"value": 1}},
            upsert=True,
            return_document=True
        )
        cursor = result["value"]
        entry["cursor"] = cursor

        # Insert the entry
        await self.history.insert_one(entry)
        logger.debug(f"Appended history entry with cursor {cursor}")

        return cursor

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
        await self.initialize()

        cursor = self.history.find(
            {"cursor": {"$gt": since_cursor}}
        ).sort("cursor", ASCENDING).limit(limit)

        entries = []
        async for doc in cursor:
            # Remove MongoDB _id field
            doc.pop("_id", None)
            entries.append(doc)

        return entries

    async def get_last_cursor(self) -> int:
        """Get the last used cursor value.

        Returns:
            The maximum cursor value, or 0 if no entries.
        """
        await self.initialize()

        last = await self.history.find_one(
            sort=[("cursor", DESCENDING)]
        )
        return last["cursor"] if last else 0

    async def set_last_cursor(self, cursor: int) -> None:
        """Persist the last processed cursor.

        For MongoDB, we store this in the counters collection.

        Args:
            cursor: The cursor value to persist.
        """
        await self.counters.update_one(
            {"_id": "last_processed_cursor"},
            {"$set": {"value": cursor}},
            upsert=True
        )

    async def get_last_processed_cursor(self) -> int:
        """Read the last processed cursor.

        Returns:
            The stored cursor value, or 0 if not found.
        """
        doc = await self.counters.find_one({"_id": "last_processed_cursor"})
        return doc["value"] if doc else 0

    async def set_dream_cursor(self, cursor: int) -> None:
        """Persist the dream processing cursor.

        Args:
            cursor: The cursor value to persist.
        """
        await self.counters.update_one(
            {"_id": "dream_cursor"},
            {"$set": {"value": cursor}},
            upsert=True
        )

    async def get_dream_cursor(self) -> int:
        """Read the dream cursor.

        Returns:
            The stored cursor value, or 0 if not found.
        """
        doc = await self.counters.find_one({"_id": "dream_cursor"})
        return doc["value"] if doc else 0

    # ------------------------------------------------------------------
    # Semantic memory methods
    # ------------------------------------------------------------------

    async def add_semantic_documents(
        self, docs: List[Dict[str, Any]]
    ) -> List[str]:
        """Add documents to the semantic memory store.

        Args:
            docs: List of document dictionaries. Each should have
                  at least a 'vector' (List[float]) field.

        Returns:
            List of document IDs that were inserted.
        """
        await self.initialize()

        # Generate IDs if not present
        next_id = await self._get_next_semantic_id()
        ids = []

        for doc in docs:
            doc["id"] = str(next_id)
            ids.append(str(next_id))
            next_id += 1

        # Insert documents
        await self.semantic.insert_many(docs)
        logger.debug(f"Added {len(docs)} semantic documents")

        return ids

    async def query_semantic(
        self, query_vector: List[float], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Query semantic memory using vector similarity.

        Uses MongoDB Atlas Vector Search if available, otherwise
        falls back to in-memory cosine similarity.

        Args:
            query_vector: The query embedding vector.
            top_k: Maximum number of results.

        Returns:
            List of documents sorted by similarity.
        """
        await self.initialize()

        # Try Atlas Vector Search first
        try:
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": self.config.vector_index,
                        "path": "vector",
                        "queryVector": query_vector,
                        "numCandidates": top_k * 10,
                        "limit": top_k
                    }
                },
                {
                    "$project": {
                        "vector": 0,  # Exclude vector from results
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]

            results = []
            async for doc in self.semantic.aggregate(pipeline):
                doc.pop("_id", None)
                results.append(doc)

            return results

        except Exception as e:
            logger.warning(
                f"Atlas Vector Search failed (might not be configured): {e}\n"
                "Falling back to in-memory cosine similarity."
            )
            return await self._query_semantic_fallback(query_vector, top_k)

    async def delete_semantic_documents(
        self, doc_ids: List[str]
    ) -> int:
        """Delete documents from the semantic memory store.

        Args:
            doc_ids: List of document IDs to delete.

        Returns:
            Number of documents deleted.
        """
        await self.initialize()

        result = await self.semantic.delete_many(
            {"id": {"$in": doc_ids}}
        )
        logger.debug(f"Deleted {result.deleted_count} semantic documents")
        return result.deleted_count

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    async def _get_next_semantic_id(self) -> int:
        """Get the next available semantic document ID.

        Returns:
            The next ID to use.
        """
        last = await self.semantic.find_one(
            sort=[("id", DESCENDING)]
        )
        if last and last.get("id"):
            try:
                return int(last["id"]) + 1
            except ValueError:
                pass
        return 1

    async def _query_semantic_fallback(
        self, query_vector: List[float], top_k: int
    ) -> List[Dict[str, Any]]:
        """Fallback semantic query using in-memory cosine similarity.

        Args:
            query_vector: The query embedding vector.
            top_k: Maximum number of results.

        Returns:
            List of documents sorted by cosine similarity.
        """
        import math

        # Fetch all documents (warning: may be slow for large collections)
        docs = []
        async for doc in self.semantic.find({}):
            doc.pop("_id", None)
            docs.append(doc)

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
