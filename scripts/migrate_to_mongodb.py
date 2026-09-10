"""Migrate existing file-based memory to MongoDB.

Usage::

    # Install dependencies first
    pip install nanobot-ai[mongodb]

    # Run migration
    python scripts/migrate_to_mongodb.py --workspace ~/.nanobot/workspace

This script migrates:
- memory/history.jsonl → MongoDB history collection
- memory/semantic_index.jsonl → MongoDB semantic_memory collection
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

try:
    from loguru import logger

    from nanobot.agent.storage.mongo_store import MongoDBMemoryStore
    from nanobot.config.loader import load_config
    from nanobot.config.schema import MongoDBConfig
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install nanobot with mongodb support: pip install nanobot-ai[mongodb]")
    sys.exit(1)


async def migrate_history(memory_dir: Path, store: MongoDBMemoryStore) -> int:
    """Migrate history.jsonl to MongoDB.

    Args:
        memory_dir: Path to the memory directory.
        store: MongoDB memory store instance.

    Returns:
        Number of entries migrated.
    """
    history_file = memory_dir / "history.jsonl"
    if not history_file.exists():
        logger.warning(f"History file not found: {history_file}")
        return 0

    entries = []
    with open(history_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON line: {e}")
                continue

    if not entries:
        logger.info("No history entries to migrate")
        return 0

    # Insert entries (MongoDB will assign _id automatically)
    await store.history.insert_many(entries)
    logger.info(f"Migrated {len(entries)} history entries to MongoDB")

    return len(entries)


async def migrate_semantic_memory(memory_dir: Path, store: MongoDBMemoryStore) -> int:
    """Migrate semantic_index.jsonl to MongoDB.

    Args:
        memory_dir: Path to the memory directory.
        store: MongoDB memory store instance.

    Returns:
        Number of documents migrated.
    """
    semantic_file = memory_dir / "semantic_index.jsonl"
    if not semantic_file.exists():
        logger.warning(f"Semantic index file not found: {semantic_file}")
        return 0

    docs = []
    with open(semantic_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
                docs.append(doc)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON line: {e}")
                continue

    if not docs:
        logger.info("No semantic documents to migrate")
        return 0

    # Insert documents
    await store.semantic.insert_many(docs)
    logger.info(f"Migrated {len(docs)} semantic documents to MongoDB")

    return len(docs)


async def main():
    parser = argparse.ArgumentParser(description="Migrate nanobot memory to MongoDB")
    parser.add_argument(
        "--workspace",
        type=str,
        default="~/.nanobot/workspace",
        help="Path to nanobot workspace (default: ~/.nanobot/workspace)",
    )
    parser.add_argument(
        "--uri",
        type=str,
        default=None,
        help="MongoDB URI (overrides config)",
    )
    parser.add_argument(
        "--database",
        type=str,
        default=None,
        help="MongoDB database name (overrides config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be migrated without actually migrating",
    )
    args = parser.parse_args()

    # Resolve workspace path
    workspace = Path(args.workspace).expanduser().resolve()
    if not workspace.exists():
        logger.error(f"Workspace not found: {workspace}")
        sys.exit(1)

    # Load config
    config = load_config(workspace)

    # Build MongoDB config
    mongo_config = config.mongodb
    if args.uri:
        mongo_config.uri = args.uri
    if args.database:
        mongo_config.database = args.database

    if not mongo_config.enabled and not args.uri:
        logger.warning("MongoDB is not enabled in config. Use --uri to specify MongoDB URI.")
        mongo_config.enabled = True  # Enable for migration

    # Create MongoDB store
    store = MongoDBMemoryStore(mongo_config)
    await store.initialize()

    try:
        memory_dir = workspace / "memory"
        if not memory_dir.exists():
            logger.error(f"Memory directory not found: {memory_dir}")
            sys.exit(1)

        # Migrate history
        logger.info("=" * 60)
        logger.info("Migrating history...")
        logger.info("=" * 60)
        if args.dry_run:
            history_file = memory_dir / "history.jsonl"
            if history_file.exists():
                with open(history_file, "r", encoding="utf-8") as f:
                    count = sum(1 for line in f if line.strip())
                logger.info(f"[DRY RUN] Would migrate {count} history entries")
            else:
                logger.info("[DRY RUN] No history file found")
        else:
            history_count = await migrate_history(memory_dir, store)
            logger.info(f"History migration complete: {history_count} entries")

        # Migrate semantic memory
        logger.info("=" * 60)
        logger.info("Migrating semantic memory...")
        logger.info("=" * 60)
        if args.dry_run:
            semantic_file = memory_dir / "semantic_index.jsonl"
            if semantic_file.exists():
                with open(semantic_file, "r", encoding="utf-8") as f:
                    count = sum(1 for line in f if line.strip())
                logger.info(f"[DRY RUN] Would migrate {count} semantic documents")
            else:
                logger.info("[DRY RUN] No semantic index file found")
        else:
            semantic_count = await migrate_semantic_memory(memory_dir, store)
            logger.info(f"Semantic memory migration complete: {semantic_count} documents")

        if args.dry_run:
            logger.info("=" * 60)
            logger.info("DRY RUN COMPLETE - no data was migrated")
            logger.info("=" * 60)
        else:
            logger.info("=" * 60)
            logger.info("MIGRATION COMPLETE")
            logger.info("=" * 60)
            logger.info("")
            logger.info("Next steps:")
            logger.info("1. Update your config to enable MongoDB:")
            logger.info("   mongodb:")
            logger.info("     enabled: true")
            logger.info(f"     uri: {mongo_config.uri}")
            logger.info(f"     database: {mongo_config.database}")
            logger.info("2. Restart nanobot")
            logger.info("3. Verify that memory works correctly")
            logger.info("4. Optionally back up and remove old JSONL files")

    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
