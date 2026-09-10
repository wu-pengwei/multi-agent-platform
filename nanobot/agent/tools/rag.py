from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool
from nanobot.utils.document import extract_text
from nanobot.rag.chunker import chunk_text
from nanobot.rag.vector_store import RAGVectorStore


class RAGIndexTool(Tool):
    @property
    def name(self) -> str:
        return "rag_index"

    @property
    def description(self) -> str:
        return "Index a local document into the RAG knowledge base."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the document file.",
                }
            },
            "required": ["path"],
        }

    async def execute(self, path: str) -> str:
        file_path = Path(path)
        text = extract_text(file_path)

        if not text or text.startswith("[error"):
            return f"Failed to extract text from {path}: {text}"

        chunks = chunk_text(text)
        store = RAGVectorStore()
        store.add_chunks(doc_id=file_path.stem, chunks=chunks, source=str(file_path))

        return f"Indexed {len(chunks)} chunks from {path}."


class RAGSearchTool(Tool):
    @property
    def name(self) -> str:
        return "rag_search"

    @property
    def description(self) -> str:
        return "Search the RAG knowledge base for relevant document chunks."

    @property
    def read_only(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "User question or search query.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of chunks to retrieve.",
                    "default": 5,
                },
            },
            "required": ["query"],
        }

    async def execute(self, query: str, top_k: int = 5) -> str:
        store = RAGVectorStore()
        results = store.search(query, top_k=top_k)

        if not results:
            return "No relevant document chunks found."

        return "\n\n".join(
            f"[Source: {r['source']} | Chunk: {r['chunk_index']}]\n{r['content']}"
            for r in results
        )