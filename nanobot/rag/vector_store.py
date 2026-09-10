class RAGVectorStore:
    def __init__(self, persist_dir: str = ".nanobot_rag"):
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "RAG support requires the optional chromadb and sentence-transformers dependencies"
            ) from exc
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection("documents")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def add_chunks(self, doc_id: str, chunks: list[str], source: str):
        embeddings = self.model.encode(chunks).tolist()

        self.collection.add(
            ids=[f"{doc_id}-{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=embeddings,
            metadatas=[
                {"source": source, "chunk_index": i}
                for i in range(len(chunks))
            ],
        )

    def search(self, query: str, top_k: int = 5):
        query_embedding = self.model.encode([query]).tolist()[0]

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        docs = result["documents"][0]
        metas = result["metadatas"][0]

        return [
            {
                "content": doc,
                "source": meta["source"],
                "chunk_index": meta["chunk_index"],
            }
            for doc, meta in zip(docs, metas)
        ]
