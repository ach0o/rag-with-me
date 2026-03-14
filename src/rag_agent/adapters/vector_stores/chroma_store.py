import chromadb

from rag_agent.core.domain.models import Chunk


class ChromaStore:
    def __init__(
        self,
        collection_name: str = "rag_97things",
        path: str = "./data/chroma_vector_db",
    ) -> None:

        self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def add(self, chunks: list[Chunk]) -> None:
        self._collection.add(
            documents=[chunk.content for chunk in chunks],
            metadatas=[
                {
                    "document_id": chunk.document_id,
                    **chunk.metadata,
                }
                for chunk in chunks
            ],
            embeddings=[chunk.embedding for chunk in chunks],
            ids=[chunk.id for chunk in chunks],
        )

    def search(self, embedding: list[float], top_k: int) -> list[Chunk]:
        results = self._collection.query(query_embeddings=[embedding], n_results=top_k)
        chunks: list[Chunk] = []
        for i in range(len(results["ids"][0])):
            chunks.append(
                Chunk(
                    document_id=results["metadatas"][0][i].get("document_id", ""),
                    content=results["documents"][0][i],
                    metadata=results["metadatas"][0][i],
                    id=results["ids"][0][i],
                )
            )
        return chunks
