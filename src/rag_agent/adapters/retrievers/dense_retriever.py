from rag_agent.core.domain.models import Chunk
from rag_agent.ports import Embedder, VectorStore


class DenseRetriever:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        top_k: int = 5,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._top_k = top_k

    def retrieve(self, query: str) -> list[Chunk]:
        embeddings = self._embedder.embed([query])
        return self._vector_store.search(embeddings[0], top_k=self._top_k)
