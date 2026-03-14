from rag_agent.domain.models import Chunk
from rag_agent.domain.ports import Retriever


class HybridRetriever:
    def __init__(
        self,
        dense_retriever: Retriever,
        sparse_retriever: Retriever,
        top_k: int = 5,
        rrf_k: int = 60,
    ) -> None:
        self._dense = dense_retriever
        self._sparse = sparse_retriever
        self._top_k = top_k
        self._rrf_k = rrf_k

    def retrieve(self, query: str) -> list[Chunk]:
        dense_results = self._dense.retrieve(query)
        sparse_results = self._sparse.retrieve(query)

        scores: dict[str, float] = {}
        chunks_by_id: dict[str, Chunk] = {}

        for rank, chunk in enumerate(dense_results):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (self._rrf_k + rank + 1)
            chunks_by_id[chunk.id] = chunk

        for rank, chunk in enumerate(sparse_results):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (self._rrf_k + rank + 1)
            chunks_by_id[chunk.id] = chunk

        sorted_ids = sorted(scores, key=scores.__getitem__, reverse=True)
        return [chunks_by_id[cid] for cid in sorted_ids[: self._top_k]]
