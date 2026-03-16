from sentence_transformers import CrossEncoder

from rag_agent.domain.models import Chunk


class CrossEncoderReranker:
    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 3,
    ) -> None:
        self._model = CrossEncoder(model)
        self._top_k = top_k

    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return []

        pairs = [[query, chunk.content] for chunk in chunks]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in ranked[: self._top_k]]
