from rag_agent.core.domain.models import Chunk


class NoReranker:
    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        return chunks
