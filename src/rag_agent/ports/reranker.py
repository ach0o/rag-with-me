from typing import Protocol

from rag_agent.core.domain.models import Chunk

class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        ...
