from typing import Protocol

from rag_agent.core.domain.models import Chunk

class VectorStore(Protocol):
    def add(self, chunks: list[Chunk]) -> None:
        ...
    def search(self, embedding: list[float], top_k: int) -> list[Chunk]:
        ...
