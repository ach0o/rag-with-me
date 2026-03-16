from rag_agent.domain.models import Chunk
from rag_agent.domain.ports import VectorStore


class FakeVectorStore:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk]) -> None:
        self.chunks.extend(chunks)

    def search(self, embedding: list[float], top_k: int) -> list[Chunk]:
        return self.chunks[:top_k]


# Protocol conformance check — ensures FakeVectorStore satisfies the VectorStore port.
# If this breaks, the port interface changed and the fake needs updating.
_: VectorStore = FakeVectorStore()
