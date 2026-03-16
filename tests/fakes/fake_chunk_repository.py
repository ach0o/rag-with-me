from rag_agent.domain.models import Chunk
from rag_agent.domain.ports import ChunkRepository


class FakeChunkRepository:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []

    def save(self, chunks: list[Chunk]) -> None:
        self.chunks.extend(chunks)

    def get_all(self) -> list[Chunk]:
        return self.chunks


# Protocol conformance check — ensures FakeChunkRepository satisfies the ChunkRepository port.
# If this breaks, the port interface changed and the fake needs updating.
_: ChunkRepository = FakeChunkRepository()
