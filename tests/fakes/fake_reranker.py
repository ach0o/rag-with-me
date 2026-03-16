from rag_agent.domain.models import Chunk
from rag_agent.domain.ports import Reranker


class FakeReranker:
    def __init__(self) -> None:
        self.called = False

    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        self.called = True
        return list(reversed(chunks))


# Protocol conformance check — ensures FakeReranker satisfies the Reranker port.
# If this breaks, the port interface changed and the fake needs updating.
_: Reranker = FakeReranker()
