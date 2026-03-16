from rag_agent.domain.models import Chunk
from rag_agent.domain.ports import Retriever


class FakeRetriever:
    def __init__(self, chunks: list[Chunk] | None = None) -> None:
        self.chunks = chunks or []

    def retrieve(self, query: str) -> list[Chunk]:
        return self.chunks


# Protocol conformance check — ensures FakeRetriever satisfies the Retriever port.
# If this breaks, the port interface changed and the fake needs updating.
_: Retriever = FakeRetriever()
