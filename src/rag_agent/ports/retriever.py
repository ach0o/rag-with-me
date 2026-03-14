from typing import Protocol

from rag_agent.core.domain.models import Chunk

class Retriever(Protocol):
    def retrieve(self, query: str) -> list[Chunk]:
        ...
