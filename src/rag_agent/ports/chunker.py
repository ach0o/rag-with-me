from typing import Protocol

from rag_agent.core.domain.models import Chunk, Document


class Chunker(Protocol):
    def chunk(self, document: Document) -> list[Chunk]:
        ...
