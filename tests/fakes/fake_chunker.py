from rag_agent.domain.models import Chunk, Document
from rag_agent.domain.ports import Chunker


class FakeChunker:
    def chunk(self, document: Document) -> list[Chunk]:
        return [
            Chunk(
                document_id=document.id,
                content=document.content,
                metadata={**document.metadata, "chunk_index": 0},
            )
        ]


# Protocol conformance check — ensures FakeChunker satisfies the Chunker port.
# If this breaks, the port interface changed and the fake needs updating.
_: Chunker = FakeChunker()
