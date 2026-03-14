from rag_agent.domain.models import Chunk, Document


class FixedSizeChunker:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content
        chunks: list[Chunk] = []
        start = 0
        index = 0

        while start < len(text):
            end = start + self._chunk_size
            chunk = text[start:end]
            chunks.append(
                Chunk(
                    document_id=document.id,
                    content=chunk,
                    metadata={
                        **document.metadata,
                        "chunk_index": index,
                        "start_char": start,
                        "end_char": min(end, len(text)),
                    },
                )
            )
            start += self._chunk_size - self._chunk_overlap
            index += 1

        return chunks
