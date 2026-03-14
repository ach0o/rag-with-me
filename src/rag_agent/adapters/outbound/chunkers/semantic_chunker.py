import re

from rag_agent.domain.models import Chunk, Document
from rag_agent.domain.ports import Embedder


class SemanticChunker:
    def __init__(
        self,
        embedder: Embedder,
        threshold: float = 0.5,
        min_chunk_size: int = 100,
    ) -> None:
        self._embedder = embedder
        self._threshold = threshold
        self._min_chunk_size = min_chunk_size

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [sent.strip() for sent in sentences if sent.strip()]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _group_text_len(sentences: list[str], group: list[int]) -> int:
        return sum(len(sentences[i]) for i in group)

    def chunk(self, document: Document) -> list[Chunk]:
        sentences = self._split_sentences(document.content)
        if len(sentences) <= 1:
            return [
                Chunk(
                    document_id=document.id,
                    content=document.content,
                    metadata={
                        **document.metadata,
                        "chunk_index": 0,
                        "start_char": 0,
                        "end_char": len(document.content),
                    },
                )
            ]

        embeddings = self._embedder.embed(sentences)
        groups: list[list[int]] = [[0]]

        for i in range(1, len(sentences)):
            similarity = self._cosine_similarity(embeddings[i - 1], embeddings[i])
            if (
                similarity < self._threshold
                and self._group_text_len(sentences, groups[-1]) >= self._min_chunk_size
            ):
                groups.append([i])
            else:
                groups[-1].append(i)

        chunks: list[Chunk] = []
        char_offset = 0

        for idx, group in enumerate(groups):
            content = " ".join(sentences[i] for i in group)
            chunks.append(
                Chunk(
                    document_id=document.id,
                    content=content,
                    metadata={
                        **document.metadata,
                        "chunk_index": idx,
                        "start_char": char_offset,
                        "end_char": char_offset + len(content),
                    },
                )
            )
            char_offset += len(content) + 1

        return chunks
