from typing import Any

from rag_agent.core.domain.models import Chunk, Document
from rag_agent.core.domain.pipeline import Pipeline
from rag_agent.ports import DocLoader, Chunker, Embedder, VectorStore


class LoadStage:
    def __init__(self, loader: DocLoader) -> None:
        self._loader = loader

    def process(self, data: Any) -> list[Document]:
        return self._loader.load()


class ChunkStage:
    def __init__(self, chunker: Chunker) -> None:
        self._chunker = chunker

    def process(self, data: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in data:
            chunks.extend(self._chunker.chunk(doc))
        return chunks


class EmbedStage:
    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    def process(self, data: list[Chunk]) -> list[Chunk]:
        texts = [chunk.content for chunk in data]
        embeddings = self._embedder.embed(texts)
        return [
            Chunk(
                document_id=chunk.document_id,
                content=chunk.content,
                metadata=chunk.metadata,
                embedding=embedding,
                id=chunk.id,
            )
            for chunk, embedding in zip(data, embeddings)
        ]


class StoreStage:
    def __init__(self, vector_store: VectorStore) -> None:
        self._vector_store = vector_store

    def process(self, data: list[Chunk]) -> list[Chunk]:
        self._vector_store.add(data)
        return data


class IngestUseCase:
    def __init__(
        self,
        loader: DocLoader,
        chunker: Chunker,
        embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        self._pipeline = (
            Pipeline()
            .add_stage(LoadStage(loader))
            .add_stage(ChunkStage(chunker))
            .add_stage(EmbedStage(embedder))
            .add_stage(StoreStage(vector_store))
        )

    def execute(self) -> list[Chunk]:
        return self._pipeline.run(None)
