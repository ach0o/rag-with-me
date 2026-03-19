from typing import Any

from rag_agent.domain.models import Chunk, Document
from rag_agent.domain.pipeline import Pipeline
from rag_agent.domain.ports import (
    DocLoader,
    Chunker,
    Embedder,
    VectorStore,
    DocumentRepository,
    ChunkRepository,
)


class LoadStage:
    def __init__(self, loaders: list[DocLoader]) -> None:
        self._loaders = loaders

    def process(self, data: Any) -> list[Document]:
        documents: list[Document] = []
        for loader in self._loaders:
            documents.extend(loader.load())
        return documents


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


class PersistDocumentStage:
    def __init__(self, document_repository: DocumentRepository) -> None:
        self._document_repository = document_repository

    def process(self, data: list[Document]) -> list[Document]:
        self._document_repository.save(data)
        return data


class PersistChunkStage:
    def __init__(self, chunk_repository: ChunkRepository) -> None:
        self._chunk_repository = chunk_repository

    def process(self, data: list[Chunk]) -> list[Chunk]:
        self._chunk_repository.save(data)
        return data


class IngestUseCase:
    def __init__(
        self,
        loaders: list[DocLoader],
        chunker: Chunker,
        embedder: Embedder,
        vector_store: VectorStore,
        document_repository: DocumentRepository | None = None,
        chunk_repository: ChunkRepository | None = None,
    ) -> None:
        pipeline = Pipeline()
        pipeline.add_stage(LoadStage(loaders))
        if document_repository:
            pipeline.add_stage(PersistDocumentStage(document_repository))
        pipeline.add_stage(ChunkStage(chunker))
        if chunk_repository:
            pipeline.add_stage(PersistChunkStage(chunk_repository))
        pipeline.add_stage(EmbedStage(embedder))
        pipeline.add_stage(StoreStage(vector_store))

        self._pipeline = pipeline

    def execute(self) -> list[Chunk]:
        return self._pipeline.run(None)
