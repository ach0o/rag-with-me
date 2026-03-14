from typing import Protocol

from rag_agent.domain.models import Chunk, Document


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk]) -> None: ...
    def search(self, embedding: list[float], top_k: int) -> list[Chunk]: ...


class Retriever(Protocol):
    def retrieve(self, query: str) -> list[Chunk]: ...


class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]: ...


class LLMProvider(Protocol):
    def generate(self, prompt: str) -> str: ...


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class DocLoader(Protocol):
    def load(self) -> list[Document]: ...


class Chunker(Protocol):
    def chunk(self, document: Document) -> list[Chunk]: ...
