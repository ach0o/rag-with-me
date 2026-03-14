from pathlib import Path
from typing import Literal, Self

import yaml

from pydantic import BaseModel, Field, model_validator


class LLMConfig(BaseModel):
    provider: Literal["azure-openai"] = "azure-openai"
    model: str = "gpt-4o-mini"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)


class EmbedderConfig(BaseModel):
    provider: Literal["azure-openai"] = "azure-openai"
    model: str = "text-embedding-3-small"


class VectorStoreConfig(BaseModel):
    provider: Literal["chromadb"] = "chromadb"
    collection_name: str = "rag_97things"
    path: str = "./data/chroma_vector_db"


class ChunkerConfig(BaseModel):
    strategy: Literal["fixed-size", "markdown-header", "semantic"] = "fixed-size"
    chunk_size: int = Field(default=500, gt=0)
    chunk_overlap: int = Field(default=50, ge=0)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    min_chunk_size: int = Field(default=100, gt=0)

    @model_validator(mode="after")
    def check_overlap(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return self


class RetrieverConfig(BaseModel):
    provider: Literal["dense", "bm25_sparse", "hybrid"] = "dense"
    top_k: int = Field(default=5, gt=0)


class RerankerConfig(BaseModel):
    provider: Literal["none"] = "none"
    top_k: int = Field(default=3, gt=0)


class DataSourceConfig(BaseModel):
    path: str = "./data/97-things"
    type: Literal["markdown"] = "markdown"


class DatabaseConfig(BaseModel):
    url: str = ""
    enabled: bool = True


class AppConfig(BaseModel):
    llm: LLMConfig = LLMConfig()
    embedder: EmbedderConfig = EmbedderConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    chunker: ChunkerConfig = ChunkerConfig()
    retriever: RetrieverConfig = RetrieverConfig()
    reranker: RerankerConfig = RerankerConfig()
    data_source: DataSourceConfig = DataSourceConfig()
    database: DatabaseConfig = DatabaseConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
