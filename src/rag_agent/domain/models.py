from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

def generate_id():
    return str(uuid4())

@dataclass
class Document:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=generate_id)

@dataclass
class Chunk:
    document_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    id: str = field(default_factory=generate_id)

@dataclass
class QueryResult:
    answer: str
    chunks: list[Chunk] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=generate_id)
