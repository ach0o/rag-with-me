from typing import Protocol

from rag_agent.core.domain.models import Document

class DocLoader(Protocol):
    def load(self) -> list[Document]:
        ...