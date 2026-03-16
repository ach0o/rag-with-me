from rag_agent.domain.models import Document
from rag_agent.domain.ports import DocumentRepository


class FakeDocumentRepository:
    def __init__(self) -> None:
        self.documents: list[Document] = []

    def save(self, documents: list[Document]) -> None:
        self.documents.extend(documents)

    def get_all(self) -> list[Document]:
        return self.documents


# Protocol conformance check — ensures FakeDocumentRepository satisfies the DocumentRepository port.
# If this breaks, the port interface changed and the fake needs updating.
_: DocumentRepository = FakeDocumentRepository()
