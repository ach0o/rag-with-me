from rag_agent.domain.models import Document
from rag_agent.domain.ports import DocLoader


class FakeDocLoader:
    def __init__(self, documents: list[Document] | None = None) -> None:
        self.documents = documents or []

    def load(self) -> list[Document]:
        return self.documents


# Protocol conformance check — ensures FakeDocLoader satisfies the DocLoader port.
# If this breaks, the port interface changed and the fake needs updating.
_: DocLoader = FakeDocLoader()
