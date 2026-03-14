from pathlib import Path

from rag_agent.domain.models import Document


class MarkdownDocLoader:
    def __init__(self, path: str) -> None:
        self._path = Path(path)

    def load(self) -> list[Document]:
        documents: list[Document] = []
        for file in sorted(self._path.glob("**/*.md")):
            content = file.read_text(encoding="utf-8")
            documents.append(
                Document(
                    content=content,
                    metadata={
                        "source": str(file.absolute()),
                        "title": file.stem,
                    },
                )
            )
        return documents
