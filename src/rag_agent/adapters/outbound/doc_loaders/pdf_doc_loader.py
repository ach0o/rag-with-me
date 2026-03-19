from pathlib import Path

import pymupdf

from rag_agent.domain.models import Document


class PdfDocLoader:
    def __init__(self, path: str) -> None:
        self._path = Path(path)

    def load(self) -> list[Document]:
        documents: list[Document] = []
        pdf_files = (
            sorted(self._path.glob("**/*.pdf")) if self._path.is_dir() else [self._path]
        )

        for file in pdf_files:
            with pymupdf.open(file) as doc:
                for page_num, page in enumerate(doc):
                    text = page.get_text().strip()
                    if not text:
                        continue
                    documents.append(
                        Document(
                            content=text,
                            metadata={
                                "source": str(file.absolute()),
                                "title": file.stem,
                                "page": page_num + 1,
                            },
                        )
                    )
        return documents
