from pathlib import Path

import pymupdf

from rag_agent.domain.models import Document
from rag_agent.domain.ports import ImageDescriber


class PdfDocLoader:
    def __init__(
        self,
        path: str,
        image_describer: ImageDescriber | None = None,
    ) -> None:
        self._path = Path(path)
        self._image_describer = image_describer

    def _extract_images(self, page: pymupdf.Page) -> list[bytes]:
        images = []
        for img_info in page.get_images():
            xref = img_info[0]
            base_image = page.parent.extract_image(xref)
            if base_image and base_image["image"]:
                images.append(base_image["image"])
        return images

    def _describe_page_images(self, images: list[bytes]) -> str:
        if not self._image_describer or not images:
            return ""
        descriptions = []
        for img_bytes in images:
            desc = self._image_describer.describe(img_bytes)
            if desc:
                descriptions.append(f"[Image: {desc}]")
        return "\n\n".join(descriptions)

    def load(self) -> list[Document]:
        documents: list[Document] = []
        pdf_files = (
            sorted(self._path.glob("**/*.pdf")) if self._path.is_dir() else [self._path]
        )

        for file in pdf_files:
            with pymupdf.open(file) as doc:
                for page_num, page in enumerate(doc):
                    text = page.get_text().strip()
                    images = self._extract_images(page)
                    image_text = self._describe_page_images(images)
                    if image_text:
                        text = f"{text}\n\n{image_text}" if text else image_text

                    if not text:
                        continue

                    documents.append(
                        Document(
                            content=text,
                            metadata={
                                "source": str(file.absolute()),
                                "title": file.stem,
                                "page": page_num + 1,
                                "has_images": len(images) > 0,
                            },
                        )
                    )
        return documents
