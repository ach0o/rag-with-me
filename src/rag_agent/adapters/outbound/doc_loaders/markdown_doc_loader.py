import re
from pathlib import Path

import requests

from rag_agent.domain.models import Document
from rag_agent.domain.ports import ImageDescriber

IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


class MarkdownDocLoader:
    def __init__(
        self,
        path: str,
        image_describer: ImageDescriber | None = None,
    ) -> None:
        self._path = Path(path)
        self._image_describer = image_describer

    def load(self) -> list[Document]:
        return [
            self._load_file(md_file) for md_file in sorted(self._path.glob("**/*.md"))
        ]

    def _load_file(self, md_file: Path) -> Document:
        content = md_file.read_text(encoding="utf-8")
        has_images = bool(IMAGE_PATTERN.search(content))

        if self._image_describer and has_images:
            content = self._replace_images(md_file, content)

        return Document(
            content=content,
            metadata={
                "source": str(md_file.absolute()),
                "title": md_file.stem,
                "has_images": has_images,
            },
        )

    def _replace_images(self, md_file: Path, content: str) -> str:
        def replacer(match: re.Match) -> str:
            alt, path = match.group(1), match.group(2)
            image_bytes = self._load_image(md_file, path)
            if image_bytes:
                desc = self._image_describer.describe(image_bytes)
                if desc:
                    return f"[Image: {desc}]"
            return f"[Image: {alt}]" if alt else match.group(0)

        return IMAGE_PATTERN.sub(replacer, content)

    def _load_image(self, md_file: Path, img_path: str) -> bytes | None:
        if img_path.startswith(("http://", "https://")):
            return self._fetch_remote(img_path)
        return self._fetch_local(md_file, img_path)

    def _fetch_remote(self, url: str) -> bytes | None:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return resp.content
        except requests.RequestException:
            return None

    def _fetch_local(self, md_file: Path, img_path: str) -> bytes | None:
        resolved = (md_file.parent / img_path).resolve()
        if not resolved.is_file():
            return None
        return resolved.read_bytes()
