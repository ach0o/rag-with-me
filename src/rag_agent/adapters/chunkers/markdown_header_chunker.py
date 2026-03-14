import re

from rag_agent.core.domain.models import Chunk, Document


class MarkdownHeaderChunker:
    def chunk(self, document: Document) -> list[Chunk]:
        lines = document.content.split("\n")
        chunks: list[Chunk] = []
        headers: list[str] = []
        current_lines: list[str] = []
        start_char = 0

        for line in lines:
            header_match = re.match(r"^(#{1,6})\s+(.*)", line)
            if header_match and current_lines:
                content = "\n".join(current_lines).strip()
                if content:
                    chunks.append(
                        Chunk(
                            document_id=document.id,
                            content=content,
                            metadata={
                                **document.metadata,
                                "chunk_index": len(chunks),
                                "start_char": start_char,
                                "end_char": start_char + len(content),
                                "headers": list(headers),
                            },
                            id=f"{document.id}_{len(chunks)}",
                        )
                    )
                current_lines = []
                start_char += len(content) + 1

            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                headers = [h for h in headers if h.count("#") < level]
                headers.append("#" * level + " " + title)

            current_lines.append(line)

        content = "\n".join(current_lines).strip()
        if content:
            chunks.append(
                Chunk(
                    document_id=document.id,
                    content=content,
                    metadata={
                        **document.metadata,
                        "chunk_index": len(chunks),
                        "start_char": start_char,
                        "end_char": start_char + len(content),
                        "headers": list(headers),
                    },
                )
            )
        return chunks
