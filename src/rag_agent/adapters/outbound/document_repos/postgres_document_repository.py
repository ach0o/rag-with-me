import json
import psycopg

from rag_agent.domain.models import Document


class PostgresDocumentRepository:
    def __init__(self, connection_url: str) -> None:
        self._conn = psycopg.connect(connection_url)
        self._ensure_table()

    def _ensure_table(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id UUID PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{}'
                )
                """
            )
        self._conn.commit()

    def save(self, documents: list[Document]) -> None:
        with self._conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO documents (id, content, metadata)
                VALUES (%s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                content = EXCLUDED.content, metadata = EXCLUDED.metadata
                """,
                [
                    (
                        doc.id,
                        doc.content,
                        json.dumps(doc.metadata),
                    )
                    for doc in documents
                ],
            )
        self._conn.commit()

    def get_all(self) -> list[Document]:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, content, metadata FROM documents
                """
            )
            return [
                Document(
                    id=row[0],
                    content=row[1],
                    metadata=row[2],
                )
                for row in cur.fetchall()
            ]
