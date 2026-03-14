import json

import psycopg

from rag_agent.domain.models import Chunk


class PostgresChunkRepository:
    def __init__(self, connection_url: str) -> None:
        self._conn = psycopg.connect(connection_url)
        self._ensure_table()

    def _ensure_table(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id UUID PRIMARY KEY,
                    document_id UUID NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{}'
                )
                """
            )
        self._conn.commit()

    def save(self, chunks: list[Chunk]) -> None:
        with self._conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks (id, document_id, content, metadata)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                content = EXCLUDED.content, metadata = EXCLUDED.metadata
                """,
                [
                    (
                        chunk.id,
                        chunk.document_id,
                        chunk.content,
                        json.dumps(chunk.metadata),
                    )
                    for chunk in chunks
                ],
            )
        self._conn.commit()

    def get_all(self) -> list[Chunk]:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, document_id, content, metadata FROM chunks
                """
            )
            return [
                Chunk(
                    id=row[0],
                    document_id=row[1],
                    content=row[2],
                    metadata=row[3],
                )
                for row in cur.fetchall()
            ]
