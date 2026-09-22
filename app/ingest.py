"""KB ingestion — chunks runbooks, embeds them, and populates kb_documents /
kb_chunks. Runs as a standalone job, not inside the API process, because it
has different resource needs and failure modes (see README: Hosting).

Usage:
    python -m app.ingest path/to/docs   # each *.md file becomes one document
"""

import asyncio
import sys
from pathlib import Path

from sqlalchemy import select

from app.chunking import chunk_document
from app.db import Base, async_session, engine, ensure_extensions
from app.embeddings import get_embedding_client
from app.models import KbChunk, KbDocument

CHUNK_SIZE_CHARS = 1_000
CHUNK_OVERLAP_CHARS = 150


def _chunk_text(raw: str) -> list[tuple[str, str]]:
    """Heading-aware recursive character splitting. Returns
    (heading_path, chunk_content) pairs. See `app.chunking` for the strategy.
    """
    return chunk_document(raw, chunk_size=CHUNK_SIZE_CHARS, chunk_overlap=CHUNK_OVERLAP_CHARS)


async def ingest_path(path: Path) -> None:
    embedding_client = get_embedding_client()

    async with engine.begin() as conn:
        await ensure_extensions(conn)
        await conn.run_sync(Base.metadata.create_all)

    async with async_session() as session:
        for file in sorted(path.glob("*.md")):
            source_url = file.name
            existing = await session.scalar(select(KbDocument).where(KbDocument.source_url == source_url))
            if existing is not None:
                await session.delete(existing)
                await session.flush()

            document = KbDocument(title=file.stem, source_url=source_url)
            session.add(document)
            await session.flush()

            chunk_pairs = _chunk_text(file.read_text())
            if not chunk_pairs:
                continue
            embeddings = await embedding_client.embed([content for _, content in chunk_pairs])
            for (heading, content), embedding in zip(chunk_pairs, embeddings, strict=True):
                session.add(
                    KbChunk(
                        document_id=document.id,
                        heading_path=heading,
                        token_count=len(content.split()),
                        content=content,
                        embedding=embedding,
                    )
                )
            print(f"ingested {file.name}: {len(chunk_pairs)} chunks")
        await session.commit()

    await engine.dispose()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m app.ingest <docs-folder>")
        raise SystemExit(1)
    asyncio.run(ingest_path(Path(sys.argv[1])))
