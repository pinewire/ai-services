"""Hybrid retrieval: pgvector nearest-neighbour search fused with Postgres
full-text search via reciprocal rank fusion (RRF).

Keyword search is what catches exact strings like VPN-4021 — embeddings
alone treat error codes as near-noise.
"""

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

RRF_K = 60
TOP_N = 5
CANDIDATE_N = 10
MIN_VECTOR_SCORE = 0.1  # below this, a hashed-embedding "match" is just noise


@dataclass
class RetrievedChunk:
    chunk_id: UUID
    doc_title: str
    content: str
    vector_score: float | None
    keyword_score: float | None
    fused_score: float


async def _vector_search(session: AsyncSession, embedding: list[float], limit: int) -> list[tuple]:
    rows = await session.execute(
        text(
            """
            select c.id, d.title, c.content, 1 - (c.embedding <=> (:embedding)::vector) as score
            from kb_chunks c
            join kb_documents d on d.id = c.document_id
            where 1 - (c.embedding <=> (:embedding)::vector) > :min_score
            order by c.embedding <=> (:embedding)::vector
            limit :limit
            """
        ),
        {"embedding": str(embedding), "min_score": MIN_VECTOR_SCORE, "limit": limit},
    )
    return rows.all()


async def _keyword_search(session: AsyncSession, query_text: str, limit: int) -> list[tuple]:
    rows = await session.execute(
        text(
            """
            select c.id, d.title, c.content,
                   ts_rank_cd(c.content_tsv, plainto_tsquery('english', :q)) as score
            from kb_chunks c
            join kb_documents d on d.id = c.document_id
            where c.content_tsv @@ plainto_tsquery('english', :q)
            order by score desc
            limit :limit
            """
        ),
        {"q": query_text, "limit": limit},
    )
    return rows.all()


def _fuse(vector_hits: list[tuple], keyword_hits: list[tuple]) -> list[RetrievedChunk]:
    fused: dict[UUID, RetrievedChunk] = {}
    for rank, row in enumerate(vector_hits, start=1):
        fused[row.id] = RetrievedChunk(row.id, row.title, row.content, row.score, None, 1.0 / (RRF_K + rank))
    for rank, row in enumerate(keyword_hits, start=1):
        if row.id in fused:
            existing = fused[row.id]
            existing.keyword_score = row.score
            existing.fused_score += 1.0 / (RRF_K + rank)
        else:
            fused[row.id] = RetrievedChunk(row.id, row.title, row.content, None, row.score, 1.0 / (RRF_K + rank))
    return sorted(fused.values(), key=lambda c: c.fused_score, reverse=True)


async def retrieve(session: AsyncSession, query_text: str, embedding: list[float]) -> list[RetrievedChunk]:
    vector_hits = await _vector_search(session, embedding, CANDIDATE_N)
    keyword_hits = await _keyword_search(session, query_text, CANDIDATE_N)
    return _fuse(vector_hits, keyword_hits)[:TOP_N]
