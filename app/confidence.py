"""Confidence is composed from measurable retrieval/citation signals — never
self-reported by the model, which tends to answer 0.8-0.9 regardless of
whether retrieval found anything.
"""

from app.retrieval import RetrievedChunk


def compute_confidence(chunks: list[RetrievedChunk], citations_survived: int) -> float:
    if not chunks or citations_survived == 0:
        return 0.3  # hard cap: nothing survived citation validation

    top = chunks[0]
    top_score = max(top.vector_score or 0.0, top.keyword_score or 0.0)

    margin = 0.0
    if len(chunks) > 1:
        second = chunks[1]
        second_score = max(second.vector_score or 0.0, second.keyword_score or 0.0)
        margin = max(0.0, top_score - second_score)

    agreement = 1.0 if (top.vector_score is not None and top.keyword_score is not None) else 0.0

    score = 0.55 * top_score + 0.2 * min(margin, 1.0) + 0.25 * agreement
    return round(min(max(score, 0.0), 0.99), 2)
