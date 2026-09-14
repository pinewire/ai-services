from uuid import UUID

from app.evaluation import build_report, score_retrieval
from app.retrieval import RetrievedChunk


def _chunk(title: str, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=UUID(int=rank),
        doc_title=title,
        content=title,
        vector_score=0.9,
        keyword_score=0.8,
        fused_score=1.0 / rank,
    )


def test_retrieval_scores_ranked_relevant_chunks() -> None:
    scores = score_retrieval(
        [_chunk("noise", 1), _chunk("network", 2), _chunk("network", 3)],
        {"network"},
        k=3,
    )
    assert scores.recall_at_5 == 1.0
    assert scores.precision_at_5 == 1 / 3
    assert scores.mrr == 0.5
    assert 0.0 < scores.ndcg_at_5 < 1.0


def test_empty_relevance_set_scores_zero_retrieval_precision() -> None:
    scores = score_retrieval([_chunk("noise", 1)], set(), k=5)
    assert scores.recall_at_5 == 0.0
    assert scores.precision_at_5 == 0.0
    assert scores.mrr == 0.0
    assert scores.ndcg_at_5 == 0.0


def test_report_averages_case_metrics() -> None:
    scores = score_retrieval([_chunk("network", 1)], {"network"})
    from app.evaluation import CaseResult

    report = build_report(
        [
            CaseResult(
                case_id="one",
                retrieval=scores,
                category_correct=True,
                priority_correct=False,
                refusal_correct=True,
                citation_precision=1.0,
                citation_recall=1.0,
                grounded_answer=True,
                latency_ms=100,
                confidence=0.8,
            )
        ]
    )
    assert report["summary"]["mrr"] == 1.0
    assert report["summary"]["category_accuracy"] == 1.0
    assert report["summary"]["priority_accuracy"] == 0.0
