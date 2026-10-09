"""Offline RAG evaluation.

Run with:
    python -m app.evaluation evals/rag_eval.json

The labeled set uses document titles rather than generated chunk UUIDs because
KB ingestion currently assigns new UUIDs to chunks. A production dataset can
switch to stable chunk IDs without changing the metric calculations.
"""

import argparse
import asyncio
import json
import math
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import delete

from app.db import async_session
from app.embeddings import get_embedding_client
from app.models import TriageRun
from app.pipeline import run_triage
from app.retrieval import RetrievedChunk, retrieve
from app.schemas import TriageRequest, TriageResponse

EVAL_NAMESPACE = uuid.UUID("c2a4c537-1b5d-4d1c-a2b8-9c7c4e95f0f0")


class EvalCase(BaseModel):
    case_id: str
    request: TriageRequest
    relevant_doc_titles: set[str] = Field(default_factory=set)
    expected_category: str
    expected_priority: str
    expected_refusal: bool = False


@dataclass
class RetrievalScores:
    recall_at_5: float
    precision_at_5: float
    mrr: float
    ndcg_at_5: float


@dataclass
class CaseResult:
    case_id: str
    retrieval: RetrievalScores
    category_correct: bool
    priority_correct: bool
    refusal_correct: bool
    citation_precision: float
    citation_recall: float
    grounded_answer: bool
    latency_ms: int
    confidence: float


def _relevance(chunk: RetrievedChunk, relevant_titles: set[str]) -> bool:
    return chunk.doc_title in relevant_titles


def score_retrieval(chunks: list[RetrievedChunk], relevant_titles: set[str], k: int = 5) -> RetrievalScores:
    ranked = chunks[:k]
    relevant_count = len(relevant_titles)
    seen_titles: set[str] = set()
    hits: list[bool] = []
    for chunk in ranked:
        is_new_relevant = _relevance(chunk, relevant_titles) and chunk.doc_title not in seen_titles
        if is_new_relevant:
            seen_titles.add(chunk.doc_title)
        hits.append(is_new_relevant)

    recall = sum(hits) / relevant_count if relevant_count else float(not hits)
    precision = sum(hits) / k
    reciprocal_rank = next((1.0 / rank for rank, hit in enumerate(hits, start=1) if hit), 0.0)

    dcg = sum((1.0 / math.log2(rank + 1)) for rank, hit in enumerate(hits, start=1) if hit)
    ideal_hits = min(relevant_count, k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = dcg / ideal_dcg if ideal_dcg else float(not hits)

    return RetrievalScores(
        recall_at_5=recall,
        precision_at_5=precision,
        mrr=reciprocal_rank,
        ndcg_at_5=ndcg,
    )


def _citation_scores(response: TriageResponse, chunks: list[RetrievedChunk], relevant_titles: set[str]) -> tuple[float, float]:
    by_id = {str(chunk.chunk_id): chunk for chunk in chunks}
    cited = [by_id[citation.chunk_id] for citation in response.citations if citation.chunk_id in by_id]
    correct = sum(_relevance(chunk, relevant_titles) for chunk in cited)
    precision = correct / len(cited) if cited else float(not relevant_titles and not response.suggested_reply)
    recall = correct / len(relevant_titles) if relevant_titles else float(not cited)
    return precision, recall


def score_answer(response: TriageResponse, chunks: list[RetrievedChunk], case: EvalCase, latency_ms: int) -> CaseResult:
    citation_precision, citation_recall = _citation_scores(response, chunks, case.relevant_doc_titles)
    expected_refusal = case.expected_refusal or case.expected_category == "other"
    actual_refusal = response.category == "other" and response.suggested_reply is None
    return CaseResult(
        case_id=case.case_id,
        retrieval=score_retrieval(chunks, case.relevant_doc_titles),
        category_correct=response.category == case.expected_category,
        priority_correct=response.priority.value == case.expected_priority,
        refusal_correct=actual_refusal == expected_refusal,
        citation_precision=citation_precision,
        citation_recall=citation_recall,
        grounded_answer=not response.suggested_reply or bool(response.citations),
        latency_ms=latency_ms,
        confidence=response.confidence,
    )


def _average(results: list[CaseResult], field: str) -> float:
    values = [getattr(result, field) for result in results]
    return sum(values) / len(values) if values else 0.0


def build_report(results: list[CaseResult]) -> dict[str, Any]:
    return {
        "cases": [asdict(result) for result in results],
        "summary": {
            "case_count": len(results),
            "recall_at_5": _average([result.retrieval for result in results], "recall_at_5"),
            "precision_at_5": _average([result.retrieval for result in results], "precision_at_5"),
            "mrr": _average([result.retrieval for result in results], "mrr"),
            "ndcg_at_5": _average([result.retrieval for result in results], "ndcg_at_5"),
            "category_accuracy": _average(results, "category_correct"),
            "priority_accuracy": _average(results, "priority_correct"),
            "refusal_accuracy": _average(results, "refusal_correct"),
            "citation_precision": _average(results, "citation_precision"),
            "citation_recall": _average(results, "citation_recall"),
            "grounded_answer_rate": _average(results, "grounded_answer"),
            "average_latency_ms": _average(results, "latency_ms"),
            "average_confidence": _average(results, "confidence"),
        },
    }


async def evaluate(cases: list[EvalCase]) -> dict[str, Any]:
    results: list[CaseResult] = []
    embedding_client = get_embedding_client()
    async with async_session() as session:
        for case in cases:
            started = time.perf_counter()
            ticket_text = case.request.subject + " " + " ".join(message.body for message in case.request.messages)
            [embedding] = await embedding_client.embed([ticket_text])
            chunks = await retrieve(session, ticket_text, embedding)
            request = case.request.model_copy(
                update={"request_id": uuid.uuid5(EVAL_NAMESPACE, case.case_id)}
            )
            await session.execute(delete(TriageRun).where(TriageRun.request_id == request.request_id))
            await session.commit()
            response = await run_triage(session, request)
            latency_ms = int((time.perf_counter() - started) * 1000)
            results.append(score_answer(response, chunks, case, latency_ms))
    return build_report(results)


def _load_cases(path: Path) -> list[EvalCase]:
    return [EvalCase.model_validate(item) for item in json.loads(path.read_text())]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Service B retrieval and grounded answers.")
    parser.add_argument("dataset", type=Path, help="JSON file containing labeled evaluation cases")
    parser.add_argument("--output", type=Path, default=Path("evals/rag_eval_report.json"))
    args = parser.parse_args()

    report = asyncio.run(evaluate(_load_cases(args.dataset)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
