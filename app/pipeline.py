"""The triage pipeline: idempotency check, hybrid retrieval, forced-JSON
model call, citation resolution against the retrieved set, composed
confidence, and persistence. See README for the full design rationale.
"""

import hashlib
import time

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.confidence import compute_confidence
from app.embeddings import get_embedding_client
from app.errors import RequestIdConflict
from app.llm import get_triage_model
from app.metrics import (
    llm_cost_usd_total,
    llm_tokens_total,
    retrieval_chunks_returned,
    triage_cache_hits_total,
    triage_confidence,
    triage_refusals_total,
)
from app.models import TriageRun
from app.retrieval import retrieve
from app.schemas import (
    Category,
    Citation,
    TriageRequest,
    TriageResponse,
    TriageStatus,
)

PROMPT_VERSION = "pipeline-v1"


def _ticket_text(payload: TriageRequest) -> str:
    return payload.subject + " " + " ".join(m.body for m in payload.messages)


def _input_hash(payload: TriageRequest) -> str:
    normalized = _ticket_text(payload).strip().lower()
    return hashlib.sha256(normalized.encode()).hexdigest()


async def _persist_run(
    session: AsyncSession,
    payload: TriageRequest,
    input_hash: str,
    retrieved_chunk_ids: list,
    response: TriageResponse,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> None:
    run = await session.scalar(select(TriageRun).where(TriageRun.request_id == payload.request_id))
    if run is None:
        run = TriageRun(request_id=payload.request_id)
        session.add(run)
    run.input_hash = input_hash
    run.retrieved_chunk_ids = retrieved_chunk_ids
    run.model = response.model
    run.prompt_version = response.prompt_version
    run.output = response.model_dump(mode="json")
    run.confidence = response.confidence
    run.latency_ms = response.latency_ms
    run.prompt_tokens = prompt_tokens
    run.completion_tokens = completion_tokens
    run.cost_usd = response.token_cost_usd
    await session.commit()


def _as_replay(stored_output: dict, request_id, started: float) -> TriageResponse:
    """Re-serve a stored response without charging for it twice.

    `token_cost_usd` is zeroed and `replayed` set, because no model call was
    made on this request. Leaving the original cost in place meant every
    retry A made inflated the spend A computes from these responses, and
    every near-duplicate hit inflated it again — the field exists precisely
    so A can sum it, so it has to describe this call and not another one.

    `latency_ms` is likewise this call's latency, so A's latency numbers
    describe what A actually waited for rather than the original run.
    """
    return TriageResponse.model_validate(
        {
            **stored_output,
            "request_id": request_id,
            "replayed": True,
            "token_cost_usd": 0.0,
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }
    )


async def run_triage(session: AsyncSession, payload: TriageRequest) -> TriageResponse:
    started = time.perf_counter()
    model = get_triage_model()

    ticket_text = _ticket_text(payload)
    input_hash = _input_hash(payload)

    # 1. Idempotency check — A's synchronous attempt and its outbox worker
    # can both fire for the same ticket; the second one gets the cached run.
    existing = await session.scalar(select(TriageRun).where(TriageRun.request_id == payload.request_id))
    if existing is not None:
        # Same key, different question. B's guarantee is "same request_id,
        # same answer", so honouring it here would mean confidently
        # answering a ticket nobody asked about, in a response that looks
        # perfectly well-formed. Refuse instead and let A see the bug.
        # `is not None` guards rows written before input_hash was recorded.
        if existing.input_hash is not None and existing.input_hash != input_hash:
            raise RequestIdConflict(str(payload.request_id))
        if existing.model == model.model_name:
            triage_cache_hits_total.labels(cache="request_id").inc()
            return _as_replay(existing.output, payload.request_id, started)

    # Near-duplicate cache: same content, different request_id — skip the
    # whole pipeline rather than just the model call. Two distinct tickets
    # with identical text therefore get byte-identical responses, citations
    # included. That is intentional, and the contract says so, because
    # otherwise it reads as a bug.
    cached = await session.scalar(
        select(TriageRun).where(TriageRun.input_hash == input_hash).order_by(TriageRun.created_at.desc())
    )
    if cached is not None and cached.model == model.model_name:
        triage_cache_hits_total.labels(cache="input_hash").inc()
        response = _as_replay(cached.output, payload.request_id, started)
        await _persist_run(session, payload, input_hash, cached.retrieved_chunk_ids, response)
        return response

    # 2-5. Embed the ticket, hybrid-retrieve, fuse with reciprocal rank fusion.
    [embedding] = await get_embedding_client().embed([ticket_text])
    chunks = await retrieve(session, ticket_text, embedding)
    retrieval_chunks_returned.observe(len(chunks))

    # 6-7. Prompt + forced-JSON model call (validated, reject-and-retry once).
    model_result = await model.classify(ticket_text, chunks)
    output = model_result.output

    # 8. Resolve citations against the retrieved set only — a chunk ID the
    # model never saw is confabulation even if it exists in the KB.
    by_id = {str(c.chunk_id): c for c in chunks}
    surviving = [cid for cid in output.cited_chunk_ids if cid in by_id]
    citations = [
        Citation(chunk_id=cid, doc_title=by_id[cid].doc_title, score=round(min(by_id[cid].fused_score, 1.0), 4))
        for cid in surviving
    ]
    suggested_reply = output.suggested_reply if surviving else None
    clarifying_questions = output.clarifying_questions or (
        [] if surviving else ["Nothing in the knowledge base grounds a reply yet — can you share more detail?"]
    )

    # 9. Compose confidence (capped hard if no citation survived), respond.
    #
    # `ungrounded` is not a failure: B ran the whole pipeline and found that
    # the KB does not support an answer. A needs to tell that apart from a
    # 5xx, because the right handling is "route to a human with the
    # clarifying questions", not "retry later and hope".
    confidence = compute_confidence(chunks, len(surviving))
    triage_status = TriageStatus.ok if surviving else TriageStatus.ungrounded
    response = TriageResponse(
        request_id=payload.request_id,
        status=triage_status,
        category=output.category,
        priority=output.priority,
        confidence=confidence,
        suggested_reply=suggested_reply,
        clarifying_questions=clarifying_questions,
        citations=citations,
        model=model.model_name,
        prompt_version=PROMPT_VERSION,
        latency_ms=int((time.perf_counter() - started) * 1000),
        token_cost_usd=model_result.token_cost_usd,
    )

    triage_confidence.observe(response.confidence)
    if response.category == Category.other and not response.suggested_reply:
        triage_refusals_total.inc()
    if model_result.prompt_tokens:
        llm_tokens_total.labels(type="prompt").inc(model_result.prompt_tokens)
    if model_result.completion_tokens:
        llm_tokens_total.labels(type="completion").inc(model_result.completion_tokens)
    if model_result.token_cost_usd:
        llm_cost_usd_total.inc(model_result.token_cost_usd)

    await _persist_run(
        session,
        payload,
        input_hash,
        [c.chunk_id for c in chunks],
        response,
        prompt_tokens=model_result.prompt_tokens,
        completion_tokens=model_result.completion_tokens,
    )
    return response
