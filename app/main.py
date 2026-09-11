"""Service B — triage stub.

Week 3 deliverable: returns contract-shaped responses with no AI code, so the
Service A team can build and test the whole intake path against it.

Two things make this more than a fake:

  * The response validates against the real contract, so when the pipeline
    lands behind it, Service A does not change.
  * The `delay_ms` and `fail` query parameters let Service A exercise its
    timeout, circuit breaker and outbox worker on demand. You should not
    have to kill a container to test the degraded path.
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, Response, status
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.db import Base, async_session, engine
from app.models import TriageLog
from app.schemas import Citation, ErrorResponse, Priority, TriageRequest, TriageResponse

logging.basicConfig(
    level=logging.INFO,
    format='{"level":"%(levelname)s","msg":"%(message)s"}',
)
log = logging.getLogger("service-b")

STUB_VERSION = os.getenv("PROMPT_VERSION", "stub-v0")


@asynccontextmanager
async def lifespan(_: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    # Dispose pooled connections so a fresh event loop (e.g. the next test
    # run) doesn't inherit asyncpg connections tied to a closed loop.
    await engine.dispose()


app = FastAPI(
    title="Service B — Triage",
    version="0.1.0",
    description="Classifies helpdesk tickets and drafts grounded replies.",
    lifespan=lifespan,
)

# Crude keyword routing so A's team sees varied responses in development.
# The real pipeline replaces this entirely; the response shape does not change.
_RULES: list[tuple[tuple[str, ...], str, Priority, float]] = [
    (("vpn", "network", "wifi", "connection"), "network.vpn", Priority.p2, 0.83),
    (("password", "login", "locked out", "mfa"), "access.password", Priority.p3, 0.91),
    (("laptop", "screen", "keyboard", "battery"), "hardware.laptop", Priority.p3, 0.74),
    (("payroll", "invoice", "expense"), "finance.systems", Priority.p2, 0.68),
]


def _classify(text: str) -> tuple[str, Priority, float]:
    lowered = text.lower()
    for keywords, category, priority, confidence in _RULES:
        if any(k in lowered for k in keywords):
            return category, priority, confidence
    # No rule matched. B is allowed — required — to say it does not know.
    return "other", Priority.p3, 0.31


@app.get("/healthz", include_in_schema=False)
async def healthz() -> dict[str, str]:
    """Liveness. Is the process up?"""
    return {"status": "ok"}


@app.get("/readyz", include_in_schema=False, response_model=None)
async def readyz() -> dict[str, str] | JSONResponse:
    """Readiness. Confirms the process is up and Postgres is reachable."""
    try:
        async with async_session() as session:
            await session.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001 — any DB failure means not ready
        log.warning("readyz.db_unreachable error=%s", exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not-ready", "detail": "database unreachable"},
        )
    return {"status": "ready", "prompt_version": STUB_VERSION}


@app.post(
    "/v1/triage",
    response_model=TriageResponse,
    responses={503: {"model": ErrorResponse}},
    summary="Classify a ticket and draft a grounded reply",
)
async def triage(
    payload: TriageRequest,
    response: Response,
    delay_ms: int = Query(0, ge=0, le=30_000, description="Simulate a slow model call"),
    fail: bool = Query(False, description="Simulate B being overloaded"),
) -> TriageResponse | JSONResponse:
    started = time.perf_counter()

    log.info("triage.received request_id=%s ticket_ref=%s", payload.request_id, payload.ticket_ref)

    if delay_ms:
        # await, not sleep — this is why the framework is async. One slow
        # request must not occupy the event loop.
        await asyncio.sleep(delay_ms / 1000)

    if fail:
        log.warning("triage.simulated_failure request_id=%s", payload.request_id)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            headers={"Retry-After": "30"},
            content=ErrorResponse(
                error="overloaded",
                detail="Simulated backpressure. Retry via the outbox.",
                request_id=str(payload.request_id),
            ).model_dump(mode="json"),
        )

    text = payload.subject + " " + " ".join(m.body for m in payload.messages)
    category, priority, confidence = _classify(text)

    if category == "other":
        # The refusal path. No invented fix, no citations, and the useful
        # output is the questions that would make the ticket classifiable.
        result = TriageResponse(
            request_id=payload.request_id,
            category=category,
            priority=priority,
            confidence=confidence,
            suggested_reply=None,
            clarifying_questions=[
                "What exactly happens when you try — any error message on screen?",
                "When did this start, and did anything change just before?",
                "Does it happen on every attempt or only sometimes?",
            ],
            citations=[],
            model="stub",
            prompt_version=STUB_VERSION,
            latency_ms=int((time.perf_counter() - started) * 1000),
            token_cost_usd=0.0,
        )
    else:
        result = TriageResponse(
            request_id=payload.request_id,
            category=category,
            priority=priority,
            confidence=confidence,
            suggested_reply=(
                f"Thanks for reporting this. This looks like a known {category} issue. "
                "An agent will confirm the fix from the runbook shortly."
            ),
            clarifying_questions=[],
            citations=[
                Citation(chunk_id="kb_0001#c1", doc_title="Stub Runbook", score=0.88)
            ],
            model="stub",
            prompt_version=STUB_VERSION,
            latency_ms=int((time.perf_counter() - started) * 1000),
            token_cost_usd=0.0,
        )

    # Best-effort persistence: a DB hiccup should not fail a triage response.
    try:
        async with async_session() as session:
            session.add(
                TriageLog(
                    request_id=payload.request_id,
                    ticket_ref=payload.ticket_ref,
                    category=result.category,
                    priority=result.priority.value,
                    confidence=result.confidence,
                    latency_ms=result.latency_ms,
                )
            )
            await session.commit()
    except Exception as exc:  # noqa: BLE001
        log.warning("triage.persist_failed request_id=%s error=%s", payload.request_id, exc)

    response.headers["X-Request-Id"] = str(payload.request_id)
    log.info(
        "triage.completed request_id=%s category=%s confidence=%.2f latency_ms=%d",
        payload.request_id,
        result.category,
        result.confidence,
        result.latency_ms,
    )
    return result
