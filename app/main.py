"""Service B — triage engine.

Stateless with respect to tickets: it answers "what do you make of this?"
and never stores, mutates, or replies on behalf of anyone. It is allowed to
be wrong, slow, or entirely unavailable — that permission is what lets A's
guarantees not depend on B's.
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, Response, status
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import select, text

from app.concurrency import limiter
from app.db import Base, async_session, engine, ensure_extensions
from app.metrics import (
    inflight_triage,
    triage_latency_seconds,
    triage_overloaded_total,
    triage_requests_total,
)
from app.models import Feedback, TriageRun
from app.pipeline import run_triage
from app.schemas import (
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    TriageRequest,
    TriageResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format='{"level":"%(levelname)s","msg":"%(message)s"}',
)
log = logging.getLogger("service-b")

PROMPT_VERSION = os.getenv("PROMPT_VERSION", "pipeline-v1")
TRIAGE_TIMEOUT_SECONDS = float(os.getenv("TRIAGE_TIMEOUT_SECONDS", "30"))


@asynccontextmanager
async def lifespan(_: FastAPI):
    async with engine.begin() as conn:
        await ensure_extensions(conn)
        await conn.run_sync(Base.metadata.create_all)
    yield
    # Dispose pooled connections so a fresh event loop (e.g. the next test
    # run) doesn't inherit asyncpg connections tied to a closed loop.
    await engine.dispose()


app = FastAPI(
    title="Service B — Triage",
    version="0.2.0",
    description="Classifies helpdesk tickets and drafts grounded replies.",
    lifespan=lifespan,
)


@app.get("/healthz", include_in_schema=False)
async def healthz() -> dict[str, str]:
    """Liveness. Is the process up?"""
    return {"status": "ok"}


@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    """Prometheus scrape endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


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
    return {"status": "ready", "prompt_version": PROMPT_VERSION}


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

    if fail or not await limiter.try_acquire():
        triage_overloaded_total.inc()
        triage_requests_total.labels(outcome="overloaded").inc()
        triage_latency_seconds.observe(time.perf_counter() - started)
        log.warning("triage.overloaded request_id=%s", payload.request_id)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            headers={"Retry-After": "30", "X-Request-Id": str(payload.request_id)},
            content=ErrorResponse(
                error="overloaded",
                detail="Backpressure. Retry via the outbox.",
                request_id=str(payload.request_id),
            ).model_dump(mode="json"),
        )

    inflight_triage.inc()
    try:
        if delay_ms:
            # await, not sleep — this is why the framework is async. One slow
            # request must not occupy the event loop.
            await asyncio.sleep(delay_ms / 1000)
        async with async_session() as session:
            async with asyncio.timeout(TRIAGE_TIMEOUT_SECONDS):
                result = await run_triage(session, payload)
        outcome = "refused" if result.category == "other" else "success"
        triage_requests_total.labels(outcome=outcome).inc()
        triage_latency_seconds.observe(time.perf_counter() - started)
        response.headers["X-Request-Id"] = str(payload.request_id)
        log.info(
            "triage.completed request_id=%s category=%s confidence=%.2f latency_ms=%d",
            payload.request_id,
            result.category,
            result.confidence,
            result.latency_ms,
        )
        return result
    except Exception:
        triage_requests_total.labels(outcome="error").inc()
        triage_latency_seconds.observe(time.perf_counter() - started)
        error_response = ErrorResponse(
            error="triage_failed",
            detail="Triage provider failed. Retry via the outbox.",
            request_id=payload.request_id,
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            headers={"Retry-After": "30", "X-Request-Id": str(payload.request_id)},
            content=error_response.model_dump(mode="json"),
        )
    finally:
        inflight_triage.dec()
        await limiter.release()


@app.post(
    "/v1/feedback",
    response_model=FeedbackResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Record an agent correction against a past triage run",
)
async def feedback(payload: FeedbackRequest) -> FeedbackResponse | JSONResponse:
    async with async_session() as session:
        run = await session.scalar(select(TriageRun).where(TriageRun.request_id == payload.request_id))
        if run is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content=ErrorResponse(
                    error="not_found",
                    detail="No triage run for that request_id.",
                    request_id=str(payload.request_id),
                ).model_dump(mode="json"),
            )

        record = Feedback(
            run_id=run.id,
            verdict=payload.verdict,
            corrected_category=payload.corrected_category,
            corrected_priority=payload.corrected_priority.value if payload.corrected_priority else None,
        )
        session.add(record)
        await session.commit()
        await session.refresh(record)

    return FeedbackResponse(status="recorded", run_id=record.id)

