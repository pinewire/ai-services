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

from fastapi import Depends, FastAPI, Query, Response, status
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import select, text

from app.auth import require_api_key
from app.concurrency import limiter
from app.contract import customize_openapi
from app.db import async_session, engine
from app.errors import RequestIdConflict
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
    TriageStatus,
)

logging.basicConfig(
    level=logging.INFO,
    format='{"level":"%(levelname)s","msg":"%(message)s"}',
)
log = logging.getLogger("service-b")

PROMPT_VERSION = os.getenv("PROMPT_VERSION", "pipeline-v1")

# Service B's own deadline, which should stay *shorter* than Service A's
# client timeout. A bare socket timeout on A's side yields no status, no
# prompt_version and nothing to log; a 504 from here yields all three. The
# default is also a throughput ceiling: at MAX_INFLIGHT_TRIAGE=8, an 8s
# budget sustains ~1 req/s while a 30s budget sustains ~0.27.
TRIAGE_TIMEOUT_SECONDS = float(os.getenv("TRIAGE_TIMEOUT_SECONDS", "8"))

# The chaos query parameters are a free denial-of-service for anyone who can
# reach the endpoint: `delay_ms` holds an in-flight slot open for up to 30
# seconds and `fail` rejects outright. They exist to exercise A's circuit
# breaker and outbox from the integration tests, so they are opt-in per
# environment and off everywhere else.
CHAOS_PARAMS_ENABLED = os.getenv("CHAOS_PARAMS_ENABLED", "false").strip().lower() not in {
    "false",
    "0",
    "no",
    "off",
}


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Schema is owned by Alembic, not the app. The app must start even when
    # Postgres is unreachable so /healthz can answer and /readyz can report
    # the truth — a pod that exits on a transient DB failure crash-loops
    # instead of being drained from the load balancer and recovering.
    yield
    await engine.dispose()


app = FastAPI(
    title="Service B — Triage",
    version="0.3.0",
    lifespan=lifespan,
)

# Injects the operating rules, the resolved contract decisions and real
# captured examples into the generated OpenAPI document, so /openapi.json is
# the whole contract rather than just its types. See app/contract.py.
customize_openapi(app)


@app.get("/healthz", include_in_schema=False)
async def healthz() -> dict[str, str]:
    """Liveness. Is the process up?"""
    return {"status": "ok"}


@app.get("/metrics", include_in_schema=False, dependencies=[Depends(require_api_key)])
async def metrics() -> Response:
    """Prometheus scrape endpoint.

    Authenticated: these series carry per-category ticket volumes and
    cumulative model spend, which is not something to serve to anyone who
    can reach the pod. The scrape config needs the X-API-Key header set
    accordingly — in Prometheus that is `authorization` or a
    `http_headers` entry on the scrape job.
    """
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


def _error(
    *,
    status_code: int,
    error: str,
    detail: str,
    request_id,
    retry_after: int | None = None,
) -> JSONResponse:
    """One shape for every failure, so A can branch on `error` not prose."""
    headers = {"X-Request-Id": str(request_id)}
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    return JSONResponse(
        status_code=status_code,
        headers=headers,
        content=ErrorResponse(error=error, detail=detail, request_id=request_id).model_dump(mode="json"),
    )


@app.post(
    "/v1/triage",
    response_model=TriageResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Missing or invalid API key. Do not retry."},
        409: {"model": ErrorResponse, "description": "request_id reused for different content. Do not retry."},
        429: {"model": ErrorResponse, "description": "At capacity. Retry after Retry-After; not a B failure."},
        503: {"model": ErrorResponse, "description": "Dependency failed. Retry via the outbox."},
        504: {"model": ErrorResponse, "description": "B exceeded its own deadline. Retry once."},
    },
    dependencies=[Depends(require_api_key)],
    summary="Classify a ticket and draft a grounded reply",
)
async def triage(
    payload: TriageRequest,
    response: Response,
    # include_in_schema=False keeps these out of the generated OpenAPI
    # document, which is what Service A's client is generated from. A test
    # hook that appears in the contract is a test hook someone will build
    # against.
    delay_ms: int = Query(0, ge=0, le=30_000, include_in_schema=False),
    fail: bool = Query(False, include_in_schema=False),
) -> TriageResponse | JSONResponse:
    started = time.perf_counter()
    if not CHAOS_PARAMS_ENABLED:
        delay_ms, fail = 0, False
    log.info("triage.received request_id=%s ticket_ref=%s", payload.request_id, payload.ticket_ref)

    if fail or not await limiter.try_acquire():
        triage_overloaded_total.inc()
        triage_requests_total.labels(outcome="overloaded").inc()
        triage_latency_seconds.observe(time.perf_counter() - started)
        log.warning("triage.at_capacity request_id=%s", payload.request_id)
        # 429, not 503. B shed this load deliberately and is perfectly
        # healthy; if A's circuit breaker counts it as a failure then a
        # traffic burst trips the breaker and keeps triage down for the
        # whole cooldown, long after B has drained. Retry-After is short
        # for the same reason — the queue clears in seconds, not minutes.
        return _error(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            error="at_capacity",
            detail="B is at its in-flight limit. Retry after the interval; this is not a B failure.",
            request_id=payload.request_id,
            retry_after=5,
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
    except RequestIdConflict as exc:
        triage_requests_total.labels(outcome="conflict").inc()
        triage_latency_seconds.observe(time.perf_counter() - started)
        log.warning("triage.request_id_conflict request_id=%s", payload.request_id)
        return _error(
            status_code=status.HTTP_409_CONFLICT,
            error="request_id_conflict",
            detail=f"{exc} Do not retry — allocate a new request_id.",
            request_id=payload.request_id,
        )
    except TimeoutError:
        # asyncio.timeout raises the builtin TimeoutError. Caught separately
        # from the clause below so A can tell "B ran out of time" from "B's
        # provider broke" — the first is worth one retry, the second wants
        # a backoff.
        triage_requests_total.labels(outcome="timeout").inc()
        triage_latency_seconds.observe(time.perf_counter() - started)
        log.warning(
            "triage.timeout request_id=%s budget_s=%s", payload.request_id, TRIAGE_TIMEOUT_SECONDS
        )
        return _error(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            error="timeout",
            detail=f"Triage exceeded its {TRIAGE_TIMEOUT_SECONDS:g}s budget. Retry via the outbox.",
            request_id=payload.request_id,
            retry_after=30,
        )
    except Exception:
        triage_requests_total.labels(outcome="error").inc()
        triage_latency_seconds.observe(time.perf_counter() - started)
        # log.exception, not log.error: without the traceback every failure
        # here is an indistinguishable 503 and the only diagnosis available
        # is a counter going up.
        log.exception("triage.failed request_id=%s", payload.request_id)
        return _error(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error="triage_failed",
            detail="Triage provider failed. Retry via the outbox.",
            request_id=payload.request_id,
            retry_after=30,
        )
    finally:
        inflight_triage.dec()
        await limiter.release()

    outcome = "refused" if result.status is TriageStatus.ungrounded else "success"
    triage_requests_total.labels(outcome=outcome).inc()
    triage_latency_seconds.observe(time.perf_counter() - started)
    response.headers["X-Request-Id"] = str(payload.request_id)
    log.info(
        "triage.completed request_id=%s status=%s category=%s confidence=%.2f "
        "replayed=%s latency_ms=%d",
        payload.request_id,
        result.status.value,
        result.category.value,
        result.confidence,
        result.replayed,
        result.latency_ms,
    )
    return result


@app.post(
    "/v1/feedback",
    response_model=FeedbackResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Missing or invalid API key."},
        404: {"model": ErrorResponse, "description": "No triage run for that request_id."},
    },
    dependencies=[Depends(require_api_key)],
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

