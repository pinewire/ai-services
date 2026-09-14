"""Smoke tests for the triage pipeline — run against a real Postgres in CI.
KB seeding happens once per session in conftest.py.
"""

import os
import uuid

import pytest

os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://b_user:b_pass@localhost:5433/service_b"
)

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.llm import get_triage_model  # noqa: E402


def _payload(subject: str, body: str, request_id: str | None = None) -> dict:
    return {
        "request_id": request_id or str(uuid.uuid4()),
        "ticket_ref": "TKT-TEST",
        "subject": subject,
        "messages": [
            {"author_type": "customer", "body": body, "created_at": "2026-09-04T14:22:03Z"}
        ],
        "requester": {
            "department": "Finance",
            "region": "us-east",
            "tenure_days": 412,
            "prior_ticket_count": 3,
        },
    }


def test_healthz() -> None:
    with TestClient(app) as client:
        resp = client.get("/healthz")
    assert resp.status_code == 200


def test_readyz_reports_db_connectivity() -> None:
    with TestClient(app) as client:
        resp = client.get("/readyz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"


def test_metrics_exposes_prometheus_data() -> None:
    with TestClient(app) as client:
        resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "service_b_triage_requests_total" in resp.text


def test_triage_grounds_reply_in_retrieved_chunk() -> None:
    payload = _payload("VPN drops after about 30 seconds", "VPN connects then disconnects. Error VPN-4021.")
    with TestClient(app) as client:
        resp = client.post("/v1/triage", json=payload)
    body = resp.json()
    assert resp.status_code == 200
    assert body["category"] == "network.vpn"
    assert body["citations"], "expected a citation grounded in the seeded KB"
    assert body["suggested_reply"] is not None
    assert body["confidence"] > 0.3


def test_triage_refuses_when_nothing_relevant_is_retrieved() -> None:
    payload = _payload("Office plant needs watering", "The ficus by reception looks droopy, can someone water it")
    with TestClient(app) as client:
        resp = client.post("/v1/triage", json=payload)
    body = resp.json()
    assert resp.status_code == 200
    assert body["category"] == "other"
    assert body["suggested_reply"] is None
    assert body["citations"] == []
    assert body["clarifying_questions"]
    assert body["confidence"] == 0.3


def test_triage_is_idempotent_on_request_id() -> None:
    request_id = str(uuid.uuid4())
    payload = _payload("VPN drops after about 30 seconds", "Error VPN-4021 again.", request_id=request_id)
    with TestClient(app) as client:
        first = client.post("/v1/triage", json=payload).json()
        second = client.post("/v1/triage", json=payload).json()
    assert first == second


def test_triage_fail_param_returns_503() -> None:
    request_id = str(uuid.uuid4())
    with TestClient(app) as client:
        resp = client.post(
            "/v1/triage?fail=true",
            json=_payload("VPN issue", "vpn drops", request_id=request_id),
        )
    assert resp.status_code == 503
    assert resp.headers["retry-after"] == "30"
    assert resp.headers["x-request-id"] == request_id


def test_unknown_llm_provider_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "unknown-provider")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        get_triage_model()


def test_feedback_links_to_run() -> None:
    request_id = str(uuid.uuid4())
    payload = _payload("VPN drops", "Error VPN-4021.", request_id=request_id)
    with TestClient(app) as client:
        client.post("/v1/triage", json=payload)
        resp = client.post(
            "/v1/feedback",
            json={"request_id": request_id, "verdict": "correct"},
        )
    assert resp.status_code == 200
    assert resp.json()["status"] == "recorded"


def test_feedback_unknown_request_id_is_404() -> None:
    with TestClient(app) as client:
        resp = client.post(
            "/v1/feedback",
            json={"request_id": str(uuid.uuid4()), "verdict": "correct"},
        )
    assert resp.status_code == 404

