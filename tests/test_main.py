"""Smoke tests for the triage API — run against a real Postgres in CI."""

import os
import uuid

os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://b_user:b_pass@localhost:5432/service_b"
)

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


def _payload() -> dict:
    return {
        "request_id": str(uuid.uuid4()),
        "ticket_ref": "TKT-TEST",
        "subject": "VPN drops after about 30 seconds",
        "messages": [
            {
                "author_type": "customer",
                "body": "VPN connects then disconnects.",
                "created_at": "2026-09-04T14:22:03Z",
            }
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


def test_triage_known_category() -> None:
    with TestClient(app) as client:
        resp = client.post("/v1/triage", json=_payload())
    body = resp.json()
    assert resp.status_code == 200
    assert body["category"] == "network.vpn"
    assert body["citations"]


def test_triage_fail_param_returns_503() -> None:
    with TestClient(app) as client:
        resp = client.post("/v1/triage?fail=true", json=_payload())
    assert resp.status_code == 503
    assert resp.headers["retry-after"] == "30"
