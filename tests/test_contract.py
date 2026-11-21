"""Contract guarantees Service A depends on.

Each of these encodes a promise made in the OpenAPI document, so a change
that breaks one is a contract change and should fail here before it reaches
A's integration tests. Needs a real Postgres, like test_main.py.
"""

import os
import uuid

os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://b_user:b_pass@localhost:5433/service_b"
)

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.schemas import MAX_MESSAGES, MAX_TOTAL_CHARS, Category, TriageStatus  # noqa: E402


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


def test_same_request_id_same_content_is_a_free_replay() -> None:
    """Idempotency is what licenses A to retry at all.

    The second call must not bill: A sums token_cost_usd across responses,
    so a replay that reported the original cost would multiply reported
    spend by A's retry count.
    """
    rid = str(uuid.uuid4())
    payload = _payload("VPN drops after about 30 seconds", "VPN connects then drops. VPN-4021.", rid)

    with TestClient(app) as client:
        first = client.post("/v1/triage", json=payload)
        second = client.post("/v1/triage", json=payload)

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text

    assert first.json()["replayed"] is False
    assert second.json()["replayed"] is True
    assert second.json()["token_cost_usd"] == 0.0
    # The answer itself must be identical — same category, same citations.
    assert second.json()["category"] == first.json()["category"]
    assert second.json()["citations"] == first.json()["citations"]


def test_reused_request_id_with_different_content_is_409() -> None:
    """Answering the old question here would be silently, plausibly wrong."""
    rid = str(uuid.uuid4())
    with TestClient(app) as client:
        first = client.post("/v1/triage", json=_payload("VPN drops", "VPN-4021 disconnect.", rid))
        clash = client.post(
            "/v1/triage",
            json=_payload("Payroll export failed", "Expense report will not submit.", rid),
        )

    assert first.status_code == 200, first.text
    assert clash.status_code == 409, clash.text
    assert clash.json()["error"] == "request_id_conflict"
    # No Retry-After: retrying cannot help, A has to allocate a new id.
    assert "Retry-After" not in clash.headers


def test_oversized_thread_is_rejected_not_truncated() -> None:
    """Truncating would change the input without telling anyone."""
    body = "x" * 20_000
    payload = _payload("Long thread", "seed")
    payload["messages"] = [
        {"author_type": "customer", "body": body, "created_at": "2026-09-04T14:22:03Z"}
        for _ in range(5)
    ]
    assert 5 * len(body) > MAX_TOTAL_CHARS

    with TestClient(app) as client:
        resp = client.post("/v1/triage", json=payload)

    assert resp.status_code == 422, resp.text


def test_too_many_messages_is_rejected() -> None:
    payload = _payload("Very long thread", "seed")
    payload["messages"] = [
        {"author_type": "customer", "body": "short", "created_at": "2026-09-04T14:22:03Z"}
        for _ in range(MAX_MESSAGES + 1)
    ]

    with TestClient(app) as client:
        resp = client.post("/v1/triage", json=payload)

    assert resp.status_code == 422, resp.text


def test_category_is_always_in_the_closed_set() -> None:
    """A routes on these values, so an unknown one has nowhere to go."""
    allowed = {c.value for c in Category}
    with TestClient(app) as client:
        resp = client.post(
            "/v1/triage",
            json=_payload("VPN drops after about 30 seconds", "VPN connects then drops. VPN-4021."),
        )
    assert resp.status_code == 200, resp.text
    assert resp.json()["category"] in allowed


def test_unknown_topic_is_ungrounded_not_an_error() -> None:
    """A correct refusal is a 200 with status=ungrounded, never a 5xx.

    A has to tell "nothing in the KB covers this, send it to a human" apart
    from "B is broken, retry later", and a status code alone cannot say it.
    """
    with TestClient(app) as client:
        resp = client.post(
            "/v1/triage",
            json=_payload(
                "Request for a standing desk riser",
                "Can I get a desk riser for my back? Occupational health suggested one.",
            ),
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == TriageStatus.ungrounded.value
    assert body["category"] == Category.other.value
    assert body["suggested_reply"] is None
    assert body["clarifying_questions"], "a refusal must tell the agent what to ask"


def test_chaos_params_are_inert_by_default() -> None:
    """They are a free denial-of-service wherever they are reachable."""
    payload = _payload("VPN drops", "VPN-4021 disconnect.")
    with TestClient(app) as client:
        resp = client.post("/v1/triage?fail=true&delay_ms=30000", json=payload)

    # Not 429 and not slow: both parameters were ignored.
    assert resp.status_code == 200, resp.text


def test_chaos_params_are_absent_from_the_published_contract() -> None:
    """A's client is generated from this document."""
    params = app.openapi()["paths"]["/v1/triage"]["post"].get("parameters", [])
    names = {p["name"] for p in params}
    assert "delay_ms" not in names
    assert "fail" not in names


def test_contract_advertises_the_api_key_scheme() -> None:
    """Otherwise A discovers the header the first time it gets a 401."""
    schemes = app.openapi()["components"].get("securitySchemes", {})
    assert any(s.get("name") == "X-API-Key" for s in schemes.values()), schemes
