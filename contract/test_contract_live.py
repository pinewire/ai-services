"""Contract tests for Service B, run against a LIVE endpoint.

Two things are checked, and the distinction matters:

  1. Every response validates against the schema the service itself
     publishes at /openapi.json. This catches drift between the contract
     Service A generates its client from and what B actually returns —
     the failure mode where both sides are internally consistent and
     disagree with each other.

  2. The behavioural guarantees A depends on: idempotency, the conflict
     case, the error taxonomy, auth, and the refusal path. A schema cannot
     express "the same request_id returns the same verdict", so those are
     asserted directly.

Run it anywhere:

    export SERVICE_B_URL=http://localhost:8099
    export SERVICE_B_KEY=sb_...
    pytest test_contract_live.py -v

Against the VM over the tunnel:

    ssh -f -N -L 8081:127.0.0.1:30080 pinewire-vm
    export SERVICE_B_URL=http://localhost:8081
    pytest test_contract_live.py -v

Nothing here writes to the knowledge base or needs the repo on the path —
it talks HTTP only, so it works against any deployment.
"""

import os
import uuid

import httpx
import pytest
from jsonschema import Draft202012Validator

BASE_URL = os.environ.get("SERVICE_B_URL", "http://localhost:8099").rstrip("/")
API_KEY = os.environ.get("SERVICE_B_KEY", "")

TIMEOUT = httpx.Timeout(30.0, connect=5.0)


# --------------------------------------------------------------- fixtures


@pytest.fixture(scope="session")
def client() -> httpx.Client:
    headers = {"content-type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    with httpx.Client(base_url=BASE_URL, headers=headers, timeout=TIMEOUT) as c:
        yield c


@pytest.fixture(scope="session", autouse=True)
def _is_actually_service_b() -> None:
    """Fail once, clearly, when SERVICE_B_URL points somewhere else.

    Without this the whole suite fails 23 times with `{"detail":"Not Found"}`
    and `ErrorResponse not in spec`, which reads like a broken contract
    rather than a wrong URL. Pointing at another FastAPI app on a port you
    forgot was occupied is an easy mistake — uvicorn's "address already in
    use" scrolls past and the previous process keeps answering.
    """
    with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as probe:
        try:
            resp = probe.get("/openapi.json")
        except httpx.ConnectError:
            # Nothing listening at all: skip rather than abort. These tests
            # need a server by design, and a bare `pytest` that happens to
            # collect them should not fail the build over it.
            pytest.skip(
                f"no server at {BASE_URL}; set SERVICE_B_URL to a running "
                f"Service B to run the contract suite",
                allow_module_level=True,
            )
        if resp.status_code != 200:
            pytest.exit(f"no OpenAPI document at {BASE_URL}/openapi.json", returncode=1)
        doc = resp.json()
        if "/v1/triage" not in doc.get("paths", {}):
            pytest.exit(
                f"{BASE_URL} is not Service B.\n"
                f"  title: {doc.get('info', {}).get('title')!r}\n"
                f"  paths: {sorted(doc.get('paths', {}))}\n"
                f"Set SERVICE_B_URL to a Service B instance "
                f"(/healthz must return exactly {{\"status\":\"ok\"}}).",
                returncode=1,
            )


@pytest.fixture(scope="session")
def spec(client: httpx.Client) -> dict:
    """The contract as the running service publishes it."""
    resp = client.get("/openapi.json")
    assert resp.status_code == 200, f"no spec at {BASE_URL}/openapi.json"
    return resp.json()


def validator_for(spec: dict, schema_name: str) -> Draft202012Validator:
    """A validator for one component, with $refs resolving against the spec.

    Merging the spec into the schema document makes "#/components/..."
    resolvable, which is what lets a $ref-heavy OpenAPI document be used
    directly as a JSON Schema.
    """
    assert schema_name in spec["components"]["schemas"], f"{schema_name} not in spec"
    return Draft202012Validator({**spec, "$ref": f"#/components/schemas/{schema_name}"})


def check(spec: dict, schema_name: str, payload: dict) -> None:
    errors = sorted(validator_for(spec, schema_name).iter_errors(payload), key=str)
    assert not errors, "\n".join(
        f"{list(e.absolute_path) or '<root>'}: {e.message}" for e in errors
    )


# ---------------------------------------------------------------- payloads


def ticket(subject: str, body: str, request_id: str | None = None, nonce: str | None = None) -> dict:
    """Build a request whose content has not been seen before.

    The nonce is load-bearing. B has a SECOND cache beyond request_id
    idempotency: a near-duplicate cache keyed on a hash of the ticket
    content, which returns a stored answer for a *different* request_id
    when the text matches. Without a nonce these tests become
    order-dependent — the first call of a run is fresh, and every later
    run of the same suite gets `replayed: true` on what it believes is a
    first call.

    That is worth knowing on A's side too: two genuinely distinct tickets
    with identical text get byte-identical responses, citations included.
    It is intentional and it saves real money, but it reads as a bug the
    first time you see it.
    """
    marker = nonce or uuid.uuid4().hex[:8]
    return {
        "request_id": request_id or str(uuid.uuid4()),
        "ticket_ref": "TKT-CONTRACT",
        "subject": subject,
        "messages": [
            {
                "author_type": "customer",
                "body": f"{body} [case {marker}]",
                "created_at": "2026-11-20T18:00:00Z",
            }
        ],
        "requester": {
            "department": "Engineering",
            "region": "us-east",
            "tenure_days": 120,
            "prior_ticket_count": 2,
        },
    }


VPN = ("VPN keeps dropping after 30 seconds", "VPN disconnects repeatedly. Error VPN-4021.")
UNKNOWN = (
    "Request for a standing desk riser",
    "Can I get a desk riser? Occupational health suggested one.",
)


# ------------------------------------------------------- the spec itself


def test_spec_declares_the_api_key_scheme(spec: dict) -> None:
    """Without this, A generates a client with no auth and learns at runtime."""
    schemes = spec.get("components", {}).get("securitySchemes", {})
    assert any(
        s.get("in") == "header" and s.get("name", "").lower() == "x-api-key"
        for s in schemes.values()
    ), f"no X-API-Key header scheme in spec: {schemes}"


def test_spec_does_not_leak_test_hooks(spec: dict) -> None:
    """A test hook in the contract is one someone will build against."""
    params = spec["paths"]["/v1/triage"]["post"].get("parameters", [])
    names = {p["name"] for p in params}
    assert not (names & {"delay_ms", "fail"}), f"chaos params published: {names}"


def test_spec_documents_the_error_taxonomy(spec: dict) -> None:
    """A's retry policy and circuit breaker branch on these."""
    declared = set(spec["paths"]["/v1/triage"]["post"]["responses"])
    for code in ("200", "401", "422", "429", "503", "504"):
        assert code in declared, f"{code} undeclared; A cannot plan for it. have={declared}"


def test_category_is_a_closed_set(spec: dict) -> None:
    """A routes on these values, so an unlisted one has nowhere to go."""
    schema = spec["components"]["schemas"]
    assert "Category" in schema, "category is not an enum in the published contract"
    assert "other" in schema["Category"]["enum"], "no explicit refusal category"


# -------------------------------------------------------------- liveness


def test_probes_are_unauthenticated() -> None:
    """The kubelet cannot send a header."""
    with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as anon:
        assert anon.get("/healthz").status_code == 200
        assert anon.get("/readyz").status_code in (200, 503)


# ------------------------------------------------------------------ auth


@pytest.mark.skipif(not API_KEY, reason="SERVICE_B_KEY unset; auth not deployed here")
def test_missing_key_is_401() -> None:
    with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as anon:
        resp = anon.post("/v1/triage", json=ticket(*VPN))
    assert resp.status_code == 401


@pytest.mark.skipif(not API_KEY, reason="SERVICE_B_KEY unset; auth not deployed here")
def test_wrong_key_is_401() -> None:
    with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as anon:
        resp = anon.post(
            "/v1/triage", json=ticket(*VPN), headers={"X-API-Key": "definitely-wrong"}
        )
    assert resp.status_code == 401


@pytest.mark.skipif(not API_KEY, reason="SERVICE_B_KEY unset; auth not deployed here")
def test_metrics_requires_a_key(client: httpx.Client) -> None:
    """These series carry per-category volumes and cumulative model spend."""
    with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as anon:
        assert anon.get("/metrics").status_code == 401
    assert client.get("/metrics").status_code == 200


# ------------------------------------------------------------ happy path


def test_response_matches_the_published_schema(client: httpx.Client, spec: dict) -> None:
    resp = client.post("/v1/triage", json=ticket(*VPN))
    assert resp.status_code == 200, resp.text
    check(spec, "TriageResponse", resp.json())


def test_echoes_the_request_id(client: httpx.Client) -> None:
    """A matches the response to its outbox row by this."""
    payload = ticket(*VPN)
    resp = client.post("/v1/triage", json=payload)
    assert resp.status_code == 200, resp.text
    assert resp.json()["request_id"] == payload["request_id"]


def test_reports_which_model_and_prompt_produced_it(client: httpx.Client) -> None:
    """A stores these; without them no later analysis is possible."""
    body = client.post("/v1/triage", json=ticket(*VPN)).json()
    assert body["model"], "no model recorded"
    assert body["prompt_version"], "no prompt_version recorded"


def test_timestamps_are_offset_aware_and_ordered(client: httpx.Client) -> None:
    from datetime import datetime

    body = client.post("/v1/triage", json=ticket(*VPN)).json()
    received = datetime.fromisoformat(body["received_at"])
    completed = datetime.fromisoformat(body["completed_at"])
    assert received.tzinfo is not None, "naive timestamp; clocks cannot be reconciled"
    assert completed >= received


def test_suggested_reply_is_human_readable(client: httpx.Client) -> None:
    """Regression: an enum repr once reached the text a customer reads."""
    reply = client.post("/v1/triage", json=ticket(*VPN)).json().get("suggested_reply") or ""
    assert "Category." not in reply, reply
    assert "Priority." not in reply, reply


# ----------------------------------------------------------- idempotency


def test_replay_returns_the_same_verdict_and_charges_nothing(
    client: httpx.Client, spec: dict
) -> None:
    """Idempotency is what licenses A to retry at all.

    The verdict must be identical; the accounting must describe THIS call.
    A sums token_cost_usd, so a replay reporting the original cost would
    multiply reported spend by A's retry count.
    """
    payload = ticket(*VPN)
    first = client.post("/v1/triage", json=payload)
    second = client.post("/v1/triage", json=payload)
    assert first.status_code == 200 and second.status_code == 200, second.text

    a, b = first.json(), second.json()
    check(spec, "TriageResponse", b)

    verdict = ("status", "category", "priority", "confidence", "citations")
    assert {k: a[k] for k in verdict} == {k: b[k] for k in verdict}

    assert a.get("replayed") is False
    assert b.get("replayed") is True
    assert b["token_cost_usd"] == 0.0


def test_reused_request_id_with_new_content_is_409(client: httpx.Client) -> None:
    """Answering the old question would be silently, plausibly wrong."""
    rid = str(uuid.uuid4())
    first = client.post("/v1/triage", json=ticket(*VPN, request_id=rid))
    assert first.status_code == 200, first.text

    # A genuinely different ticket under the same key. The 409 must come
    # from the content differing, not from anything about the new text.
    clash = client.post(
        "/v1/triage",
        json=ticket("Payroll export failed", "Expense report will not submit.", request_id=rid),
    )
    assert clash.status_code == 409, clash.text
    assert "Retry-After" not in clash.headers, "409 must not invite a retry"


# -------------------------------------------------------- error taxonomy


def test_error_bodies_share_one_shape(client: httpx.Client, spec: dict) -> None:
    """A branches on `error`, so the field must exist on every failure."""
    rid = str(uuid.uuid4())
    client.post("/v1/triage", json=ticket(*VPN, request_id=rid))
    clash = client.post(
        "/v1/triage", json=ticket("Something else", "Unrelated text.", request_id=rid)
    )
    assert clash.status_code == 409
    check(spec, "ErrorResponse", clash.json())
    assert clash.json()["error"] == "request_id_conflict"


def test_a_shaped_payload_is_rejected_with_detail(client: httpx.Client) -> None:
    """The nesting disagreement, as a test rather than a surprise.

    Service A's draft contract nests under `ticket` / `requester_context`.
    If that ever starts returning 200, the two sides have converged and
    this test should be updated deliberately, not silently.
    """
    resp = client.post(
        "/v1/triage",
        json={
            "request_id": str(uuid.uuid4()),
            "received_at": "2026-11-20T18:00:00Z",
            "ticket": {"subject": VPN[0], "messages": [{"author_type": "customer", "body": VPN[1]}]},
            "requester_context": {
                "department": "Engineering",
                "region": "us-east",
                "tenure_days": 120,
                "prior_ticket_count": 2,
            },
        },
    )
    assert resp.status_code == 422, resp.text
    missing = {tuple(d["loc"])[-1] for d in resp.json()["detail"] if d["type"] == "missing"}
    assert {"ticket_ref", "subject", "messages", "requester"} <= missing, missing


def test_oversized_thread_is_rejected_not_truncated(client: httpx.Client) -> None:
    """Silent truncation changes the input without telling anyone."""
    payload = ticket(*VPN)
    payload["messages"] = [
        {"author_type": "customer", "body": "x" * 20_000, "created_at": "2026-11-20T18:00:00Z"}
        for _ in range(5)
    ]
    assert client.post("/v1/triage", json=payload).status_code == 422


def test_too_many_messages_is_rejected(client: httpx.Client) -> None:
    payload = ticket(*VPN)
    payload["messages"] = [
        {"author_type": "customer", "body": "short", "created_at": "2026-11-20T18:00:00Z"}
        for _ in range(200)
    ]
    assert client.post("/v1/triage", json=payload).status_code == 422


def test_chaos_params_are_inert(client: httpx.Client) -> None:
    """Reachable, they are a free denial of service."""
    resp = client.post("/v1/triage?fail=true&delay_ms=30000", json=ticket(*VPN))
    assert resp.status_code == 200, f"chaos params are live: {resp.status_code}"


# ----------------------------------------------------------- the refusal


def test_unknown_topic_refuses_with_200_not_5xx(client: httpx.Client, spec: dict) -> None:
    """A must tell 'nothing covers this' apart from 'B is broken'.

    They need different handling — one goes to a human with the clarifying
    questions, the other retries via the outbox — and a status code alone
    cannot carry the difference.
    """
    resp = client.post("/v1/triage", json=ticket(*UNKNOWN))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    check(spec, "TriageResponse", body)

    assert body["status"] == "ungrounded"
    assert body["category"] == "other"
    assert body["suggested_reply"] is None
    assert body["clarifying_questions"], "a refusal must tell the agent what to ask"
    assert body["citations"] == []


def test_citations_only_reference_retrieved_chunks(client: httpx.Client) -> None:
    """A chunk id the model never saw is confabulation."""
    body = client.post("/v1/triage", json=ticket(*VPN)).json()
    for citation in body["citations"]:
        assert citation["chunk_id"], "citation with no chunk_id"
        assert citation["doc_title"], "citation with no doc_title"
        assert 0.0 <= citation["score"] <= 1.0


# ------------------------------------------------------------- feedback


def test_feedback_rejects_an_unknown_request_id(client: httpx.Client, spec: dict) -> None:
    resp = client.post(
        "/v1/feedback",
        json={"request_id": str(uuid.uuid4()), "verdict": "incorrect"},
    )
    assert resp.status_code == 404, resp.text
    check(spec, "ErrorResponse", resp.json())


def test_feedback_records_a_correction(client: httpx.Client, spec: dict) -> None:
    """This is how the eval set grows, so it is part of the contract."""
    payload = ticket(*VPN)
    assert client.post("/v1/triage", json=payload).status_code == 200

    resp = client.post(
        "/v1/feedback",
        json={
            "request_id": payload["request_id"],
            "verdict": "incorrect",
            "corrected_category": "access.password",
            "corrected_priority": "p2",
        },
    )
    assert resp.status_code == 200, resp.text
    check(spec, "FeedbackResponse", resp.json())
