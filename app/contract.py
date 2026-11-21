"""The A↔B contract, emitted from this service's own code.

Why this file exists
--------------------
A contract can be written by hand or generated from the provider. Written
by hand it drifts, because nothing fails when the code and the document
disagree — Service A's first draft was written from two Confluence pages
and disagreed with this service on nine points, including the shape of the
request body, so its own example payload could not have worked.

Generated, the schemas cannot drift: they come from the Pydantic models
that actually validate the requests. But a purely generated document has
no prose, no policy and no examples, and those are the parts a consumer
most needs — the retry rules, what `confidence` means, which failures to
count against a circuit breaker.

So: the schemas are generated, and the policy that cannot be inferred from
types is written here and injected into the same document. One artifact,
one source of truth, and `contract/test_contract_live.py` asserts that a
running instance still matches it.

Regenerate with:
    python -m app.contract > contract/triage-contract.yaml
"""

from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

# Not a placeholder: this is the Service's DNS name in the cluster, which is
# <service>.<namespace>.svc.cluster.local on port 80 (the Service's port, not
# the container's 8000).
IN_CLUSTER_URL = "http://service-b.service-b.svc.cluster.local"

CONTRACT_RULES = """
Contract between **Service A** (ticket intake, the caller) and **Service B**
(triage, the provider).

This document is generated from Service B's code. The schemas come from the
Pydantic models that validate real requests, so they cannot drift from the
implementation. Do not hand-edit it — change the code and regenerate.

## Bounded contexts

Service A owns the system of record. It must be correct and available.
Service B owns judgement. It is allowed to be wrong, slow, or entirely
unavailable, and that permission is what lets A's guarantees not depend on
B's. A must never fail a ticket create because of B.

A sends a projection of the ticket. No name, email or internal identifier
crosses the boundary — department and tenure help classification, a person's
name does not.

## Idempotency, and the two caches

`request_id` is the idempotency key, generated once by A when the ticket is
created and stored on the outbox row, so every retry reuses it.

* **Same `request_id`, same content** → the stored verdict, with
  `replayed: true` and `token_cost_usd: 0.0`. The judgement fields are
  byte-identical; `received_at`, `completed_at` and `latency_ms` describe
  *this* call. A can sum `token_cost_usd` across responses to get real
  spend — a replay reporting the original cost would multiply it by A's
  retry count.
* **Same `request_id`, different content** → `409`. Honouring the
  idempotency guarantee here would mean confidently answering a ticket
  nobody asked about, in a response that looks perfectly well-formed. Do
  not retry a 409; allocate a new `request_id`.
* **Different `request_id`, identical content** → also a replay. B keeps a
  second cache keyed on a hash of the ticket text, so two genuinely distinct
  tickets with the same words get byte-identical responses, citations
  included. This is deliberate and saves real inference cost. It reads as a
  bug the first time you see it.

## Error taxonomy

A's retry policy and circuit breaker branch on these. The distinction that
matters most is 429 versus 503.

| Status | `error`                | Retry?             | Count toward A's breaker? |
| ------ | ---------------------- | ------------------ | ------------------------- |
| 401    | —                      | No                 | No — alert instead        |
| 409    | `request_id_conflict`  | No, new id         | No — a bug in A           |
| 422    | —                      | No                 | No — a bug in A           |
| 429    | `at_capacity`          | Yes, `Retry-After` | **No**                    |
| 503    | `triage_failed`        | Yes, back off      | Yes                       |
| 504    | `timeout`              | Once               | Yes                       |

A 429 is B working correctly: it shed load deliberately and is healthy. A's
breaker opens after five consecutive failures and half-opens after 30
seconds, so counting load-shedding as failure lets a traffic burst trip the
breaker and hold triage down for the whole cooldown, long after B has
drained. `Retry-After` on a 429 is short (seconds) for the same reason.

B never returns 500.

## Timeouts, and the retry interval

B enforces its own deadline (`TRIAGE_TIMEOUT_SECONDS`, 8s by default) and
returns 504 rather than hanging. Keep A's patient timeout *longer* than
that: a 504 carries `prompt_version` and a request id, a socket timeout
carries nothing.

The 800ms and 2500ms figures in A's and B's design pages are both right,
for different callers. A's synchronous attempt during ticket creation is
opportunistic and should fail fast — 800ms–1s — falling through to the
outbox. A's outbox worker has nobody waiting and should be patient — 10s.

**A's outbox retry interval must exceed B's deadline.** If A gives up at
800ms while B is still working, and the outbox retries before B commits, B
starts a second full run for the same `request_id`. Both finish, one loses
the race on the unique index, and A sees a failure for work that succeeded.
With an 8s deadline the first retry should be no sooner than ~10s.

## What `confidence` is, and is not

`confidence` is composed from retrieval signals: the top chunk's similarity,
the margin to the next, and whether the vector and keyword retrievers agreed
on it. A value at or below 0.3 means citation validation found nothing to
ground an answer in.

**It measures retrieval quality, not classification correctness.** A
well-grounded answer in a well-covered area scores high even when the
category is wrong, and a correct category in a thin area scores low — a
correctly-classified `network.vpn` ticket measured 0.19 in testing.

So do not route on a numeric threshold. The scale is an artifact of B's
retrieval internals and will move when the embedder, the reranker or the
weights change; a threshold in A's code would be silently re-tuned by every
such change. Route on `status` instead:

* `ok` — B grounded its answer in the knowledge base.
* `ungrounded` — B ran fine and nothing supports a reply. `category` is
  `other`, `suggested_reply` is null, `clarifying_questions` is populated.
  This is a **correct refusal, not a failure**: send it to a human with the
  questions. It is a 200, and A must tell it apart from a 5xx.
* `degraded` — B returned a partial result rather than nothing. Treat the
  classification as provisional and do not auto-route on it.

## Citations

`citations[].score` is a reciprocal-rank-fusion value, not a relevance
percentage. First place is 1/61 ≈ 0.0164. Do not render it as a percentage.

## Authentication

Every callable endpoint requires `X-API-Key`. Service A has its own key, so
its traffic is distinguishable from test clients. B accepts a comma-separated
list, so rotation is: deploy both keys, move callers, deploy one.

The probes (`/healthz`, `/readyz`) are deliberately unauthenticated — the
kubelet cannot send a header.

## Versioning

Both services deploy independently; they are never in lockstep. Every
response carries `model` and `prompt_version`, and A should persist them
alongside the ticket — without them, "did accuracy change when we changed
the prompt" is unanswerable because the evidence was never recorded.

A breaking change to this contract rolls out in two steps: B accepts both
shapes, A moves, B drops the old shape.
""".strip()

# Service A's draft raised six open questions. Answering them in the
# generated document means the answer cannot drift from the code either.
CONTRACT_DECISIONS = [
    "request_id: UUID v4, 36 characters with hyphens. Confirmed.",
    "priority: lowercase p1-p4, not P1-P4.",
    (
        "category: a closed enum — network.vpn, access.password, "
        "hardware.laptop, finance.systems, other. Adding a member is a "
        "contract change. 'network' alone is not valid."
    ),
    (
        "citations: an array of objects {chunk_id, doc_title, score}, not an "
        "array of strings. chunk_id is a UUID, not 'kb-chunk-1042'."
    ),
    "received_at / completed_at: implemented, both offset-aware UTC.",
    (
        "cached: named 'replayed', because B has two caches and the flag "
        "covers both. See the idempotency section."
    ),
    (
        "timeout: B's deadline is 8s and returns 504. A uses ~1s on the "
        "synchronous path and 10s in the outbox worker. See the timeout "
        "section."
    ),
    (
        "split classify/draft: agreed in principle, not yet implemented. "
        "Blocked on deciding whether /v1/draft takes a request_id and "
        "reuses the persisted chunk ids, or re-retrieves from raw text."
    ),
]

_REQUEST_EXAMPLE = {
    "request_id": "1fd016de-d411-4977-844c-4c2651adfecd",
    "ticket_ref": "TKT-10432",
    "subject": "VPN keeps dropping after 30 seconds",
    "messages": [
        {
            "author_type": "customer",
            "body": "My VPN disconnects every 30 seconds since this morning. Error VPN-4021.",
            "created_at": "2026-11-16T22:10:04.100Z",
        }
    ],
    "requester": {
        "department": "Engineering",
        "region": "us-east",
        "tenure_days": 120,
        "prior_ticket_count": 2,
    },
}

# Captured from a running instance rather than written by hand, so they
# cannot describe a response the service would never produce.
_RESPONSE_EXAMPLES = {
    "answered": {
        "summary": "Grounded answer",
        "value": {
            "request_id": "1fd016de-d411-4977-844c-4c2651adfecd",
            "status": "ok",
            "received_at": "2026-11-21T17:12:37.710529Z",
            "completed_at": "2026-11-21T17:12:37.736797Z",
            "category": "network.vpn",
            "priority": "p2",
            "confidence": 0.19,
            "suggested_reply": "Thanks for reporting this. This looks like a known network.vpn issue...",
            "clarifying_questions": [],
            "citations": [
                {
                    "chunk_id": "cea73949-7027-4fd6-b59f-dc8666e95774",
                    "doc_title": "network",
                    "score": 0.0164,
                }
            ],
            "model": "rule-based-v1",
            "prompt_version": "pipeline-v1",
            "latency_ms": 26,
            "replayed": False,
            "token_cost_usd": 0.0,
        },
    },
    "replayed": {
        "summary": "Replay — identical verdict, zero cost, this call's clocks",
        "value": {
            "request_id": "1fd016de-d411-4977-844c-4c2651adfecd",
            "status": "ok",
            "received_at": "2026-11-21T17:12:37.740112Z",
            "completed_at": "2026-11-21T17:12:37.741903Z",
            "category": "network.vpn",
            "priority": "p2",
            "confidence": 0.19,
            "suggested_reply": "Thanks for reporting this. This looks like a known network.vpn issue...",
            "clarifying_questions": [],
            "citations": [
                {
                    "chunk_id": "cea73949-7027-4fd6-b59f-dc8666e95774",
                    "doc_title": "network",
                    "score": 0.0164,
                }
            ],
            "model": "rule-based-v1",
            "prompt_version": "pipeline-v1",
            "latency_ms": 1,
            "replayed": True,
            "token_cost_usd": 0.0,
        },
    },
    "ungrounded": {
        "summary": "Correct refusal — a 200, not a failure. Route to a human.",
        "value": {
            "request_id": "513c5354-d7f1-4d6d-8362-f8cbb608403b",
            "status": "ungrounded",
            "received_at": "2026-11-21T17:12:37.744888Z",
            "completed_at": "2026-11-21T17:12:37.747541Z",
            "category": "other",
            "priority": "p3",
            "confidence": 0.3,
            "suggested_reply": None,
            "clarifying_questions": [
                "What exactly happens when you try — any error message on screen?",
                "When did this start, and did anything change just before?",
                "Does it happen on every attempt or only sometimes?",
            ],
            "citations": [],
            "model": "rule-based-v1",
            "prompt_version": "pipeline-v1",
            "latency_ms": 2,
            "replayed": False,
            "token_cost_usd": 0.0,
        },
    },
}

_RETRY_AFTER_HEADER = {
    "Retry-After": {
        "description": "Seconds to wait before retrying.",
        "schema": {"type": "integer", "minimum": 1},
    }
}


def _enrich(schema: dict[str, Any]) -> dict[str, Any]:
    schema["info"]["description"] = CONTRACT_RULES
    schema["info"]["x-contract-decisions"] = CONTRACT_DECISIONS
    schema["servers"] = [
        {"url": IN_CLUSTER_URL, "description": "In-cluster (k3s, namespace service-b)"},
        {"url": "http://localhost:8000", "description": "Local development"},
    ]

    triage = schema["paths"]["/v1/triage"]["post"]
    triage["x-timeout-ms"] = 8000
    triage["x-client-timeout-ms"] = {"synchronous": 1000, "outbox-worker": 10000}

    body = triage["requestBody"]["content"]["application/json"]
    body["example"] = _REQUEST_EXAMPLE

    ok = triage["responses"]["200"]["content"]["application/json"]
    ok["examples"] = _RESPONSE_EXAMPLES

    # Retry-After is only meaningful where retrying can help. Its absence on
    # 409 is part of the contract, not an omission.
    for code in ("429", "503", "504"):
        if code in triage["responses"]:
            triage["responses"][code]["headers"] = _RETRY_AFTER_HEADER

    return schema


def customize_openapi(app: FastAPI) -> None:
    """Replace app.openapi so the generated document carries the contract."""

    def _openapi() -> dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema
        app.openapi_schema = _enrich(
            get_openapi(
                title=app.title,
                version=app.version,
                routes=app.routes,
                summary="Classifies helpdesk tickets and drafts grounded replies.",
            )
        )
        return app.openapi_schema

    app.openapi = _openapi  # type: ignore[method-assign]


def main() -> None:
    """Print the contract as YAML, for committing to contract/."""
    import os

    import yaml

    # The document is built from route definitions and Pydantic models, so
    # nothing here touches the database or any provider.
    os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://u:p@127.0.0.1:5432/x")

    from app.main import app

    class _Dumper(yaml.SafeDumper):
        """Keeps the prose readable: a human reads this file."""

    def _str(dumper: yaml.SafeDumper, value: str):
        # Literal block style for anything multi-line, so the markdown
        # description stays markdown instead of one escaped line.
        style = "|" if "\n" in value else None
        return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)

    _Dumper.add_representer(str, _str)

    print(
        yaml.dump(
            app.openapi(),
            Dumper=_Dumper,
            sort_keys=False,
            width=100,
            allow_unicode=True,
            default_flow_style=False,
        )
    )


if __name__ == "__main__":
    main()
