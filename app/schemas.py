"""Pydantic models for the /v1/triage contract.

These are the shared types between Service A and Service B. In the real
project they get generated from openapi.yaml with datamodel-code-generator
so neither service can drift from the contract.
"""

import os
from datetime import datetime
from enum import Enum
from typing import Any
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class Priority(str, Enum):
    p1 = "p1"
    p2 = "p2"
    p3 = "p3"
    p4 = "p4"


class Category(str, Enum):
    """The closed set of categories B is allowed to return.

    `other` is the deliberate escape hatch. A model with no honest option
    will pick the nearest-looking label and report high confidence for it,
    and a confidently wrong routing decision costs more than an explicit
    "I don't know" that A can send to the general queue. Adding a member
    here is a contract change: A routes on these values, so both services
    have to agree before a new one can be returned.
    """

    network_vpn = "network.vpn"
    access_password = "access.password"
    hardware_laptop = "hardware.laptop"
    finance_systems = "finance.systems"
    other = "other"


class TriageStatus(str, Enum):
    """Why the response looks the way it does.

    A routes on this before it looks at anything else:

    ok          B ran the full pipeline and grounded its answer in the KB.
    ungrounded  B ran fine but nothing in the KB supports a reply. Category
                is `other`, `suggested_reply` is null, and
                `clarifying_questions` is populated. Route to a human — this
                is a correct refusal, not a failure.
    degraded    B ran out of time or a dependency failed, and this is the
                partial result rather than nothing. Treat the classification
                as provisional and do not auto-route on it.
    """

    ok = "ok"
    ungrounded = "ungrounded"
    degraded = "degraded"


# Guards against a 200-message email thread. Per-message length is capped on
# Message.body; these cap the request as a whole, which is what actually
# determines the context window and the bill.
MAX_MESSAGES = int(os.getenv("MAX_MESSAGES_PER_TICKET", "50"))
MAX_TOTAL_CHARS = int(os.getenv("MAX_TOTAL_TICKET_CHARS", "60000"))


# ---------------------------------------------------------------- request


class Message(BaseModel):
    author_type: Literal["customer", "agent", "system"]
    body: str = Field(min_length=1, max_length=20_000)
    created_at: datetime


class Requester(BaseModel):
    """Non-identifying context only.

    Department and tenure genuinely help classification: a Finance ticket
    during month-end close is more urgent than the same ticket in week two.
    A person's name does not help at all, so it never crosses the boundary.
    """

    department: str
    region: str
    tenure_days: int = Field(ge=0)
    prior_ticket_count: int = Field(ge=0)


class TriageRequest(BaseModel):
    request_id: UUID
    ticket_ref: str  # for logging only; B never looks it up
    # 500 rather than the 200 in A's draft on purpose: the server should be
    # the more permissive of the two. A stricter limit on A's side inside a
    # looser one here means an over-long subject is caught before it leaves
    # A, where there is a ticket to attach the error to. The other way round,
    # a 250-character email subject becomes a ticket that cannot be triaged.
    subject: str = Field(min_length=1, max_length=500)
    messages: list[Message] = Field(min_length=1, max_length=MAX_MESSAGES)
    requester: Requester
    hints: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _within_total_size(self) -> "TriageRequest":
        # Rejected rather than truncated on purpose. Silently dropping the
        # tail of a thread changes the input without telling anyone, which
        # makes an eval result impossible to reproduce and a wrong answer
        # impossible to explain. A 422 is non-retryable, so A should split
        # or summarise the thread rather than send it again unchanged.
        total = len(self.subject) + sum(len(m.body) for m in self.messages)
        if total > MAX_TOTAL_CHARS:
            raise ValueError(
                f"ticket is {total} characters across {len(self.messages)} messages; "
                f"the limit is {MAX_TOTAL_CHARS}"
            )
        return self


# --------------------------------------------------------------- response


class Citation(BaseModel):
    chunk_id: str
    doc_title: str
    score: float = Field(ge=0.0, le=1.0)


class TriageResponse(BaseModel):
    request_id: UUID
    status: TriageStatus = TriageStatus.ok

    # B's own clock, both UTC. A asked for these and they are worth having:
    # with A's received_at alongside them, the gap between A receiving a
    # message and B starting work is measurable, which is the only way to
    # see queue delay in A's outbox. They also make the two services' clocks
    # reconcilable after the fact, which matters the first time a latency
    # number from one side disagrees with the other.
    received_at: datetime
    completed_at: datetime

    category: Category
    priority: Priority
    confidence: float = Field(ge=0.0, le=1.0)
    suggested_reply: str | None = None
    clarifying_questions: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    model: str
    prompt_version: str
    latency_ms: int

    # True when this response came from the idempotency log or the
    # near-duplicate cache rather than a fresh pipeline run. When it is
    # true, token_cost_usd is 0.0 because no model call was made — summing
    # the field across responses therefore gives real spend, not spend
    # multiplied by A's retry count.
    replayed: bool = False
    token_cost_usd: float

    def replay_as(self, request_id: UUID, received_at: datetime, completed_at: datetime) -> "TriageResponse":
        """This response's judgement, with this call's accounting."""
        data: dict[str, Any] = self.model_dump()
        data.update(
            request_id=request_id,
            received_at=received_at,
            completed_at=completed_at,
            replayed=True,
            token_cost_usd=0.0,
            latency_ms=int((completed_at - received_at).total_seconds() * 1000),
        )
        return TriageResponse.model_validate(data)


class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: UUID | None = None


# --------------------------------------------------------------- feedback


class FeedbackRequest(BaseModel):
    request_id: UUID
    verdict: Literal["correct", "incorrect"]
    corrected_category: Category | None = None
    corrected_priority: Priority | None = None


class FeedbackResponse(BaseModel):
    status: Literal["recorded"]
    run_id: UUID
