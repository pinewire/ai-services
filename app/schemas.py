"""Pydantic models for the /v1/triage contract.

These are the shared types between Service A and Service B. In the real
project they get generated from openapi.yaml with datamodel-code-generator
so neither service can drift from the contract.
"""

from datetime import datetime
from enum import Enum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class Priority(str, Enum):
    p1 = "p1"
    p2 = "p2"
    p3 = "p3"
    p4 = "p4"


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
    subject: str
    messages: list[Message] = Field(min_length=1)
    requester: Requester
    hints: dict[str, str] = Field(default_factory=dict)


# --------------------------------------------------------------- response


class Citation(BaseModel):
    chunk_id: str
    doc_title: str
    score: float = Field(ge=0.0, le=1.0)


class TriageResponse(BaseModel):
    request_id: UUID
    category: str
    priority: Priority
    confidence: float = Field(ge=0.0, le=1.0)
    suggested_reply: str | None = None
    clarifying_questions: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    model: str
    prompt_version: str
    latency_ms: int
    token_cost_usd: float


class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: UUID | None = None
