"""Model-call abstraction.

`TriageModel.classify()` must return a validated, schema-shaped output; a
Pydantic validation failure triggers one reject-and-retry. Claude Sonnet is
the real provider, while `RuleBasedTriageModel` remains the zero-dependency
offline fallback selected by `LLM_PROVIDER`.
"""

import json
import os
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from app.retrieval import RetrievedChunk
from app.schemas import Priority

_REFUSAL_QUESTIONS = [
    "What exactly happens when you try — any error message on screen?",
    "When did this start, and did anything change just before?",
    "Does it happen on every attempt or only sometimes?",
]

_CATEGORY_RULES: list[tuple[tuple[str, ...], str, Priority]] = [
    (("vpn", "network", "wifi", "connection"), "network.vpn", Priority.p2),
    (("password", "login", "locked out", "mfa"), "access.password", Priority.p3),
    (("laptop", "screen", "keyboard", "battery"), "hardware.laptop", Priority.p3),
    (("payroll", "invoice", "expense"), "finance.systems", Priority.p2),
]


class ModelOutput(BaseModel):
    """The forced-JSON shape the model must return."""

    category: str
    priority: Priority
    suggested_reply: str | None
    clarifying_questions: list[str]
    cited_chunk_ids: list[str]


@dataclass
class ModelCallResult:
    output: ModelOutput
    prompt_tokens: int = 0
    completion_tokens: int = 0
    token_cost_usd: float = 0.0


def _refusal() -> ModelOutput:
    return ModelOutput(
        category="other",
        priority=Priority.p3,
        suggested_reply=None,
        clarifying_questions=_REFUSAL_QUESTIONS,
        cited_chunk_ids=[],
    )


class TriageModel:
    @property
    def model_name(self) -> str:
        return "unknown"

    async def classify(self, ticket_text: str, chunks: list[RetrievedChunk]) -> ModelCallResult:
        raise NotImplementedError


class RuleBasedTriageModel(TriageModel):
    @property
    def model_name(self) -> str:
        return "rule-based-v1"

    async def classify(self, ticket_text: str, chunks: list[RetrievedChunk]) -> ModelCallResult:
        last_error: ValidationError | None = None
        for _ in range(2):  # one reject-and-retry on schema validation failure
            try:
                return ModelCallResult(output=self._run(ticket_text, chunks))
            except ValidationError as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    def _run(self, ticket_text: str, chunks: list[RetrievedChunk]) -> ModelOutput:
        if not chunks:
            # Nothing grounds a reply — refuse rather than invent one.
            return _refusal()

        lowered = ticket_text.lower()
        for keywords, category, priority in _CATEGORY_RULES:
            if any(k in lowered for k in keywords):
                top = chunks[0]
                return ModelOutput(
                    category=category,
                    priority=priority,
                    suggested_reply=(
                        f"Thanks for reporting this. This looks like a known {category} issue. "
                        f"Per {top.doc_title}: {top.content[:200].strip()}"
                    ),
                    clarifying_questions=[],
                    cited_chunk_ids=[str(top.chunk_id)],
                )

        return _refusal()


class AnthropicTriageModel(TriageModel):
    """Async Claude model with Pydantic-validated JSON output."""

    def __init__(self) -> None:
        from anthropic import AsyncAnthropic

        self._client = AsyncAnthropic()
        self._model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    @property
    def model_name(self) -> str:
        return self._model

    @staticmethod
    def _cost(prompt_tokens: int, completion_tokens: int) -> float:
        input_rate = float(os.getenv("ANTHROPIC_INPUT_COST_PER_1M", "3.00"))
        output_rate = float(os.getenv("ANTHROPIC_OUTPUT_COST_PER_1M", "15.00"))
        return (prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000

    async def classify(self, ticket_text: str, chunks: list[RetrievedChunk]) -> ModelCallResult:
        context = "\n\n".join(
            f"CHUNK_ID: {chunk.chunk_id}\nTITLE: {chunk.doc_title}\nCONTENT: {chunk.content}"
            for chunk in chunks
        )
        system = (
            "You are a helpdesk triage engine. Classify the ticket and draft a reply "
            "using only the supplied knowledge-base passages. Return ONLY valid JSON "
            "matching this shape: {\"category\": string, \"priority\": \"p1\"|\"p2\"|\"p3\"|\"p4\", "
            "\"suggested_reply\": string|null, \"clarifying_questions\": string[], "
            "\"cited_chunk_ids\": string[]}. If the passages do not support a grounded answer, "
            "return category 'other', suggested_reply null, clarifying questions, and no "
            "cited_chunk_ids. cited_chunk_ids must contain only exact CHUNK_ID values from "
            "the passages."
        )
        user_message = f"TICKET:\n{ticket_text}\n\nKNOWLEDGE BASE:\n{context or '(no relevant passages)'}"

        last_error: Exception | None = None
        for _ in range(2):
            try:
                message = await self._client.messages.create(
                    model=self._model,
                    system=system,
                    messages=[{"role": "user", "content": user_message}],
                    max_tokens=1000,
                    temperature=0,
                )
                text_blocks = [block.text for block in message.content if block.type == "text"]
                if not text_blocks:
                    raise ValueError("Claude returned no text content")
                raw = text_blocks[0].strip()
                if raw.startswith("```"):
                    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                parsed = ModelOutput.model_validate(json.loads(raw))
                return ModelCallResult(
                    output=parsed,
                    prompt_tokens=message.usage.input_tokens,
                    completion_tokens=message.usage.output_tokens,
                    token_cost_usd=self._cost(message.usage.input_tokens, message.usage.output_tokens),
                )
            except Exception as exc:
                last_error = exc

        assert last_error is not None
        raise RuntimeError("Claude triage failed after one retry") from last_error


def get_triage_model() -> TriageModel:
    provider = os.getenv("LLM_PROVIDER", "rule-based")
    if provider == "anthropic":
        return AnthropicTriageModel()
    if provider == "rule-based":
        return RuleBasedTriageModel()
    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")
