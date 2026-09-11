"""Model-call abstraction.

`TriageModel.classify()` must return a validated, schema-shaped output; a
Pydantic ValidationError is what triggers one reject-and-retry, not a
try/except around free text. `RuleBasedTriageModel` is a zero-dependency
stand-in until a real LLM is wired in — swap it in `get_triage_model` behind
an env var the same way `app.embeddings` does for embeddings.
"""

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


def _refusal() -> ModelOutput:
    return ModelOutput(
        category="other",
        priority=Priority.p3,
        suggested_reply=None,
        clarifying_questions=_REFUSAL_QUESTIONS,
        cited_chunk_ids=[],
    )


class TriageModel:
    async def classify(self, ticket_text: str, chunks: list[RetrievedChunk]) -> ModelOutput:
        raise NotImplementedError


class RuleBasedTriageModel(TriageModel):
    async def classify(self, ticket_text: str, chunks: list[RetrievedChunk]) -> ModelOutput:
        last_error: ValidationError | None = None
        for _ in range(2):  # one reject-and-retry on schema validation failure
            try:
                return self._run(ticket_text, chunks)
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


def get_triage_model() -> TriageModel:
    return RuleBasedTriageModel()
