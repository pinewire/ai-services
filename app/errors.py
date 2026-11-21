"""Domain errors that map onto specific HTTP statuses.

The point of these is that `main.py` can tell Service A *which* kind of
failure happened. Service A's circuit breaker and outbox behave differently
for each, and a single 503 for everything forces A to guess:

  409  the caller contradicted itself — never retry, this is a bug in A
  429  B is at capacity and shed the load — retry, do NOT open the breaker
  504  B exceeded its own deadline — retry once, count toward the breaker
  503  a dependency failed — back off, count toward the breaker

Returning 503 for the 429 case is the one that bites hardest: load-shedding
is B working correctly, so counting it as a failure lets a traffic burst
trip A's breaker and keep triage down for the breaker's whole cooldown,
well after B has recovered.
"""


class TriageError(Exception):
    """Base for errors the triage endpoint translates into a status code."""


class RequestIdConflict(TriageError):
    """A known request_id came back with different ticket content.

    `request_id` is the idempotency key, so B's guarantee is "same id, same
    answer". If A reuses an id for a different ticket, honouring that
    guarantee means answering the wrong question — and silently, because the
    response looks perfectly well-formed. Failing loudly with a 409 turns a
    data-corruption bug in A into an error A can see.
    """

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        super().__init__(
            f"request_id {request_id} was already used for different ticket content"
        )
