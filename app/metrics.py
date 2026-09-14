"""Prometheus metrics for the triage service.

All labels are bounded values. Request IDs, ticket references, prompts, and
model output must never become metric labels because that would create
unbounded time-series cardinality.
"""

from prometheus_client import Counter, Gauge, Histogram

triage_requests_total = Counter(
    "service_b_triage_requests_total",
    "Total triage requests by outcome.",
    ("outcome",),
)
triage_latency_seconds = Histogram(
    "service_b_triage_latency_seconds",
    "End-to-end triage latency in seconds.",
    buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)
triage_confidence = Histogram(
    "service_b_triage_confidence",
    "Composed triage confidence score.",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
triage_refusals_total = Counter(
    "service_b_triage_refusals_total",
    "Triage responses with category other and no grounded reply.",
)
triage_overloaded_total = Counter(
    "service_b_triage_overloaded_total",
    "Triage requests rejected by backpressure.",
)
triage_cache_hits_total = Counter(
    "service_b_triage_cache_hits_total",
    "Triage cache hits by cache type.",
    ("cache",),
)
retrieval_chunks_returned = Histogram(
    "service_b_retrieval_chunks_returned",
    "Number of fused chunks returned to the model.",
    buckets=(0, 1, 2, 3, 4, 5),
)
llm_tokens_total = Counter(
    "service_b_llm_tokens_total",
    "LLM tokens consumed by token type.",
    ("type",),
)
llm_cost_usd_total = Counter(
    "service_b_llm_cost_usd_total",
    "Estimated LLM cost in USD.",
)
inflight_triage = Gauge(
    "service_b_triage_inflight",
    "Current number of triage requests inside the pipeline.",
)