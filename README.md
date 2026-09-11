# Service B — triage engine

Async FastAPI service that classifies helpdesk tickets and drafts grounded replies using hybrid retrieval over company runbooks.

## Run locally

    docker compose up --build

App: http://localhost:8001
Interactive docs: http://localhost:8001/docs

For offline development:

    LLM_PROVIDER=rule-based EMBEDDING_PROVIDER=hashing docker compose up -d --build

For real providers, configure `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` in the ignored `.env` file, then recreate the service.

## Knowledge base

`app/ingest.py` chunks Markdown runbooks, generates embeddings, and populates `kb_documents` and `kb_chunks` as a separate job:

    docker compose exec service-b python -m app.ingest kb

The production embedding column is `vector(1536)` with an HNSW index. PostgreSQL full-text search uses a generated `tsvector` column and GIN index. The retrieval flow is vector search + keyword search + Reciprocal Rank Fusion, returning the top five chunks.

## API

- `POST /v1/triage`
- `POST /v1/feedback`
- `GET /healthz`
- `GET /readyz`
- `GET /metrics`

## Pipeline

1. Check `request_id` idempotency.
2. Check duplicate `input_hash`.
3. Embed the ticket.
4. Run pgvector cosine search and PostgreSQL full-text search.
5. Fuse ranked results with Reciprocal Rank Fusion.
6. Classify with Claude Sonnet or the offline rule-based model.
7. Validate retrieved citations.
8. Compute confidence.
9. Persist the result in `triage_runs`.

Unsupported tickets use the refusal path with `category: other`, no suggested reply, and clarifying questions.

## Tests and evaluation

    ruff check .
    pytest -v
    EMBEDDING_PROVIDER=hashing LLM_PROVIDER=rule-based python -m app.evaluation evals/rag_eval.json

The evaluation reports Recall@5, Precision@5, MRR, NDCG@5, category/priority/refusal accuracy, citation metrics, grounded-response rate, latency, and confidence.

## Metrics

Prometheus metrics are exposed at `/metrics`, including request outcomes, latency, confidence, refusals, overloads, cache hits, retrieval size, in-flight requests, LLM tokens, and estimated cost.

## Database

PostgreSQL with pgvector stores `kb_documents`, `kb_chunks`, `triage_runs`, and `feedback`. Alembic migrations are in `alembic/`; the current vector-dimension migration is `8d1c4d2a9f31`.

## CI/CD

GitHub Actions runs lint, tests, RAG evaluation, Docker build, and AWS deployment. The deployment path pushes an immutable commit-tagged image to ECR and deploys it to ECS/Fargate using OIDC and Secrets Manager. See `deploy/ecs-task-definition.json` and `.github/workflows/ci.yml` for required configuration.
