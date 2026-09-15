# Service B — triage engine

Async FastAPI service for grounded helpdesk triage over company runbooks.

## Run locally

    docker compose up --build

App: http://localhost:8001
Docs: http://localhost:8001/docs

Offline mode:

    LLM_PROVIDER=rule-based EMBEDDING_PROVIDER=hashing docker compose up -d --build

Real mode uses Claude Sonnet and OpenAI embeddings configured through the ignored `.env` file.

## Architecture

The service ingests Markdown runbooks, chunks them, embeds the chunks, and stores them in PostgreSQL with pgvector. Each query uses pgvector cosine search plus PostgreSQL full-text search, merges ranked results with Reciprocal Rank Fusion, sends the top five chunks to Claude Sonnet, validates citations, computes confidence, and persists the run.

The database contains `kb_documents`, `kb_chunks`, `triage_runs`, and `feedback`. The embedding column is `vector(1536)` with an HNSW index; lexical search uses a generated `tsvector` column and GIN index.

## Ingestion

    docker compose exec service-b python -m app.ingest kb

Ingestion is idempotent by source filename. Re-run it after changing embedding providers or KB documents.

## API

- `POST /v1/triage`
- `POST /v1/feedback`
- `GET /healthz`
- `GET /readyz`
- `GET /metrics`

## Tests and evaluation

    ruff check .
    pytest -v
    EMBEDDING_PROVIDER=hashing LLM_PROVIDER=rule-based python -m app.evaluation evals/rag_eval.json

The evaluation reports Recall@5, Precision@5, MRR, NDCG@5, category/priority/refusal accuracy, citation metrics, grounded-response rate, latency, and confidence.

## CI/CD

GitHub Actions runs lint, tests, RAG evaluation, and Docker builds. The deployment workflow pushes an immutable commit-tagged image to ECR and deploys it to ECS/Fargate using OIDC and Secrets Manager. Required configuration is in `.github/workflows/ci.yml` and `deploy/ecs-task-definition.json`.

## Migrations

    alembic upgrade head

The current vector-dimension migration is `8d1c4d2a9f31`. Existing vectors must be re-ingested after applying it.
