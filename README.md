# Service B — triage engine

Async FastAPI service that classifies helpdesk tickets and drafts grounded replies using hybrid retrieval over company runbooks.

## Run locally

    docker compose up --build

App: http://localhost:8001
Interactive docs: http://localhost:8001/docs

For offline development:

    LLM_PROVIDER=rule-based EMBEDDING_PROVIDER=hashing docker compose up -d --build

For the real provider path, configure `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` in the ignored `.env` file, then recreate the service.

## Architecture

The pipeline embeds the ticket, runs pgvector cosine search and PostgreSQL full-text search, merges results with Reciprocal Rank Fusion, calls Claude Sonnet, validates citations, computes confidence, and persists the run.

PostgreSQL + pgvector stores `kb_documents`, `kb_chunks`, `triage_runs`, and `feedback`. The vector column is `vector(1536)` and uses an HNSW index; lexical search uses `tsvector` with a GIN index.

Ingest the knowledge base as a separate job:

    docker compose exec service-b python -m app.ingest kb

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

GitHub Actions runs lint, tests, RAG evaluation, Docker build, and the AWS deployment workflow. The deployment path pushes an immutable commit-tagged image to ECR and deploys it to ECS/Fargate using OIDC and Secrets Manager. See `deploy/ecs-task-definition.json` and `.github/workflows/ci.yml` for required AWS/GitHub configuration.
