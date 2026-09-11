# Service B — triage engine

Classifies helpdesk tickets and drafts grounded replies via hybrid retrieval
(pgvector + Postgres full-text search) over ingested runbooks. Stateless with
respect to tickets: it answers "what do you make of this?" and never stores,
mutates, or replies on behalf of anyone.

## Run it

    docker compose up --build

App: http://localhost:8001 (see `docker-compose.yml` — 8001 avoids clashing with
other local services; container itself listens on 8000).
Interactive docs: http://localhost:8001/docs

Or without Docker:

    pip install -r requirements-dev.txt
    export DATABASE_URL=postgresql+asyncpg://b_user:b_pass@localhost:5433/service_b
    uvicorn app.main:app --reload

## Ingest the knowledge base

The triage pipeline can't ground a reply in anything until the KB has
content. `kb/*.md` are sample runbooks; ingestion chunks them by heading,
embeds each chunk, and populates `kb_documents` / `kb_chunks`:

    docker exec <service-b container> python -m app.ingest kb
    # or, without Docker:
    python -m app.ingest kb

This runs as a one-off job, not inside the API process — see "Hosting"
below. `tests/conftest.py` re-runs it automatically before the test suite.

## Tests

    pip install -r requirements-dev.txt
    docker compose up -d db
    DATABASE_URL=postgresql+asyncpg://b_user:b_pass@localhost:5433/service_b pytest -v
    ruff check .

## The endpoint

    POST /v1/triage

    curl -X POST localhost:8000/v1/triage \
      -H 'Content-Type: application/json' \
      -d '{
        "request_id": "0f2c9a1e-4b77-4c31-9f4a-1d8e5b2a7c60",
        "ticket_ref": "TKT-8F3A21",
        "subject": "VPN drops after about 30 seconds",
        "messages": [{
          "author_type": "customer",
          "body": "VPN connects then disconnects. Error VPN-4021.",
          "created_at": "2026-09-04T14:22:03Z"
        }],
        "requester": {
          "department": "Finance", "region": "us-east",
          "tenure_days": 412, "prior_ticket_count": 3
        }
      }'

## Testing Service A's resilience path

Two query parameters exist so A can exercise its timeout, circuit breaker and
outbox worker without anyone killing a container by hand.

| Parameter  | Effect                                      |
|------------|---------------------------------------------|
| `delay_ms` | Sleeps before responding. Trip A's timeout.  |
| `fail`     | Returns 503 with `Retry-After: 30`.          |

    # exceed A's 2500ms budget
    curl -X POST 'localhost:8000/v1/triage?delay_ms=3000' ...

    # five of these in a row should open A's circuit breaker
    curl -X POST 'localhost:8000/v1/triage?fail=true' ...

Both belong in A's integration tests. Remove them before this is ever
reachable from anything but a dev environment.

A third mechanism sheds load under real concurrency pressure: once
`MAX_INFLIGHT_TRIAGE` (default 8) requests are in flight, new ones get the
same 503 + `Retry-After: 30` without touching the pipeline.

## The pipeline

1. **Idempotency check** — `request_id` is unique in `triage_runs`; a repeat
   call (A's sync attempt and its outbox worker can both fire) returns the
   cached run instead of re-running inference.
2. **Embed** the ticket subject + message bodies (`app/embeddings.py`).
3. **Vector search** `kb_chunks` via pgvector cosine similarity.
4. **Keyword search** the same chunks via Postgres full-text search — this
   is what catches exact strings like `VPN-4021` that embeddings alone treat
   as near-noise.
5. **Fuse** both result sets with reciprocal rank fusion, take the top 5
   (`app/retrieval.py`).
6. **Classify** against a forced JSON schema (`app/llm.py`); a Pydantic
   validation failure triggers one reject-and-retry.
7. **Resolve citations** against the retrieved set only — a chunk ID the
   model never saw is confabulation even if it exists in the KB.
8. **Compose confidence** from measurable signals (top similarity, margin,
   citation survival, vector/keyword agreement), never self-reported
   (`app/confidence.py`). Capped hard at 0.3 if no citation survives.
9. **Persist and respond** (`app/pipeline.py`).

**Refusal is a first-class outcome.** When retrieval finds nothing above the
relevance threshold, the response is `category: "other"`, `confidence: 0.3`,
`suggested_reply: null`, and clarifying questions — never an invented fix.

The default embedding/model implementations (`HashingEmbeddingClient`,
`RuleBasedTriageModel`) are zero-dependency stand-ins: deterministic feature
hashing instead of a real embedding model, keyword rules instead of an LLM
call. They keep local dev and CI free of API keys and model downloads while
keeping the pipeline's shape (retrieval → forced JSON → citation resolution →
composed confidence) identical to what a real model swap-in would use. Swap
them via `get_embedding_client()` / `get_triage_model()`.

## Feedback

    POST /v1/feedback
    { "request_id": "...", "verdict": "incorrect", "corrected_category": "network.vpn" }

Links an agent correction to the `triage_runs` row for that `request_id`
(404 if no run exists yet).

## Database

`docker-compose.yml` runs Postgres (`pgvector/pgvector:pg16`) alongside the
app. On startup the app creates the `vector` extension and four tables:
`kb_documents`, `kb_chunks` (embedding + full-text index), `triage_runs`
(the idempotency log), and `feedback`. `/readyz` fails with 503 if Postgres
is unreachable.

This is a separate Postgres instance from Service A's, with separate
credentials — same engine, different database, no cross-service joins.

## Hosting notes

B's hosting requirements are the inverse of A's: it's permitted to be down,
so deploy it separately from A with its own scaling policy and deploy
cadence — shipping a prompt change shouldn't redeploy the ticket API. No GPU
needed (embeddings are CPU-sized or hosted-API; the expensive call is the
LLM, which is someone else's infrastructure). KB ingestion (`app/ingest.py`)
runs as a scheduled job, never inside the API process.

## CI/CD

`.github/workflows/ci.yml` runs on every push/PR to `main`:

1. **test** — spins up a Postgres service container, installs
   `requirements-dev.txt`, runs `ruff check` and `pytest`.
2. **build** — builds the Docker image.
3. **deploy** (main branch only) — assumes an AWS IAM role via OIDC (no
   long-lived AWS secrets in the repo), pushes the image to ECR, and forces
   a new ECS deployment.

To enable deploys, add a repository secret `AWS_DEPLOY_ROLE_ARN` pointing at
an IAM role trusted for GitHub's OIDC provider, and create the ECR repo,
ECS cluster/service ahead of time (see "Cloud hosting plan" below).

## Cloud hosting plan (AWS)

Minimal path once local + CI are solid:

1. **ECR** — repository for the `service-b` image (CI pushes here).
2. **RDS Postgres** (with `pgvector` extension) replacing the local `db`
   container — set `DATABASE_URL` via ECS task secrets/SSM Parameter Store,
   never plain env vars for credentials.
3. **ECS Fargate** — service running the container, `/healthz` as the ALB
   target group health check, `/readyz` for deployment gating.
4. **Application Load Balancer** in front of the service, HTTPS via ACM.
5. **Secrets** — `AWS_DEPLOY_ROLE_ARN` (GitHub OIDC) instead of static AWS
   keys; DB credentials in Secrets Manager, injected into the task
   definition.

This repo currently ships the app + CI scaffolding; the ECR repo, RDS
instance, ECS cluster/service, and IAM OIDC role still need to be
provisioned (e.g. via Terraform/CloudFormation) before `deploy` can run.
