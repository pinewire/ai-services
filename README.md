# Service B — triage stub

Week 3 deliverable. Contract-shaped responses, no AI code, so Service A can
build the whole intake path before the pipeline exists.

## Run it

    docker compose up --build

App: http://localhost:8001 (see `docker-compose.yml` — 8001 avoids clashing with
other local services; container itself listens on 8000).
Interactive docs: http://localhost:8001/docs

Or without Docker:

    pip install -r requirements-dev.txt
    export DATABASE_URL=postgresql+asyncpg://b_user:b_pass@localhost:5433/service_b
    uvicorn app.main:app --reload

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

## What is fake and what is not

Fake: keyword rules instead of retrieval, no embeddings, no model call,
`token_cost_usd` always 0.

Real: the request and response schemas, the 422 on invalid input, the 503
shape, the refusal path (`category: other`, no citations, clarifying
questions returned). When the pipeline lands behind this, Service A does
not change.

## Database

`docker-compose.yml` runs Postgres (`pgvector/pgvector:pg16`) alongside the
app. On startup the app creates a `triage_log` table and persists a row for
every `/v1/triage` call (best-effort — a DB hiccup never fails the request).
`/readyz` fails with 503 if Postgres is unreachable.

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
