"""Session-scoped fixture that seeds the KB from `kb/` before tests run, so
retrieval has something real to find. Runs in its own event loop and
disposes the engine afterwards — the app disposes it again on lifespan
shutdown, so nothing is left bound to a closed loop (see app.main.lifespan).
"""

import asyncio
import os
from pathlib import Path

os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://b_user:b_pass@localhost:5433/service_b"
)

import pytest  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.db import engine  # noqa: E402
from app.ingest import ingest_path  # noqa: E402

KB_DIR = Path(__file__).resolve().parent.parent / "kb"


@pytest.fixture(scope="session", autouse=True)
def seed_kb() -> None:
    async def _reset_and_seed() -> None:
        async with engine.begin() as conn:
            for table in ("feedback", "triage_runs", "kb_chunks", "kb_documents"):
                await conn.execute(text(f"drop table if exists {table} cascade"))
        await ingest_path(KB_DIR)

    asyncio.run(_reset_and_seed())
