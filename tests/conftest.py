"""Session-scoped fixture that creates the schema and seeds the KB from `kb/`
before tests run, so retrieval has something real to find. Runs in its own
event loop and disposes the engine afterwards — otherwise the pool holds
asyncpg connections bound to a loop that is already closed.

The app no longer creates tables at startup (Alembic owns the schema), so the
test session does it here with create_all: faster than running the migration
chain, and the from-scratch migration path is verified separately.
"""

import asyncio
import os
from pathlib import Path

os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://b_user:b_pass@localhost:5433/service_b"
)

import pytest  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.db import Base, engine, ensure_extensions  # noqa: E402
from app.ingest import ingest_path  # noqa: E402

KB_DIR = Path(__file__).resolve().parent.parent / "kb"


@pytest.fixture(scope="session", autouse=True)
def seed_kb() -> None:
    async def _reset_and_seed() -> None:
        # Drop, recreate and seed inside one connection on one loop. Splitting
        # this across two fixtures gave each its own event loop, and asyncpg
        # refuses a connection checked out on a different loop than it was
        # created on.
        async with engine.begin() as conn:
            for table in ("feedback", "triage_runs", "kb_chunks", "kb_documents"):
                await conn.execute(text(f"drop table if exists {table} cascade"))
            await ensure_extensions(conn)
            await conn.run_sync(Base.metadata.create_all)
        await ingest_path(KB_DIR)
        await engine.dispose()

    asyncio.run(_reset_and_seed())
