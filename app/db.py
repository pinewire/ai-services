"""Async SQLAlchemy engine/session wiring for Service B's Postgres database."""

import os
from collections.abc import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql+asyncpg://b_user:b_pass@localhost:5433/service_b"
)

engine: AsyncEngine = create_async_engine(DATABASE_URL, pool_pre_ping=True)
async_session = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def ensure_extensions(conn: AsyncConnection) -> None:
    """pgvector must exist before any table declares a vector column."""
    await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        yield session
