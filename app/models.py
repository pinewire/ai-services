"""Persisted record of each triage call — audit trail and future training data."""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class TriageLog(Base):
    __tablename__ = "triage_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), index=True)
    ticket_ref: Mapped[str] = mapped_column(String)
    category: Mapped[str] = mapped_column(String)
    priority: Mapped[str] = mapped_column(String)
    confidence: Mapped[float] = mapped_column(Float)
    latency_ms: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
