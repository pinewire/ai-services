"""In-flight request limiter.

B sheds load with 503 rather than queuing internally — a fast rejection lets
A's circuit breaker and outbox absorb the backlog in order, instead of
requests piling up silently behind a full event loop.
"""

import asyncio
import os


class ConcurrencyLimiter:
    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._count = 0
        self._lock = asyncio.Lock()

    async def try_acquire(self) -> bool:
        async with self._lock:
            if self._count >= self._limit:
                return False
            self._count += 1
            return True

    async def release(self) -> None:
        async with self._lock:
            self._count -= 1


MAX_INFLIGHT_TRIAGE = int(os.getenv("MAX_INFLIGHT_TRIAGE", "8"))
limiter = ConcurrencyLimiter(MAX_INFLIGHT_TRIAGE)
