"""API key authentication for Service B's callable endpoints.

Keys are read from SERVICE_B_API_KEYS as a comma-separated list, so rotation
is: deploy with both the old and new key, move callers across, deploy again
with only the new one. No window where a valid caller is locked out.

This fails closed. If AUTH_REQUIRED is true and no keys are configured,
every request gets a 503 rather than being waved through — a deployment
that forgot the secret should be loudly broken, not quietly open. Local
development and the test suite set AUTH_REQUIRED=false.
"""

import logging
import os
import secrets

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

log = logging.getLogger("service-b.auth")

_FALSEY = {"false", "0", "no", "off"}

API_KEY_HEADER = "X-API-Key"

# Used as the dependency below rather than a plain Header, because this is
# what puts the security scheme into the generated OpenAPI document. Service
# A's client is generated from that document, so with a bare Header the
# contract would not mention authentication at all and A would discover the
# requirement the first time it got a 401.
#
# auto_error=False so a missing header returns None instead of raising
# FastAPI's own 403 — the 401 and the WWW-Authenticate header below are the
# correct response, and the distinction matters to A's error handling.
api_key_scheme = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


def _configured_keys() -> list[str]:
    raw = os.getenv("SERVICE_B_API_KEYS", "")
    return [key.strip() for key in raw.split(",") if key.strip()]


def _auth_required() -> bool:
    return os.getenv("AUTH_REQUIRED", "true").strip().lower() not in _FALSEY


async def require_api_key(x_api_key: str | None = Security(api_key_scheme)) -> None:
    if not _auth_required():
        return

    keys = _configured_keys()
    if not keys:
        log.error("auth.not_configured AUTH_REQUIRED is true but SERVICE_B_API_KEYS is empty")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured.",
        )

    if x_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing {API_KEY_HEADER} header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Deliberately not `any(...)`: that short-circuits on the first match, so
    # response time would leak which position in the list matched. During a
    # rotation that distinguishes the old key from the new one.
    matched = False
    for key in keys:
        if secrets.compare_digest(x_api_key, key):
            matched = True

    if not matched:
        log.warning("auth.rejected reason=invalid_key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
