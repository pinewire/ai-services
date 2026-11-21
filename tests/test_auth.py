"""API key authentication.

`asyncio_mode = auto` in pytest.ini runs these without a marker.

These tests drive the dependency directly rather than through the app, so
they need no database and no KB. The flag and the key list are read per
request, so monkeypatching the environment is enough to switch modes.
"""

import os

import pytest
from fastapi import HTTPException

os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://b_user:b_pass@localhost:5433/service_b"
)

from app.auth import require_api_key  # noqa: E402

KEY = "k" * 43
OTHER_KEY = "j" * 43


async def test_disabled_allows_anything(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTH_REQUIRED", "false")
    monkeypatch.delenv("SERVICE_B_API_KEYS", raising=False)
    assert await require_api_key(x_api_key=None) is None


async def test_required_but_unconfigured_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A deployment that forgot the secret must break loudly, not open."""
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    monkeypatch.setenv("SERVICE_B_API_KEYS", "")
    with pytest.raises(HTTPException) as exc:
        await require_api_key(x_api_key=KEY)
    assert exc.value.status_code == 503


async def test_missing_header_is_401(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    monkeypatch.setenv("SERVICE_B_API_KEYS", KEY)
    with pytest.raises(HTTPException) as exc:
        await require_api_key(x_api_key=None)
    assert exc.value.status_code == 401


async def test_wrong_key_is_401(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    monkeypatch.setenv("SERVICE_B_API_KEYS", KEY)
    with pytest.raises(HTTPException) as exc:
        await require_api_key(x_api_key=OTHER_KEY)
    assert exc.value.status_code == 401


async def test_correct_key_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    monkeypatch.setenv("SERVICE_B_API_KEYS", KEY)
    assert await require_api_key(x_api_key=KEY) is None


async def test_either_key_accepted_during_rotation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both keys valid at once is what makes rotation a non-event."""
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    monkeypatch.setenv("SERVICE_B_API_KEYS", f"{KEY}, {OTHER_KEY}")
    assert await require_api_key(x_api_key=KEY) is None
    assert await require_api_key(x_api_key=OTHER_KEY) is None


async def test_whitespace_in_key_list_is_tolerated(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--from-env-file` and hand-edited secrets both produce stray spaces."""
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    monkeypatch.setenv("SERVICE_B_API_KEYS", f"  {KEY} ,, ")
    assert await require_api_key(x_api_key=KEY) is None
    with pytest.raises(HTTPException):
        await require_api_key(x_api_key=OTHER_KEY)
