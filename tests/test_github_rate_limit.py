"""Tests for GitHub rate-limit utilities."""

from datetime import UTC, datetime

import httpx

from src.github.rate_limit import (
    calculate_rate_limit_wait_seconds,
    extract_rate_limit_status,
)


def test_extract_rate_limit_headers() -> None:
    """Confirm GitHub headers are parsed correctly."""

    response = httpx.Response(
        status_code=200,
        headers={
            "X-RateLimit-Limit": "5000",
            "X-RateLimit-Remaining": "4990",
            "X-RateLimit-Used": "10",
            "X-RateLimit-Reset": "1900000000",
            "X-RateLimit-Resource": "core",
        },
    )

    status = extract_rate_limit_status(response)

    assert status.limit == 5000
    assert status.remaining == 4990
    assert status.used == 10
    assert status.reset_timestamp == 1_900_000_000
    assert status.resource == "core"
    assert status.is_exhausted is False

    assert status.reset_at == datetime.fromtimestamp(
        1_900_000_000,
        tz=UTC,
    )


def test_wait_seconds_include_buffer() -> None:
    """Confirm reset wait includes the safety buffer."""

    result = calculate_rate_limit_wait_seconds(
        reset_timestamp=1_000,
        buffer_seconds=5,
        current_timestamp=990,
    )

    assert result == 15.0


def test_past_reset_returns_zero() -> None:
    """Confirm past reset times do not cause waiting."""

    result = calculate_rate_limit_wait_seconds(
        reset_timestamp=900,
        buffer_seconds=5,
        current_timestamp=1_000,
    )

    assert result == 0.0
