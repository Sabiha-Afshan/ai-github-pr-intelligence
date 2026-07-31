"""GitHub API rate-limit utilities."""

import time
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RateLimitStatus:
    """Rate-limit values extracted from GitHub headers."""

    limit: int | None
    remaining: int | None
    used: int | None
    reset_timestamp: int | None
    resource: str | None

    @property
    def reset_at(self) -> datetime | None:
        """Return the reset time as a UTC datetime."""

        if self.reset_timestamp is None:
            return None

        return datetime.fromtimestamp(
            self.reset_timestamp,
            tz=UTC,
        )

    @property
    def is_exhausted(self) -> bool:
        """Return whether the current limit is exhausted."""

        return self.remaining == 0


def parse_integer_header(
    response: httpx.Response,
    header_name: str,
) -> int | None:
    """Safely parse an integer response header."""

    raw_value = response.headers.get(header_name)

    if raw_value is None:
        return None

    try:
        return int(raw_value)
    except ValueError:
        return None


def extract_rate_limit_status(
    response: httpx.Response,
) -> RateLimitStatus:
    """Extract rate-limit information from a response."""

    return RateLimitStatus(
        limit=parse_integer_header(
            response,
            "X-RateLimit-Limit",
        ),
        remaining=parse_integer_header(
            response,
            "X-RateLimit-Remaining",
        ),
        used=parse_integer_header(
            response,
            "X-RateLimit-Used",
        ),
        reset_timestamp=parse_integer_header(
            response,
            "X-RateLimit-Reset",
        ),
        resource=response.headers.get("X-RateLimit-Resource"),
    )


def calculate_rate_limit_wait_seconds(
    reset_timestamp: int | None,
    buffer_seconds: int = 5,
    current_timestamp: float | None = None,
) -> float:
    """Calculate how long the client should wait."""

    if reset_timestamp is None:
        return 0.0

    now = current_timestamp if current_timestamp is not None else time.time()

    return max(
        0.0,
        float(reset_timestamp) - now + buffer_seconds,
    )
