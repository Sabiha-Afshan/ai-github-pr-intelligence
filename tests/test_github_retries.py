"""Tests for GitHub retry behaviour."""

from unittest.mock import patch

import httpx

from src.config.settings import Settings
from src.github.client import GitHubClient


def create_settings() -> Settings:
    """Create retry-test settings."""

    return Settings(
        github_token="test-token",
        github_api_base_url="https://api.github.test",
        github_max_retries=2,
        github_retry_backoff_seconds=0.01,
        github_rate_limit_wait_enabled=False,
    )


def test_server_error_is_retried() -> None:
    """Confirm temporary server errors are retried."""

    request_count = 0

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        nonlocal request_count

        request_count += 1

        if request_count < 3:
            return httpx.Response(
                status_code=503,
                json={"message": "Temporarily unavailable"},
            )

        return httpx.Response(
            status_code=200,
            json={
                "full_name": "pallets/flask",
            },
        )

    transport = httpx.MockTransport(handler)

    with patch("src.github.client.time.sleep") as sleep_mock:
        with GitHubClient(
            settings=create_settings(),
            transport=transport,
        ) as client:
            result = client.get_repository(
                owner="pallets",
                repository="flask",
            )

    assert request_count == 3
    assert result["full_name"] == "pallets/flask"
    assert sleep_mock.call_count == 2
