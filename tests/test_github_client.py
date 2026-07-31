"""Tests for the GitHub REST API client."""

import httpx
import pytest

from src.config.settings import Settings
from src.github.client import GitHubClient
from src.github.errors import (
    GitHubAuthenticationError,
    GitHubNotFoundError,
    GitHubRateLimitError,
)


def create_test_settings() -> Settings:
    """Return isolated settings for GitHub tests."""

    return Settings(
        github_token="test-token",
        github_api_base_url="https://api.github.test",
        github_repository_owner="pallets",
        github_repository_name="flask",
        github_rate_limit_wait_enabled=False,
        github_max_retries=0,
    )


def test_repository_request_returns_json() -> None:
    """Confirm successful repository retrieval."""

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        assert request.url.path == "/repos/pallets/flask"

        assert request.headers["Authorization"] == "Bearer test-token"

        return httpx.Response(
            status_code=200,
            json={
                "id": 123,
                "full_name": "pallets/flask",
            },
        )

    transport = httpx.MockTransport(handler)

    with GitHubClient(
        settings=create_test_settings(),
        transport=transport,
    ) as client:
        repository = client.get_repository(
            owner="pallets",
            repository="flask",
        )

    assert repository["id"] == 123
    assert repository["full_name"] == "pallets/flask"


def test_invalid_token_raises_authentication_error() -> None:
    """Confirm that HTTP 401 becomes a clear error."""

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=401,
            json={
                "message": "Bad credentials",
            },
        )

    transport = httpx.MockTransport(handler)

    with GitHubClient(
        settings=create_test_settings(),
        transport=transport,
    ) as client:
        with pytest.raises(GitHubAuthenticationError):
            client.get_authenticated_user()


def test_not_found_raises_clear_error() -> None:
    """Confirm that HTTP 404 is handled."""

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=404,
            json={
                "message": "Not Found",
            },
        )

    transport = httpx.MockTransport(handler)

    with GitHubClient(
        settings=create_test_settings(),
        transport=transport,
    ) as client:
        with pytest.raises(GitHubNotFoundError):
            client.get_repository(
                owner="missing",
                repository="repository",
            )


def test_rate_limit_error_contains_reset_time() -> None:
    """Confirm exhausted rate limits are detected."""

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=403,
            headers={
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": "1900000000",
            },
            json={
                "message": "API rate limit exceeded",
            },
        )

    transport = httpx.MockTransport(handler)

    with GitHubClient(
        settings=create_test_settings(),
        transport=transport,
    ) as client:
        with pytest.raises(GitHubRateLimitError) as captured_error:
            client.get_rate_limit()

    assert captured_error.value.reset_timestamp == 1_900_000_000
