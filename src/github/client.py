"""Reusable GitHub REST API client."""

import time
from collections.abc import Iterator, Mapping
from typing import Any

import httpx

from src.config.settings import Settings, get_settings
from src.github.errors import (
    GitHubAuthenticationError,
    GitHubClientError,
    GitHubNotFoundError,
    GitHubRateLimitError,
    GitHubResponseError,
)
from src.github.rate_limit import (
    calculate_rate_limit_wait_seconds,
    extract_rate_limit_status,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


RETRYABLE_STATUS_CODES = {
    429,
    500,
    502,
    503,
    504,
}


class GitHubClient:
    """Authenticated client for the GitHub REST API."""

    def __init__(
        self,
        settings: Settings | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.settings = settings or get_settings()

        self._client = httpx.Client(
            base_url=self.settings.github_api_base_url,
            headers=self._build_headers(),
            timeout=httpx.Timeout(self.settings.github_request_timeout_seconds),
            transport=transport,
            follow_redirects=True,
        )

    def _build_headers(self) -> dict[str, str]:
        """Build default GitHub request headers."""

        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": (self.settings.github_api_version),
            "User-Agent": "ai-github-pr-intelligence",
        }

        if self.settings.github_token:
            headers["Authorization"] = f"Bearer {self.settings.github_token}"

        return headers

    def close(self) -> None:
        """Close the underlying HTTP client."""

        self._client.close()

    def __enter__(self) -> "GitHubClient":
        """Enter the client context."""

        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception_value: BaseException | None,
        traceback: Any,
    ) -> None:
        """Exit the client context."""

        self.close()

    @staticmethod
    def _extract_error_message(
        response: httpx.Response,
    ) -> str:
        """Extract a readable GitHub error message."""

        try:
            payload = response.json()
        except ValueError:
            payload = {}

        if isinstance(payload, dict):
            message = payload.get("message")

            if message:
                return str(message)

        return response.text.strip() or "GitHub request failed."

    def _raise_for_non_retryable_error(
        self,
        response: httpx.Response,
    ) -> None:
        """Convert final HTTP failures into clear errors."""

        if response.is_success:
            return

        message = self._extract_error_message(response)

        rate_limit = extract_rate_limit_status(response)

        if response.status_code == 401:
            raise GitHubAuthenticationError(
                "GitHub authentication failed. Check the configured token."
            )

        if response.status_code == 403 and rate_limit.is_exhausted:
            raise GitHubRateLimitError(
                message=("GitHub API rate limit was exhausted."),
                reset_timestamp=(rate_limit.reset_timestamp),
            )

        if response.status_code == 403:
            raise GitHubAuthenticationError(f"GitHub rejected the request: {message}")

        if response.status_code == 404:
            raise GitHubNotFoundError(message)

        if response.status_code >= 500:
            raise GitHubResponseError(
                f"GitHub remained unavailable after retries: {message}"
            )

        raise GitHubClientError(
            f"GitHub request failed with status {response.status_code}: {message}"
        )

    def _calculate_retry_delay(
        self,
        attempt_number: int,
    ) -> float:
        """Calculate exponential retry delay."""

        return self.settings.github_retry_backoff_seconds * (2 ** (attempt_number - 1))

    def _wait_for_rate_limit(
        self,
        response: httpx.Response,
    ) -> bool:
        """
        Wait for GitHub's rate limit when configured.

        Return True when a wait occurred.
        """

        rate_limit = extract_rate_limit_status(response)

        if not rate_limit.is_exhausted:
            return False

        if not self.settings.github_rate_limit_wait_enabled:
            return False

        wait_seconds = calculate_rate_limit_wait_seconds(
            reset_timestamp=(rate_limit.reset_timestamp),
            buffer_seconds=(self.settings.github_rate_limit_buffer_seconds),
        )

        if wait_seconds <= 0:
            return False

        logger.warning(
            "GitHub rate limit exhausted. Waiting %.2f seconds until %s.",
            wait_seconds,
            rate_limit.reset_at,
        )

        time.sleep(wait_seconds)

        return True

    def get(
        self,
        path: str,
        params: Mapping[str, Any] | None = None,
    ) -> httpx.Response:
        """Send a GET request with retries."""

        last_response: httpx.Response | None = None
        last_error: Exception | None = None

        total_attempts = self.settings.github_max_retries + 1

        for attempt_number in range(
            1,
            total_attempts + 1,
        ):
            try:
                response = self._client.get(
                    path,
                    params=params,
                )

                last_response = response

            except (
                httpx.TimeoutException,
                httpx.RequestError,
            ) as error:
                last_error = error

                if attempt_number >= total_attempts:
                    break

                delay = self._calculate_retry_delay(attempt_number)

                logger.warning(
                    "GitHub connection error for %s. "
                    "Retrying in %.2f seconds. "
                    "Attempt %s/%s.",
                    path,
                    delay,
                    attempt_number,
                    total_attempts,
                )

                time.sleep(delay)
                continue

            if response.is_success:
                rate_limit = extract_rate_limit_status(response)

                logger.debug(
                    "GitHub request completed: path=%s status=%s remaining=%s",
                    path,
                    response.status_code,
                    rate_limit.remaining,
                )

                return response

            if response.status_code == 403:
                if self._wait_for_rate_limit(response):
                    continue

            if response.status_code in RETRYABLE_STATUS_CODES:
                if attempt_number >= total_attempts:
                    break

                delay = self._calculate_retry_delay(attempt_number)

                logger.warning(
                    "Retryable GitHub response for %s: "
                    "status=%s. Retrying in %.2f seconds. "
                    "Attempt %s/%s.",
                    path,
                    response.status_code,
                    delay,
                    attempt_number,
                    total_attempts,
                )

                time.sleep(delay)
                continue

            self._raise_for_non_retryable_error(response)

        if last_response is not None:
            self._raise_for_non_retryable_error(last_response)

        raise GitHubResponseError(
            "Unable to connect to GitHub after retries."
        ) from last_error

    def get_json(
        self,
        path: str,
        params: Mapping[str, Any] | None = None,
    ) -> Any:
        """Send a GET request and decode JSON."""

        response = self.get(
            path=path,
            params=params,
        )

        try:
            return response.json()
        except ValueError as error:
            raise GitHubResponseError("GitHub returned invalid JSON.") from error

    def iterate_pages(
        self,
        path: str,
        params: Mapping[str, Any] | None = None,
        item_key: str | None = None,
        max_pages: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """
        Yield records from a paginated GitHub endpoint.

        Use item_key for endpoints such as GitHub Search,
        where records are contained in an 'items' field.
        """

        request_params = dict(params or {})

        request_params.setdefault(
            "per_page",
            self.settings.github_default_page_size,
        )

        maximum_pages = (
            max_pages if max_pages is not None else self.settings.github_max_pages
        )

        page_number = 1

        while page_number <= maximum_pages:
            request_params["page"] = page_number

            payload = self.get_json(
                path=path,
                params=request_params,
            )

            if item_key is not None:
                if not isinstance(payload, dict):
                    raise GitHubResponseError(
                        f"Expected a dictionary response for paginated endpoint {path}."
                    )

                records = payload.get(
                    item_key,
                    [],
                )
            else:
                records = payload

            if not isinstance(records, list):
                raise GitHubResponseError(
                    f"Expected a list of paginated records from {path}."
                )

            if not records:
                break

            for record in records:
                if not isinstance(record, dict):
                    raise GitHubResponseError(
                        "GitHub pagination returned a non-dictionary record."
                    )

                yield record

            if len(records) < request_params["per_page"]:
                break

            page_number += 1

        if page_number > maximum_pages:
            logger.warning(
                "Pagination stopped after reaching "
                "the configured maximum of %s pages "
                "for %s.",
                maximum_pages,
                path,
            )

    def get_all_pages(
        self,
        path: str,
        params: Mapping[str, Any] | None = None,
        item_key: str | None = None,
        max_pages: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return all records from a paginated endpoint."""

        return list(
            self.iterate_pages(
                path=path,
                params=params,
                item_key=item_key,
                max_pages=max_pages,
            )
        )

    def get_authenticated_user(
        self,
    ) -> dict[str, Any]:
        """Return the authenticated GitHub user."""

        data = self.get_json("/user")

        if not isinstance(data, dict):
            raise GitHubResponseError("Unexpected authenticated-user response.")

        return data

    def get_repository(
        self,
        owner: str,
        repository: str,
    ) -> dict[str, Any]:
        """Return repository metadata."""

        data = self.get_json(f"/repos/{owner}/{repository}")

        if not isinstance(data, dict):
            raise GitHubResponseError("Unexpected repository response.")

        return data

    def get_rate_limit(
        self,
    ) -> dict[str, Any]:
        """Return GitHub rate-limit information."""

        data = self.get_json("/rate_limit")

        if not isinstance(data, dict):
            raise GitHubResponseError("Unexpected rate-limit response.")

        return data
