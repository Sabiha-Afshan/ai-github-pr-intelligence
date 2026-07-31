"""GitHub pull-request collection services."""

from collections.abc import Iterator
from typing import Any

from src.config.settings import Settings, get_settings
from src.github.client import GitHubClient
from src.utils.logging import get_logger

logger = get_logger(__name__)


class GitHubPullRequestCollector:
    """Collect pull-request data from one repository."""

    def __init__(
        self,
        client: GitHubClient,
        settings: Settings | None = None,
    ) -> None:
        self.client = client
        self.settings = settings or get_settings()

    def iterate_closed_pull_requests(
        self,
        owner: str | None = None,
        repository: str | None = None,
        max_pages: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield closed pull requests in descending order."""

        repository_owner = owner or self.settings.github_repository_owner

        repository_name = repository or self.settings.github_repository_name

        path = f"/repos/{repository_owner}/{repository_name}/pulls"

        params = {
            "state": "closed",
            "sort": "created",
            "direction": "desc",
        }

        logger.info(
            "Starting closed PR collection for %s/%s.",
            repository_owner,
            repository_name,
        )

        yield from self.client.iterate_pages(
            path=path,
            params=params,
            max_pages=max_pages,
        )

    def collect_closed_pull_requests(
        self,
        owner: str | None = None,
        repository: str | None = None,
        max_pages: int | None = None,
    ) -> list[dict[str, Any]]:
        """Collect closed pull requests into a list."""

        pull_requests = list(
            self.iterate_closed_pull_requests(
                owner=owner,
                repository=repository,
                max_pages=max_pages,
            )
        )

        logger.info(
            "Collected %s closed pull-request summaries.",
            len(pull_requests),
        )

        return pull_requests

    def get_pull_request_details(
        self,
        pr_number: int,
        owner: str | None = None,
        repository: str | None = None,
    ) -> dict[str, Any]:
        """Retrieve detailed information for one PR."""

        repository_owner = owner or self.settings.github_repository_owner

        repository_name = repository or self.settings.github_repository_name

        path = f"/repos/{repository_owner}/{repository_name}/pulls/{pr_number}"

        payload = self.client.get_json(path)

        if not isinstance(payload, dict):
            raise TypeError(
                "GitHub returned an unexpected pull-request detail response."
            )

        return payload

    def get_pull_request_files(
        self,
        pr_number: int,
        owner: str | None = None,
        repository: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve all changed files for one PR."""

        repository_owner = owner or self.settings.github_repository_owner

        repository_name = repository or self.settings.github_repository_name

        path = f"/repos/{repository_owner}/{repository_name}/pulls/{pr_number}/files"

        return self.client.get_all_pages(
            path=path,
        )
