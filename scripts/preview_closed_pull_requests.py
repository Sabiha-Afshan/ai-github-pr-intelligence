"""Preview one page of closed pull requests."""

import sys

from src.config.settings import get_settings
from src.github.client import GitHubClient
from src.github.collector import (
    GitHubPullRequestCollector,
)
from src.github.errors import GitHubClientError


def main() -> int:
    """Collect and display one page of closed PRs."""

    settings = get_settings()

    try:
        with GitHubClient(settings=settings) as client:
            collector = GitHubPullRequestCollector(
                client=client,
                settings=settings,
            )

            pull_requests = collector.collect_closed_pull_requests(max_pages=1)

    except GitHubClientError as error:
        print(f"FAIL: {error}")
        return 1

    print(f"Collected {len(pull_requests)} closed PR summaries.")

    print("-" * 70)

    for pull_request in pull_requests[:5]:
        print(f"#{pull_request.get('number')} | {pull_request.get('title')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
