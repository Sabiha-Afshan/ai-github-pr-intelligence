"""Check GitHub authentication and repository access."""

import sys

from src.config.settings import get_settings
from src.github.client import GitHubClient
from src.github.errors import GitHubClientError
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> int:
    """Run the GitHub connectivity check."""

    settings = get_settings()

    print("GitHub connection verification")
    print("-" * 35)

    if not settings.github_token:
        print("FAIL: GITHUB_TOKEN is not configured.")
        return 1

    try:
        with GitHubClient(settings=settings) as client:
            authenticated_user = client.get_authenticated_user()

            repository = client.get_repository(
                owner=settings.github_repository_owner,
                repository=(settings.github_repository_name),
            )

            rate_limit = client.get_rate_limit()

    except GitHubClientError as error:
        logger.error(
            "GitHub verification failed: %s",
            error,
        )

        print(f"FAIL: {error}")
        return 1

    user_login = authenticated_user.get(
        "login",
        "Unavailable",
    )

    repository_name = repository.get(
        "full_name",
        "Unavailable",
    )

    core_limits = rate_limit.get("resources", {}).get("core", {})

    remaining = core_limits.get(
        "remaining",
        "Unavailable",
    )

    limit = core_limits.get(
        "limit",
        "Unavailable",
    )

    print(f"PASS: Authenticated as {user_login}")
    print(f"PASS: Repository access: {repository_name}")
    print(f"PASS: Core API requests remaining: {remaining}/{limit}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
