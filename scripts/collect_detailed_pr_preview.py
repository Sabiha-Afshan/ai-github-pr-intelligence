"""Collect detailed information for five PRs."""

import sys

from src.config.settings import get_settings
from src.github.client import GitHubClient
from src.github.detail_collector import (
    DetailedPullRequestCollector,
    load_pr_numbers_from_summary,
)
from src.github.errors import GitHubClientError
from src.utils.paths import (
    CHECKPOINT_DIRECTORY,
    RAW_DATA_DIRECTORY,
)


def main() -> int:
    """Run a safe five-record detailed extraction."""

    settings = get_settings()

    summary_file = RAW_DATA_DIRECTORY / "pallets_flask_closed_pr_summaries.parquet"

    try:
        pr_numbers = load_pr_numbers_from_summary(
            summary_file=summary_file,
            maximum_records=5,
        )

        if not pr_numbers:
            print("FAIL: No PR numbers were found in the summary dataset.")
            return 1

        print(
            "Selected PR numbers:",
            pr_numbers,
        )

        with GitHubClient(settings=settings) as client:
            collector = DetailedPullRequestCollector(
                client=client,
                repository=(settings.github_repository_full_name),
                output_directory=(RAW_DATA_DIRECTORY),
                checkpoint_directory=(CHECKPOINT_DIRECTORY),
            )

            summary = collector.collect(
                pr_numbers=pr_numbers,
                resume=True,
            )

    except (
        FileNotFoundError,
        ValueError,
        GitHubClientError,
    ) as error:
        print(f"FAIL: {error}")
        return 1

    print()
    print("Detailed PR extraction completed")
    print("-" * 50)

    for key, value in summary.items():
        print(f"{key}: {value}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
