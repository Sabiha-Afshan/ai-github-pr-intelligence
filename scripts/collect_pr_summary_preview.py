"""Collect one resumable page of normalized PR summaries."""

import sys

from src.config.settings import get_settings
from src.github.client import GitHubClient
from src.github.errors import GitHubClientError
from src.github.summary_collector import (
    ResumablePullRequestSummaryCollector,
)
from src.utils.paths import (
    CHECKPOINT_DIRECTORY,
    RAW_DATA_DIRECTORY,
)


def main() -> int:
    """Collect one safe preview page."""

    settings = get_settings()

    try:
        with GitHubClient(settings=settings) as client:
            collector = ResumablePullRequestSummaryCollector(
                client=client,
                repository=(settings.github_repository_full_name),
                output_directory=(RAW_DATA_DIRECTORY),
                checkpoint_directory=(CHECKPOINT_DIRECTORY),
            )

            summary = collector.collect(
                maximum_pages=1,
                resume=True,
            )

    except GitHubClientError as error:
        print(f"FAIL: {error}")
        return 1

    print("PR summary collection completed")
    print("-" * 50)
    print(
        "Repository:",
        summary.repository,
    )
    print(
        "Records collected:",
        summary.records_collected,
    )
    print(
        "Records failed:",
        summary.records_failed,
    )
    print(
        "Duplicates skipped:",
        summary.duplicates_removed,
    )
    print(
        "Dataset:",
        summary.output_file,
    )
    print(
        "Checkpoint:",
        summary.checkpoint_file,
    )
    print(
        "Status:",
        summary.status,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
