"""Resumable collection of normalized PR summaries."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.data.checkpoints import (
    load_checkpoint,
    save_checkpoint,
)
from src.data.storage import (
    load_existing_records,
    save_json_records,
    save_records_csv,
    save_records_parquet,
)
from src.github.client import GitHubClient
from src.github.normalization import (
    normalize_pull_request,
)
from src.github.schemas import (
    CollectionCheckpoint,
    CollectionSummary,
    PullRequestExtractionFailure,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ResumablePullRequestSummaryCollector:
    """Collect and normalize PR summaries with checkpoints."""

    def __init__(
        self,
        client: GitHubClient,
        repository: str,
        output_directory: Path,
        checkpoint_directory: Path,
    ) -> None:
        self.client = client
        self.repository = repository
        self.owner, self.repository_name = repository.split("/", maxsplit=1)

        self.output_directory = output_directory
        self.checkpoint_directory = checkpoint_directory

        safe_repository_name = repository.replace("/", "_")

        self.collection_name = f"{safe_repository_name}_closed_pr_summaries"

        self.csv_path = self.output_directory / f"{self.collection_name}.csv"

        self.parquet_path = self.output_directory / f"{self.collection_name}.parquet"

        self.failure_path = (
            self.output_directory / f"{self.collection_name}_failures.json"
        )

        self.checkpoint_path = (
            self.checkpoint_directory / f"{self.collection_name}_checkpoint.json"
        )

    def _load_existing_records(
        self,
    ) -> list[dict[str, Any]]:
        """Load previously saved records."""

        if self.parquet_path.exists():
            return load_existing_records(self.parquet_path)

        return load_existing_records(self.csv_path)

    def collect(
        self,
        maximum_pages: int = 1,
        resume: bool = True,
    ) -> CollectionSummary:
        """Collect normalized PR summary pages."""

        started_at = datetime.now(UTC)

        checkpoint = load_checkpoint(
            checkpoint_path=self.checkpoint_path,
            repository=self.repository,
            collection_name=self.collection_name,
        )

        if not resume:
            checkpoint = CollectionCheckpoint(
                repository=self.repository,
                collection_name=self.collection_name,
                updated_at=datetime.now(UTC),
            )

        existing_records = self._load_existing_records() if resume else []

        records_by_pr_number = {
            int(record["pr_number"]): record
            for record in existing_records
            if record.get("pr_number") is not None
        }

        completed_pr_numbers = set(checkpoint.completed_pr_numbers)

        failures: list[PullRequestExtractionFailure] = []

        duplicates_removed = 0

        start_page = checkpoint.last_completed_page + 1

        end_page = start_page + maximum_pages - 1

        path = f"/repos/{self.owner}/{self.repository_name}/pulls"

        for page_number in range(
            start_page,
            end_page + 1,
        ):
            logger.info(
                "Collecting page %s for %s.",
                page_number,
                self.repository,
            )

            raw_records = self.client.get_json(
                path=path,
                params={
                    "state": "closed",
                    "sort": "created",
                    "direction": "desc",
                    "per_page": (self.client.settings.github_default_page_size),
                    "page": page_number,
                },
            )

            if not isinstance(
                raw_records,
                list,
            ):
                raise TypeError("Expected GitHub to return a list of pull requests.")

            if not raw_records:
                checkpoint.status = "completed"

                save_checkpoint(
                    checkpoint,
                    self.checkpoint_path,
                )

                break

            for raw_record in raw_records:
                pr_number = raw_record.get("number")

                if not isinstance(
                    pr_number,
                    int,
                ):
                    failures.append(
                        PullRequestExtractionFailure(
                            repository=(self.repository),
                            pr_number=None,
                            stage="summary_normalization",
                            error_type=("MissingPRNumber"),
                            error_message=(
                                "GitHub record did not contain a valid PR number."
                            ),
                            occurred_at=datetime.now(UTC),
                            retryable=False,
                        )
                    )

                    continue

                if pr_number in completed_pr_numbers:
                    duplicates_removed += 1
                    continue

                try:
                    normalized_record = normalize_pull_request(
                        raw_pull_request=(raw_record),
                        repository=(self.repository),
                    )
                except Exception as error:
                    logger.exception(
                        "Failed to normalize PR #%s.",
                        pr_number,
                    )

                    failures.append(
                        PullRequestExtractionFailure(
                            repository=(self.repository),
                            pr_number=pr_number,
                            stage="summary_normalization",
                            error_type=type(error).__name__,
                            error_message=str(error),
                            occurred_at=datetime.now(UTC),
                            retryable=False,
                        )
                    )

                    continue

                records_by_pr_number[pr_number] = normalized_record.model_dump(
                    mode="json"
                )

                completed_pr_numbers.add(pr_number)

            checkpoint.last_completed_page = page_number

            checkpoint.collected_record_count = len(records_by_pr_number)

            checkpoint.failed_record_count += len(failures)

            checkpoint.completed_pr_numbers = sorted(completed_pr_numbers)

            save_checkpoint(
                checkpoint,
                self.checkpoint_path,
            )

            records = list(records_by_pr_number.values())

            save_records_csv(
                records,
                self.csv_path,
            )

            save_records_parquet(
                records,
                self.parquet_path,
            )

            save_json_records(
                [failure.model_dump(mode="json") for failure in failures],
                self.failure_path,
            )

            if len(raw_records) < self.client.settings.github_default_page_size:
                checkpoint.status = "completed"

                save_checkpoint(
                    checkpoint,
                    self.checkpoint_path,
                )

                break

        completed_at = datetime.now(UTC)

        return CollectionSummary(
            repository=self.repository,
            collection_name=self.collection_name,
            records_collected=(len(records_by_pr_number)),
            records_failed=len(failures),
            duplicates_removed=(duplicates_removed),
            output_file=str(self.parquet_path),
            checkpoint_file=str(self.checkpoint_path),
            failure_file=str(self.failure_path),
            started_at=started_at,
            completed_at=completed_at,
            status=checkpoint.status,
        )
