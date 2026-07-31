"""Resumable detailed pull-request extraction."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.checkpoints import (
    load_detailed_checkpoint,
    save_detailed_checkpoint,
)
from src.data.storage import (
    load_existing_records,
    save_json_records,
    save_records_csv,
    save_records_parquet,
)
from src.github.client import GitHubClient
from src.github.detail_normalization import (
    build_detailed_pull_request_record,
)
from src.github.schemas import (
    PullRequestExtractionFailure,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DetailedPullRequestCollector:
    """Collect detailed PR and changed-file data."""

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

        safe_repository_name = repository.replace(
            "/",
            "_",
        )

        self.collection_name = f"{safe_repository_name}_detailed_pull_requests"

        self.detail_csv_path = output_directory / f"{self.collection_name}.csv"

        self.detail_parquet_path = output_directory / f"{self.collection_name}.parquet"

        self.files_csv_path = output_directory / (
            f"{safe_repository_name}_changed_files.csv"
        )

        self.files_parquet_path = output_directory / (
            f"{safe_repository_name}_changed_files.parquet"
        )

        self.failure_path = output_directory / (f"{self.collection_name}_failures.json")

        self.checkpoint_path = checkpoint_directory / (
            f"{self.collection_name}_checkpoint.json"
        )

    def _get_pull_request_details(
        self,
        pr_number: int,
    ) -> dict[str, Any]:
        """Retrieve one full PR response."""

        payload = self.client.get_json(
            path=(f"/repos/{self.owner}/{self.repository_name}/pulls/{pr_number}")
        )

        if not isinstance(payload, dict):
            raise TypeError("GitHub returned an invalid PR-detail response.")

        return payload

    def _get_pull_request_files(
        self,
        pr_number: int,
    ) -> list[dict[str, Any]]:
        """Retrieve all changed files for one PR."""

        return self.client.get_all_pages(
            path=(f"/repos/{self.owner}/{self.repository_name}/pulls/{pr_number}/files")
        )

    @staticmethod
    def _records_by_key(
        records: list[dict[str, Any]],
        key_columns: list[str],
    ) -> dict[tuple[Any, ...], dict[str, Any]]:
        """Index records using one or more key columns."""

        indexed_records = {}

        for record in records:
            key = tuple(record.get(column) for column in key_columns)

            if None not in key:
                indexed_records[key] = record

        return indexed_records

    def collect(
        self,
        pr_numbers: list[int],
        resume: bool = True,
    ) -> dict[str, Any]:
        """Collect detailed data for selected PR numbers."""

        checkpoint = load_detailed_checkpoint(
            checkpoint_path=self.checkpoint_path,
            repository=self.repository,
            collection_name=self.collection_name,
        )

        if not resume:
            checkpoint.completed_pr_numbers = []
            checkpoint.failed_pr_numbers = []
            checkpoint.detailed_record_count = 0
            checkpoint.changed_file_record_count = 0
            checkpoint.failure_count = 0
            checkpoint.status = "in_progress"

        existing_detail_records = (
            load_existing_records(self.detail_parquet_path)
            if (resume and self.detail_parquet_path.exists())
            else []
        )

        existing_file_records = (
            load_existing_records(self.files_parquet_path)
            if (resume and self.files_parquet_path.exists())
            else []
        )

        detail_records_by_key = self._records_by_key(
            existing_detail_records,
            [
                "repository",
                "pr_number",
            ],
        )

        file_records_by_key = self._records_by_key(
            existing_file_records,
            [
                "repository",
                "pr_number",
                "file_path",
            ],
        )

        completed_pr_numbers = set(checkpoint.completed_pr_numbers)

        failures: list[PullRequestExtractionFailure] = []

        for position, pr_number in enumerate(
            pr_numbers,
            start=1,
        ):
            if pr_number in completed_pr_numbers:
                logger.info(
                    "Skipping completed PR #%s.",
                    pr_number,
                )
                continue

            logger.info(
                "Extracting detailed PR #%s (%s/%s).",
                pr_number,
                position,
                len(pr_numbers),
            )

            try:
                raw_pull_request = self._get_pull_request_details(pr_number)

                raw_files = self._get_pull_request_files(pr_number)

                (
                    detailed_record,
                    normalized_files,
                ) = build_detailed_pull_request_record(
                    raw_pull_request=(raw_pull_request),
                    raw_files=raw_files,
                    repository=self.repository,
                )

            except Exception as error:
                logger.exception(
                    "Detailed extraction failed for PR #%s.",
                    pr_number,
                )

                failures.append(
                    PullRequestExtractionFailure(
                        repository=self.repository,
                        pr_number=pr_number,
                        stage="detailed_extraction",
                        error_type=type(error).__name__,
                        error_message=str(error),
                        occurred_at=datetime.now(UTC),
                        retryable=True,
                    )
                )

                checkpoint.failed_pr_numbers = sorted(
                    set(checkpoint.failed_pr_numbers) | {pr_number}
                )

                checkpoint.failure_count += 1

                save_detailed_checkpoint(
                    checkpoint,
                    self.checkpoint_path,
                )

                continue

            detail_records_by_key[
                (
                    self.repository,
                    pr_number,
                )
            ] = detailed_record.model_dump(mode="json")

            for changed_file in normalized_files:
                file_records_by_key[
                    (
                        self.repository,
                        pr_number,
                        changed_file.file_path,
                    )
                ] = changed_file.model_dump(mode="json")

            completed_pr_numbers.add(pr_number)

            checkpoint.completed_pr_numbers = sorted(completed_pr_numbers)

            checkpoint.detailed_record_count = len(detail_records_by_key)

            checkpoint.changed_file_record_count = len(file_records_by_key)

            save_detailed_checkpoint(
                checkpoint,
                self.checkpoint_path,
            )

            detail_records = list(detail_records_by_key.values())

            file_records = list(file_records_by_key.values())

            save_records_csv(
                detail_records,
                self.detail_csv_path,
            )

            save_records_parquet(
                detail_records,
                self.detail_parquet_path,
            )

            save_records_csv(
                file_records,
                self.files_csv_path,
            )

            save_records_parquet(
                file_records,
                self.files_parquet_path,
            )

            save_json_records(
                [failure.model_dump(mode="json") for failure in failures],
                self.failure_path,
            )

        checkpoint.status = (
            "completed"
            if set(pr_numbers).issubset(completed_pr_numbers)
            else "partially_completed"
        )

        save_detailed_checkpoint(
            checkpoint,
            self.checkpoint_path,
        )

        return {
            "repository": self.repository,
            "requested_pr_count": len(pr_numbers),
            "completed_pr_count": len(completed_pr_numbers),
            "failed_in_this_run": len(failures),
            "detailed_record_count": len(detail_records_by_key),
            "changed_file_record_count": len(file_records_by_key),
            "detail_dataset": str(self.detail_parquet_path),
            "changed_file_dataset": str(self.files_parquet_path),
            "checkpoint": str(self.checkpoint_path),
            "status": checkpoint.status,
        }


def load_pr_numbers_from_summary(
    summary_file: Path,
    maximum_records: int | None = None,
) -> list[int]:
    """Load PR numbers from the summary dataset."""

    if not summary_file.exists():
        raise FileNotFoundError(f"Summary dataset not found: {summary_file}")

    if summary_file.suffix.lower() == ".csv":
        dataframe = pd.read_csv(summary_file)
    else:
        dataframe = pd.read_parquet(summary_file)

    if "pr_number" not in dataframe.columns:
        raise ValueError("Summary dataset does not contain the pr_number column.")

    pr_numbers = dataframe["pr_number"].dropna().astype(int).drop_duplicates().tolist()

    if maximum_records is not None:
        return pr_numbers[:maximum_records]

    return pr_numbers
