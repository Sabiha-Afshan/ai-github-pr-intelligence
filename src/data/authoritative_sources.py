"""Authoritative dataset definitions and lineage."""

from dataclasses import dataclass
from pathlib import Path

from src.utils.paths import (
    DATA_EVALUATION_DIRECTORY,
    PROCESSED_DATA_DIRECTORY,
    RAW_DATA_DIRECTORY,
)


@dataclass(frozen=True)
class AuthoritativeDataset:
    """Definition of one authoritative project dataset."""

    dataset_name: str
    stage: str
    path: Path
    expected_rows: int | None
    expected_pr_coverage: int | None
    description: str
    authoritative_for: str


def get_authoritative_datasets() -> list[AuthoritativeDataset]:
    """Return the expected project data lineage."""

    return [
        AuthoritativeDataset(
            dataset_name=("Time-matched PR population"),
            stage="Population selection",
            path=(
                RAW_DATA_DIRECTORY
                / ("pallets_flask_time_matched_600_pr_population.csv")
            ),
            expected_rows=600,
            expected_pr_coverage=600,
            description=(
                "Selected PR identities and quarterly time-matched sampling metadata."
            ),
            authoritative_for=("Population membership and sampling design"),
        ),
        AuthoritativeDataset(
            dataset_name=("Detailed 600-PR extraction"),
            stage="Raw detailed extraction",
            path=(
                PROCESSED_DATA_DIRECTORY
                / ("pallets_flask_time_matched_600_detailed.csv")
            ),
            expected_rows=600,
            expected_pr_coverage=600,
            description=(
                "Complete PR metadata, outcomes, lifecycle "
                "information and changed-file indicators."
            ),
            authoritative_for=("Raw detailed PR attributes and target outcome"),
        ),
        AuthoritativeDataset(
            dataset_name=("Validated 600-PR dataset"),
            stage="Data validation",
            path=(
                PROCESSED_DATA_DIRECTORY / ("pallets_flask_corrected_600_validated.csv")
            ),
            expected_rows=600,
            expected_pr_coverage=600,
            description=("Quality-checked and standardized analytical dataset."),
            authoritative_for=("Data-quality analysis and validated values"),
        ),
        AuthoritativeDataset(
            dataset_name=("Outlier-flagged 600-PR dataset"),
            stage="Outlier analysis",
            path=(
                PROCESSED_DATA_DIRECTORY
                / ("pallets_flask_corrected_600_outlier_flagged.csv")
            ),
            expected_rows=600,
            expected_pr_coverage=600,
            description=(
                "Validated dataset with outlier indicators and treatment metadata."
            ),
            authoritative_for=("Outlier analysis and robust transformations"),
        ),
        AuthoritativeDataset(
            dataset_name=("Feature-engineered 600-PR dataset"),
            stage="Feature engineering",
            path=(
                PROCESSED_DATA_DIRECTORY
                / ("pallets_flask_corrected_600_feature_engineered.csv")
            ),
            expected_rows=600,
            expected_pr_coverage=600,
            description=("Modelling-ready features with timing and leakage controls."),
            authoritative_for=("Model training and inference feature schema"),
        ),
        AuthoritativeDataset(
            dataset_name=("Model 1 time-based split assignments"),
            stage="Chronological splitting",
            path=(
                DATA_EVALUATION_DIRECTORY
                / "corrected_model1_time_based_split_assignments.csv"
            ),
            expected_rows=600,
            expected_pr_coverage=600,
            description=(
                "Chronological train, validation and test "
                "assignment for every selected PR."
            ),
            authoritative_for=("Model 1 train-validation-test membership"),
        ),
    ]
