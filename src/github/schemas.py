"""Pydantic schemas for normalized GitHub data."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class GitHubRateLimit(BaseModel):
    """GitHub API rate-limit information."""

    limit: int
    remaining: int
    reset_at: datetime
    used: int | None = None
    resource: str = "core"


class GitHubRepository(BaseModel):
    """Basic GitHub repository information."""

    repository_id: int
    owner: str
    name: str
    full_name: str
    html_url: str
    default_branch: str | None = None
    is_private: bool = False
    archived: bool = False


class NormalizedPullRequest(BaseModel):
    """Clean pull-request record used by the project."""

    model_config = ConfigDict(
        extra="ignore",
    )

    repository: str
    pr_number: int

    title: str
    body: str | None = None

    state: str
    draft: bool = False

    author_login: str | None = None
    author_association: str | None = None

    created_at: datetime
    updated_at: datetime | None = None
    closed_at: datetime | None = None
    merged_at: datetime | None = None

    target_branch: str | None = None
    source_branch: str | None = None

    html_url: str

    labels: list[str] = Field(
        default_factory=list,
    )

    was_merged: bool
    merge_target: int
    outcome_label: str

    additions: int | None = None
    deletions: int | None = None
    changed_files: int | None = None
    commit_count: int | None = None

    raw_data_available: bool = True


class NormalizedChangedFile(BaseModel):
    """Normalized changed-file record for one pull request."""

    repository: str
    pr_number: int
    file_path: str
    file_status: str | None = None

    additions: int | None = None
    deletions: int | None = None
    total_changes: int | None = None

    previous_file_path: str | None = None
    raw_url: str | None = None
    blob_url: str | None = None
    patch_available: bool = False

    file_extension: str | None = None
    file_category: str

    is_test_file: bool = False
    is_documentation_file: bool = False
    is_configuration_file: bool = False
    is_security_sensitive_file: bool = False
    is_generated_file: bool = False


class DetailedPullRequestRecord(NormalizedPullRequest):
    """Pull request enriched with detailed GitHub fields."""

    total_changes: int | None = None

    changed_file_paths: list[str] = Field(
        default_factory=list,
    )

    test_file_count: int = 0
    documentation_file_count: int = 0
    configuration_file_count: int = 0
    security_sensitive_file_count: int = 0
    generated_file_count: int = 0

    has_test_changes: bool = False
    has_documentation_changes: bool = False
    has_configuration_changes: bool = False
    has_security_sensitive_changes: bool = False

    file_category_counts: dict[str, int] = Field(
        default_factory=dict,
    )


class DetailedExtractionCheckpoint(BaseModel):
    """Progress checkpoint for detailed PR extraction."""

    repository: str
    collection_name: str

    completed_pr_numbers: list[int] = Field(
        default_factory=list,
    )

    failed_pr_numbers: list[int] = Field(
        default_factory=list,
    )

    detailed_record_count: int = 0
    changed_file_record_count: int = 0
    failure_count: int = 0

    updated_at: datetime
    status: str = "in_progress"


class PullRequestExtractionFailure(BaseModel):
    """One failed PR extraction record."""

    repository: str
    pr_number: int | None = None
    stage: str
    error_type: str
    error_message: str
    occurred_at: datetime
    retryable: bool = False


class CollectionCheckpoint(BaseModel):
    """Checkpoint describing collection progress."""

    repository: str
    collection_name: str
    last_completed_page: int = 0
    collected_record_count: int = 0
    failed_record_count: int = 0
    completed_pr_numbers: list[int] = Field(
        default_factory=list,
    )
    updated_at: datetime
    status: str = "in_progress"


class CollectionSummary(BaseModel):
    """Summary of one data-collection run."""

    repository: str
    collection_name: str
    records_collected: int
    records_failed: int
    duplicates_removed: int
    output_file: str
    checkpoint_file: str
    failure_file: str
    started_at: datetime
    completed_at: datetime
    status: str
