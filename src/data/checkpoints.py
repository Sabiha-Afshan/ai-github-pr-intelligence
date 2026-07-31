"""Checkpoint persistence for resumable collection."""

import json
from datetime import UTC, datetime
from pathlib import Path

from src.github.schemas import (
    CollectionCheckpoint,
    DetailedExtractionCheckpoint,
)


def load_checkpoint(
    checkpoint_path: Path,
    repository: str,
    collection_name: str,
) -> CollectionCheckpoint:
    """Load a checkpoint or create a new one."""

    if not checkpoint_path.exists():
        return CollectionCheckpoint(
            repository=repository,
            collection_name=collection_name,
            updated_at=datetime.now(UTC),
        )

    with checkpoint_path.open(
        "r",
        encoding="utf-8",
    ) as checkpoint_file:
        payload = json.load(checkpoint_file)

    return CollectionCheckpoint.model_validate(
        payload
    )


def save_checkpoint(
    checkpoint: CollectionCheckpoint,
    checkpoint_path: Path,
) -> None:
    """Save a checkpoint safely."""

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = checkpoint_path.with_suffix(
        checkpoint_path.suffix + ".tmp"
    )

    checkpoint.updated_at = datetime.now(UTC)

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as checkpoint_file:
        json.dump(
            checkpoint.model_dump(
                mode="json"
            ),
            checkpoint_file,
            indent=2,
            ensure_ascii=False,
        )

    temporary_path.replace(
        checkpoint_path
    )


def load_detailed_checkpoint(
    checkpoint_path: Path,
    repository: str,
    collection_name: str,
) -> DetailedExtractionCheckpoint:
    """Load or create a detailed extraction checkpoint."""

    if not checkpoint_path.exists():
        return DetailedExtractionCheckpoint(
            repository=repository,
            collection_name=collection_name,
            updated_at=datetime.now(UTC),
        )

    with checkpoint_path.open(
        "r",
        encoding="utf-8",
    ) as checkpoint_file:
        payload = json.load(
            checkpoint_file
        )

    return (
        DetailedExtractionCheckpoint
        .model_validate(payload)
    )


def save_detailed_checkpoint(
    checkpoint: DetailedExtractionCheckpoint,
    checkpoint_path: Path,
) -> None:
    """Atomically save a detailed extraction checkpoint."""

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        checkpoint_path.with_suffix(
            checkpoint_path.suffix + ".tmp"
        )
    )

    checkpoint.updated_at = datetime.now(
        UTC
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as checkpoint_file:
        json.dump(
            checkpoint.model_dump(
                mode="json"
            ),
            checkpoint_file,
            indent=2,
            ensure_ascii=False,
        )

    temporary_path.replace(
        checkpoint_path
    )