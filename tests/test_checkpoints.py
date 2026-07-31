"""Tests for collection checkpoints."""

from datetime import UTC, datetime

from src.data.checkpoints import (
    load_checkpoint,
    save_checkpoint,
)
from src.github.schemas import (
    CollectionCheckpoint,
)


def test_new_checkpoint_is_created(
    tmp_path,
) -> None:
    """Confirm a missing checkpoint returns defaults."""

    checkpoint_path = tmp_path / "checkpoint.json"

    checkpoint = load_checkpoint(
        checkpoint_path=checkpoint_path,
        repository="pallets/flask",
        collection_name="test_collection",
    )

    assert checkpoint.last_completed_page == 0
    assert checkpoint.collected_record_count == 0
    assert checkpoint.status == "in_progress"


def test_checkpoint_can_be_saved_and_loaded(
    tmp_path,
) -> None:
    """Confirm checkpoint persistence."""

    checkpoint_path = tmp_path / "checkpoint.json"

    checkpoint = CollectionCheckpoint(
        repository="pallets/flask",
        collection_name="test_collection",
        last_completed_page=2,
        collected_record_count=150,
        completed_pr_numbers=[
            1,
            2,
            3,
        ],
        updated_at=datetime.now(UTC),
    )

    save_checkpoint(
        checkpoint,
        checkpoint_path,
    )

    loaded_checkpoint = load_checkpoint(
        checkpoint_path=checkpoint_path,
        repository="pallets/flask",
        collection_name="test_collection",
    )

    assert loaded_checkpoint.last_completed_page == 2

    assert loaded_checkpoint.collected_record_count == 150

    assert loaded_checkpoint.completed_pr_numbers == [1, 2, 3]
