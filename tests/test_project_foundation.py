"""Tests for the project foundation."""

from pathlib import Path

from src.config.settings import get_settings
from src.utils.formatting import (
    format_number,
    format_percentage,
)
from src.utils.helpers import calculate_percentage
from src.utils.paths import (
    PROJECT_ROOT,
    REQUIRED_DIRECTORIES,
    create_required_directories,
)


def test_project_root_exists() -> None:
    """Confirm that the resolved project root exists."""

    assert PROJECT_ROOT.exists()
    assert PROJECT_ROOT.is_dir()


def test_required_directories_can_be_created() -> None:
    """Confirm that required directories exist."""

    create_required_directories()

    for directory in REQUIRED_DIRECTORIES:
        assert directory.exists()
        assert directory.is_dir()


def test_settings_load() -> None:
    """Confirm that application settings load."""

    settings = get_settings()

    assert settings.app_name == "AI GitHub PR Intelligence"

    assert settings.github_repository_full_name == "pallets/flask"


def test_percentage_calculation() -> None:
    """Confirm safe percentage calculation."""

    assert calculate_percentage(1, 4) == 0.25
    assert calculate_percentage(1, 0) == 0.0


def test_percentage_formatting() -> None:
    """Confirm percentage formatting."""

    assert format_percentage(0.25) == "25.00%"
    assert format_percentage(None) == "Unavailable"


def test_number_formatting() -> None:
    """Confirm numeric formatting."""

    assert format_number(1470) == "1,470"
    assert format_number(None) == "Unavailable"


def test_main_application_file_exists() -> None:
    """Confirm that the Streamlit entry point exists."""

    application_file = PROJECT_ROOT / "app.py"

    assert isinstance(
        application_file,
        Path,
    )

    assert application_file.exists()
