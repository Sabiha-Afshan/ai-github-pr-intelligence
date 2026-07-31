"""Verify the local portfolio-project environment."""

import importlib.util
import sys

from src.config.settings import get_settings
from src.github.client import GitHubClient
from src.github.errors import GitHubClientError
from src.utils.paths import (
    PROJECT_ROOT,
    REQUIRED_DIRECTORIES,
    create_required_directories,
)


def print_result(
    status: str,
    check_name: str,
    message: str,
) -> None:
    """Print one verification result."""

    print(f"{status:<7} | {check_name:<25} | {message}")


def check_python_version() -> bool:
    """Confirm Python 3.12 or newer."""

    is_valid = sys.version_info >= (3, 12)

    print_result(
        "PASS" if is_valid else "FAIL",
        "Python version",
        sys.version.split()[0],
    )

    return is_valid


def check_directories() -> bool:
    """Confirm required directories exist."""

    create_required_directories()

    missing_directories = [
        directory for directory in REQUIRED_DIRECTORIES if not directory.exists()
    ]

    is_valid = not missing_directories

    print_result(
        "PASS" if is_valid else "FAIL",
        "Project directories",
        (
            "All required directories exist."
            if is_valid
            else f"Missing: {missing_directories}"
        ),
    )

    return is_valid


def check_streamlit_import() -> bool:
    """Confirm Streamlit is installed."""

    is_valid = importlib.util.find_spec("streamlit") is not None

    print_result(
        "PASS" if is_valid else "FAIL",
        "Streamlit",
        ("Installed." if is_valid else "Package not found."),
    )

    return is_valid


def check_github() -> bool:
    """Confirm GitHub token and repository access."""

    settings = get_settings()

    if not settings.github_token:
        print_result(
            "FAIL",
            "GitHub connection",
            "GITHUB_TOKEN is missing.",
        )

        return False

    try:
        with GitHubClient(settings=settings) as client:
            repository = client.get_repository(
                owner=settings.github_repository_owner,
                repository=(settings.github_repository_name),
            )

    except GitHubClientError as error:
        print_result(
            "FAIL",
            "GitHub connection",
            str(error),
        )

        return False

    print_result(
        "PASS",
        "GitHub connection",
        repository.get(
            "full_name",
            "Repository retrieved.",
        ),
    )

    return True


def check_application_file() -> bool:
    """Confirm Streamlit entry point exists."""

    application_file = PROJECT_ROOT / "app.py"

    is_valid = application_file.exists()

    print_result(
        "PASS" if is_valid else "FAIL",
        "Streamlit entry point",
        str(application_file),
    )

    return is_valid


def main() -> int:
    """Run all project verification checks."""

    print("AI GitHub PR Intelligence verification")
    print("=" * 70)

    results = [
        check_python_version(),
        check_directories(),
        check_streamlit_import(),
        check_application_file(),
        check_github(),
    ]

    print("=" * 70)

    if all(results):
        print("Overall result: PASS")
        return 0

    print("Overall result: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
