"""Tests for changed-file categories."""

from src.features.file_categories import (
    categorize_file,
    is_configuration_file,
    is_documentation_file,
    is_generated_file,
    is_security_sensitive_file,
    is_test_file,
)


def test_python_test_file_is_detected() -> None:
    """Confirm Python test files are detected."""

    assert is_test_file("tests/test_app.py")

    assert categorize_file("tests/test_app.py") == "test"


def test_javascript_test_file_is_detected() -> None:
    """Confirm JavaScript tests are detected."""

    assert is_test_file("src/app.test.ts")


def test_documentation_file_is_detected() -> None:
    """Confirm documentation files are detected."""

    assert is_documentation_file("docs/index.rst")

    assert categorize_file("README.md") == "documentation"


def test_configuration_file_is_detected() -> None:
    """Confirm configuration files are detected."""

    assert is_configuration_file("pyproject.toml")

    assert is_configuration_file(".github/workflows/tests.yml")


def test_security_file_is_detected() -> None:
    """Confirm security-sensitive paths are detected."""

    assert is_security_sensitive_file("src/auth/token_service.py")


def test_generated_file_is_detected() -> None:
    """Confirm generated files are detected."""

    assert is_generated_file("package-lock.json")
