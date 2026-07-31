"""Changed-file categorization utilities."""

from pathlib import PurePosixPath

TEST_PATH_PATTERNS = (
    "tests/",
    "test/",
    "__tests__/",
)

TEST_NAME_PATTERNS = (
    "test_",
    "_test.",
    ".test.",
    ".spec.",
)

DOCUMENTATION_EXTENSIONS = {
    ".md",
    ".mdx",
    ".rst",
    ".txt",
    ".adoc",
}

DOCUMENTATION_PATH_PATTERNS = (
    "docs/",
    "documentation/",
)

CONFIGURATION_FILE_NAMES = {
    "pyproject.toml",
    "setup.cfg",
    "setup.py",
    "tox.ini",
    "pytest.ini",
    "requirements.txt",
    "package.json",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    ".gitignore",
    ".dockerignore",
    "makefile",
}

CONFIGURATION_EXTENSIONS = {
    ".toml",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".json",
    ".xml",
}

SECURITY_PATH_KEYWORDS = {
    "auth",
    "authentication",
    "authorization",
    "permission",
    "permissions",
    "session",
    "sessions",
    "token",
    "tokens",
    "secret",
    "secrets",
    "credential",
    "credentials",
    "crypto",
    "encryption",
    "password",
    "oauth",
    "jwt",
    "csrf",
    "cookie",
    "cookies",
    "dependency",
    "dependencies",
}

GENERATED_PATH_PATTERNS = (
    "dist/",
    "build/",
    "vendor/",
    "generated/",
    "migrations/versions/",
)

GENERATED_FILE_NAMES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
}


def normalize_file_path(file_path: str) -> str:
    """Normalize a GitHub path for rule matching."""

    return file_path.replace("\\", "/").strip().lower()


def get_file_extension(
    file_path: str,
) -> str | None:
    """Return a lowercase file extension."""

    extension = PurePosixPath(normalize_file_path(file_path)).suffix.lower()

    return extension or None


def is_test_file(file_path: str) -> bool:
    """Determine whether a path appears to be a test file."""

    normalized_path = normalize_file_path(file_path)

    file_name = PurePosixPath(normalized_path).name

    return any(pattern in normalized_path for pattern in TEST_PATH_PATTERNS) or any(
        pattern in file_name for pattern in TEST_NAME_PATTERNS
    )


def is_documentation_file(
    file_path: str,
) -> bool:
    """Determine whether a path is documentation."""

    normalized_path = normalize_file_path(file_path)

    extension = get_file_extension(normalized_path)

    return extension in DOCUMENTATION_EXTENSIONS or any(
        normalized_path.startswith(pattern) for pattern in DOCUMENTATION_PATH_PATTERNS
    )


def is_configuration_file(
    file_path: str,
) -> bool:
    """Determine whether a path is configuration-related."""

    normalized_path = normalize_file_path(file_path)

    file_name = PurePosixPath(normalized_path).name

    extension = get_file_extension(normalized_path)

    return (
        file_name in CONFIGURATION_FILE_NAMES
        or extension in CONFIGURATION_EXTENSIONS
        or normalized_path.startswith(".github/")
    )


def is_security_sensitive_file(
    file_path: str,
) -> bool:
    """Identify files that may need security-focused review."""

    normalized_path = normalize_file_path(file_path)

    path_parts = {part for part in PurePosixPath(normalized_path).parts}

    file_stem = PurePosixPath(normalized_path).stem

    candidate_terms = path_parts | {file_stem}

    return any(
        keyword in candidate
        for candidate in candidate_terms
        for keyword in SECURITY_PATH_KEYWORDS
    )


def is_generated_file(
    file_path: str,
) -> bool:
    """Determine whether a file is probably generated."""

    normalized_path = normalize_file_path(file_path)

    file_name = PurePosixPath(normalized_path).name

    return file_name in GENERATED_FILE_NAMES or any(
        normalized_path.startswith(pattern) for pattern in GENERATED_PATH_PATTERNS
    )


def categorize_file(file_path: str) -> str:
    """Assign one primary category to a changed file."""

    if is_test_file(file_path):
        return "test"

    if is_configuration_file(file_path):
        return "configuration"

    if is_security_sensitive_file(file_path):
        return "security_sensitive"

    if is_documentation_file(file_path):
        return "documentation"

    if is_generated_file(file_path):
        return "generated"

    extension = get_file_extension(file_path)

    if extension == ".py":
        return "python_source"

    if extension in {
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
    }:
        return "javascript_typescript"

    if extension in {
        ".html",
        ".htm",
        ".css",
        ".scss",
    }:
        return "web_frontend"

    if extension in {
        ".sql",
        ".db",
    }:
        return "database"

    return "other"
