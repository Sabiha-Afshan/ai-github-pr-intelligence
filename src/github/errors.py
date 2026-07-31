"""Custom exceptions for GitHub integration."""


class GitHubClientError(Exception):
    """Base error raised by the GitHub client."""


class GitHubAuthenticationError(GitHubClientError):
    """Raised when GitHub authentication fails."""


class GitHubNotFoundError(GitHubClientError):
    """Raised when a GitHub resource is not found."""


class GitHubRateLimitError(GitHubClientError):
    """Raised when a GitHub API rate limit is exhausted."""

    def __init__(
        self,
        message: str,
        reset_timestamp: int | None = None,
    ) -> None:
        super().__init__(message)

        self.reset_timestamp = reset_timestamp


class GitHubResponseError(GitHubClientError):
    """Raised when GitHub returns an unexpected response."""
