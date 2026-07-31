"""Constants shared across the application."""

APPLICATION_TITLE = "AI GitHub PR Intelligence"

OUTCOME_LABELS = {
    0: "Closed without merge",
    1: "Merged",
}

RISK_LEVELS = {
    "low": (0, 29),
    "medium": (30, 59),
    "high": (60, 79),
    "critical": (80, 100),
}

CONFIDENCE_LEVELS = (
    "Low",
    "Medium",
    "High",
)

POLICY_STATUSES = (
    "Passed",
    "Failed",
    "Warning",
    "Not applicable",
    "Unable to determine",
)

DEFAULT_REPOSITORY = "pallets/flask"
