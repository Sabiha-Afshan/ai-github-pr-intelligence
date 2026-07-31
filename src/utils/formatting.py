"""Formatting helpers used by Streamlit pages."""

from datetime import datetime
from typing import Any


def format_percentage(
    value: float | int | None,
    decimal_places: int = 2,
) -> str:
    """Convert a decimal value into a percentage string."""

    if value is None:
        return "Unavailable"

    return f"{float(value) * 100:.{decimal_places}f}%"


def format_number(
    value: float | int | None,
    decimal_places: int = 0,
) -> str:
    """Format a numeric value for display."""

    if value is None:
        return "Unavailable"

    return f"{float(value):,.{decimal_places}f}"


def format_datetime(
    value: datetime | str | None,
) -> str:
    """Format a datetime-like value for display."""

    if value is None:
        return "Unavailable"

    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M UTC")

    return str(value)


def safe_display_value(
    value: Any,
) -> str:
    """Return a safe human-readable value."""

    if value is None:
        return "Unavailable"

    if isinstance(value, str) and not value.strip():
        return "Unavailable"

    return str(value)
