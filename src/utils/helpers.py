"""General helper functions."""

from collections.abc import Mapping
from typing import Any


def safely_get_nested_value(
    data: Mapping[str, Any],
    keys: list[str],
    default: Any = None,
) -> Any:
    """Safely retrieve a nested dictionary value."""

    current_value: Any = data

    for key in keys:
        if not isinstance(
            current_value,
            Mapping,
        ):
            return default

        if key not in current_value:
            return default

        current_value = current_value[key]

    return current_value


def calculate_percentage(
    numerator: int | float,
    denominator: int | float,
) -> float:
    """Calculate a safe decimal percentage."""

    if denominator == 0:
        return 0.0

    return float(numerator) / float(denominator)
