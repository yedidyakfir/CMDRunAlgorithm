from logging import Logger
from typing import Any, Dict, Pattern


def get_values_from_matching_patterns(patterns: Dict[Pattern, Any], text: str) -> list:
    values = []
    for pattern, value in patterns.items():
        match = pattern.match(text)
        if match:
            values.append(value)
    return values


def get_first_value_for_matching_patterns(
    patterns: Dict[Pattern, Any], text: str, logger: Logger
) -> Any:
    values = get_values_from_matching_patterns(patterns, text)
    if len(values) > 1:
        logger.warning(f"Multiple values found for {text} param. Using the first one.")
    return values[0] if values else None
