from typing import Any, Dict, Pattern


def get_values_from_matching_patterns(patterns: Dict[Pattern, Any], text: str) -> list:
    values = []
    for pattern, value in patterns.items():
        match = pattern.match(text)
        if match:
            values.append(value)
    return values
