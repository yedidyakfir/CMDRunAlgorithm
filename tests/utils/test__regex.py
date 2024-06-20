import re

import pytest

from runner.utils.regex import get_values_from_matching_patterns


@pytest.mark.parametrize(
    ["text", "expected"],
    [
        [".classA", ["Any"]],
        ["CLASss.classB.asdfas", ["Dict"]],
        [".classB.classA", ["Any", "Dict"]],
        ["classA", []],
    ],
)
def test__get_values_from_matching_patterns__sanity(text, expected):
    # Arrange
    patterns = {
        re.compile(r".*\.classA"): "Any",
        re.compile(r".*\.classB\.*"): "Dict",
    }

    # Act + Assert
    assert get_values_from_matching_patterns(patterns, text) == expected
