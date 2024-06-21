import inspect
import re
from unittest.mock import MagicMock

import pytest
from torch.optim import Adam

from runner.parameters_analysis import (
    need_params_for_signature,
    get_full_signature_parameters,
    needed_parameters_for_creation,
    ParameterCLI,
)


class MockA:
    def __init__(self, a: int, aa: str):
        pass


class MockB:
    def __init__(self, a: int, b: str):
        pass

    def func_name(self, e: int, f: str):
        pass


class MockC(MockB):
    def __init__(self, a: int, b: str, c: float):
        pass

    def func_name(self, a: int, b: MockA, c: float = 0.2, *args, **kwargs):
        pass


@pytest.mark.parametrize(
    ["obj", "add_options_from_outside_packages", "expected"],
    [
        (1, True, False),
        (1, False, False),
        (int, True, False),
        (MockA, True, True),
        (MockA, False, True),
        (object, False, False),
        (Adam, True, True),
        (Adam, False, False),
    ],
)
def test__need_params_for_signature__sanity(obj, add_options_from_outside_packages, expected):
    # Arrange
    if inspect.isclass(obj) and inspect.getmodule(obj).__name__ == __name__:
        obj.__module__ = "runner.parameters_analysis"

    # Act
    result = need_params_for_signature(obj, add_options_from_outside_packages)

    # Assert
    assert result == expected


def test__get_full_signature_parameters__sanity():
    # Act
    result = get_full_signature_parameters(MockC, MockC)

    # Assert
    assert result == {
        "a": inspect.Parameter("a", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=int),
        "b": inspect.Parameter("b", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str),
        "c": inspect.Parameter("c", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=float),
    }


def test__get_full_signature_parameters__new_function():
    # Act
    result = get_full_signature_parameters(MockC, MockC, "func_name")

    # Assert
    assert result == {
        "self": inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        "args": inspect.Parameter("args", inspect.Parameter.VAR_POSITIONAL),
        "kwargs": inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD),
        "a": inspect.Parameter("a", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=int),
        "b": inspect.Parameter("b", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=MockA),
        "c": inspect.Parameter(
            "c", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=float, default=0.2
        ),
    }


def test__needed_parameters_for_creation__sanity():
    # Arrange
    key_value_config = {
        "a_type": MockB,
        "a": {"b": "BBBBB"},
        "b_type": str,
        "c": 0.1,
        "f": "F",
        "b": "bbb",
    }
    regex_config = {re.compile(r".*\.a$"): 12, re.compile(r"^f_type$"): MockA}
    signature_name = "func_name"
    expected = {
        "a": ParameterCLI(
            type=MockB,
            default=None,
            requirements={
                "b": ParameterCLI(type=str, default="BBBBB", requirements={}),
                "a": ParameterCLI(type=int, default=12, requirements={}),
            },
        ),
        "b": ParameterCLI(
            type=str,
            default="bbb",
            requirements={},
        ),
        "c": ParameterCLI(type=float, default=0.1, requirements={}),
        "f": ParameterCLI(
            type=MockA,
            default=None,
            requirements={
                "a": ParameterCLI(type=int, default=12, requirements={}),
                "aa": ParameterCLI(type=str, default=None, requirements={}),
            },
        ),
        "e": ParameterCLI(type=int, default=None, requirements={}),
    }
    logger = MagicMock()

    # Act
    result = needed_parameters_for_creation(
        MockC, signature_name, key_value_config, regex_config, True, logger=logger
    )

    # Assert
    logger.warning.assert_not_called()
    assert result == expected


def test__needed_parameters_for_creation__warning_fur_multiple_matching_rules():
    # Arrange
    regex_config = {re.compile(r".*\.a$"): 12, re.compile(r"^b\.a$"): MockA}
    logger = MagicMock()

    # Act
    needed_parameters_for_creation(MockC, "func_name", {}, regex_config, True, logger=logger)

    # Assert
    logger.warning.assert_called_once()


# TODO -
