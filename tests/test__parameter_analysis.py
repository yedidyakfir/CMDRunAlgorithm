import inspect
import re
from unittest.mock import MagicMock

import pytest
from torch.optim import Adam

from runner.parameters_analysis import (
    need_params_for_signature,
    get_full_signature_parameters,
    needed_parameters_for_calling,
    cli_parameters_for_calling,
    CliParam,
)
from tests import mock_module
from tests.conftest import EXPECTED_GRAPH
from tests.mock_module.a import MockA, MockB
from tests.mock_module.sub_mock_module.b import MockC, MockE, MockG, MockF


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
    if (
        inspect.isclass(obj)
        and inspect.getmodule(obj).__name__.split(".")[0] == __name__.split(".")[0]
    ):
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


def test__get_full_signature_parameters__stops_at_class_with_no_signature_name():
    # Act
    results = get_full_signature_parameters(MockE, MockE, "func_name")

    # Assert
    assert results == {
        "dd": inspect.Parameter("dd", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=int),
        "self": inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    }


def test__needed_parameters_for_creation__sanity():
    # Arrange
    key_value_config = {
        "a_type": "MockB",
        "a": {"b_type": "torch.optim.SGD", "a": "None"},
        "b_type": str,
        "c": 0.1,
        "b": "bbb",
    }
    regex_config = {re.compile(r".*\.a$"): 12, re.compile(r"^f_type$"): MockA}
    signature_name = "func_name"
    expected = EXPECTED_GRAPH
    logger = MagicMock()

    # Act
    result = needed_parameters_for_calling(
        MockC, signature_name, key_value_config, regex_config, mock_module, True, logger=logger
    )

    # Assert
    logger.warning.assert_not_called()
    assert result == expected


def test__needed_parameters_for_creation__warning_for_unmatching_value_and_type():
    # Arrange
    key_value_config = {"a": {"b": "BBBBB"}}
    regex_config = {re.compile(r"a_type$"): str}
    logger = MagicMock()

    # Act
    needed_parameters_for_calling(
        MockA, None, key_value_config, regex_config, mock_module, True, logger=logger
    )

    # Assert
    logger.warning.assert_called_once()


def test__needed_parameters_for_creation__warning_fur_multiple_matching_rules():
    # Arrange
    regex_config = {re.compile(r".*\.a$"): 12, re.compile(r"^b\.a$"): MockA}
    logger = MagicMock()

    # Act
    needed_parameters_for_calling(
        MockC, "func_name", {}, regex_config, mock_module, True, logger=logger
    )

    # Assert
    logger.warning.assert_called_once()


@pytest.mark.parametrize(
    ["klass", "signature_name", "outside_classes", "expected"],
    [
        [
            MockB,
            None,
            True,
            [
                CliParam(type=str, multiple=False, default=None, name="a_type"),
                CliParam(type=int, multiple=False, default=None, name="a"),
                CliParam(type=str, multiple=False, default=None, name="b_type"),
                CliParam(type=str, multiple=False, default=None, name="b"),
            ],
        ],
        [
            MockC,
            "func_name",
            True,
            [
                CliParam(type=str, multiple=False, default=None, name="e_type"),
                CliParam(type=int, multiple=False, default=None, name="e"),
                CliParam(type=str, multiple=False, default=None, name="f_type"),
                CliParam(type=str, multiple=False, default=None, name="f"),
                CliParam(type=str, multiple=False, default=None, name="a_type"),
                CliParam(type=int, multiple=False, default=None, name="a"),
                CliParam(type=str, multiple=False, default=None, name="b_type"),
                CliParam(type=str, multiple=False, default=None, name="b.a_type"),
                CliParam(type=int, multiple=False, default=None, name="b.a"),
                CliParam(type=str, multiple=False, default=None, name="b.aa_type"),
                CliParam(type=str, multiple=False, default=None, name="b.aa"),
                CliParam(type=str, multiple=False, default=None, name="c_type"),
                CliParam(type=float, multiple=False, default=None, name="c"),
            ],
        ],
        [
            MockG,
            "func_name",
            True,
            [
                CliParam(type=str, multiple=False, default=None, name="opt_type"),
                CliParam(type=str, multiple=False, default=None, name="opt.params_type"),
                CliParam(type=str, multiple=False, default=None, name="opt.defaults_type"),
                CliParam(type=str, multiple=False, default=None, name="opt.defaults"),
                CliParam(type=str, multiple=False, default=None, name="opt.lr_type"),
                CliParam(type=str, multiple=False, default=None, name="opt.momentum_type"),
                CliParam(type=str, multiple=False, default=None, name="opt.dampening_type"),
                CliParam(type=str, multiple=False, default=None, name="opt.weight_decay_type"),
                CliParam(type=str, multiple=False, default=None, name="opt.nesterov_type"),
                CliParam(type=str, multiple=False, default=None, name="opt.maximize_type"),
                CliParam(type=bool, multiple=False, default=None, name="opt.maximize"),
                CliParam(type=str, multiple=False, default=None, name="opt.foreach_type"),
                CliParam(type=str, multiple=False, default=None, name="opt.foreach"),
                CliParam(
                    type=str, multiple=False, default=None, name="opt.differentiable_type"
                ),
                CliParam(type=bool, multiple=False, default=None, name="opt.differentiable"),
                CliParam(type=str, multiple=False, default=None, name="eps_type"),
                CliParam(type=str, multiple=True, default=None, name="eps"),
            ],
        ],
        [
            MockF,
            "func_name",
            True,
            [
                CliParam(type=str, multiple=False, default=None, name="opt_type"),
                CliParam(type=str, multiple=False, default=None, name="opt"),
            ],
        ],
    ],
)
def test__cli_parameters_for_calling__sanity(klass, signature_name, outside_classes, expected):
    # Act
    results = cli_parameters_for_calling(klass, signature_name, outside_classes, mock_module)

    # Assert
    assert results == expected
