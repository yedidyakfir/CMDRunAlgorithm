import inspect
import re
from unittest.mock import MagicMock

import pytest
from torch.optim import Adam, SGD

from runner.object_creation import ParameterNode
from runner.parameters_analysis import (
    need_params_for_signature,
    get_full_signature_parameters,
    needed_parameters_for_calling,
    cli_parameters_for_calling,
    CliParam,
    Rules,
)
from tests import mock_module
from tests.conftest import EXPECTED_GRAPH, create_opt
from tests.mock_module.a import MockA, MockB
from tests.mock_module.sub_mock_module.b import MockC, MockE, MockG, MockF
from tests.mock_module.utils import func


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
    # Arrange
    expected = {
        "dd": inspect.Parameter("dd", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=int),
        "self": inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        "kwargs": inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD),
    }

    # Act
    results = get_full_signature_parameters(MockE, MockE, "func_name")

    # Assert
    assert results == expected


def test__needed_parameters_for_creation__sanity():
    # Arrange
    key_value_config = {
        "a--type": "MockB",
        "a": {"b--type": "torch.optim.SGD", "a": "None"},
        "b--type": str,
        "c": 0.1,
        "b": "bbb",
    }
    regex_config = Rules(
        value_rules={re.compile(r".*\.a$"): 12}, type_rules={re.compile(r"^f--type$"): MockA}
    )
    signature_name = "func_name"
    expected = EXPECTED_GRAPH
    logger = MagicMock()

    # Act
    result = needed_parameters_for_calling(
        MockC,
        signature_name,
        {},
        key_value_config,
        Rules(),
        regex_config,
        mock_module,
        True,
        logger=logger,
    )

    # Assert
    logger.warning.assert_not_called()
    assert result == expected


@pytest.mark.parametrize(
    ["default_config", "config", "default_rules", "rules", "expected"],
    [
        [
            {"a--type": "torch.optim.SGD"},
            {"a--type": "MockB"},
            Rules(),
            Rules(),
            {"a": ParameterNode(type=MockB, value=None, edges={})},
        ],
        [
            {"a--const": "torch.optim.SGD"},
            {"a--type": "MockB"},
            Rules(),
            Rules(),
            {"a": ParameterNode(type=MockB, value=SGD, edges={})},
        ],
        [
            {},
            {"a--const": "torch.optim.SGD"},
            Rules(),
            Rules(),
            {"a": ParameterNode(type=int, value=SGD, edges={})},
        ],
        [
            {"b": 1},
            {"b": 2},
            Rules(),
            Rules(),
            {"b": ParameterNode(type=str, value=2, edges={})},
        ],
        [
            {"b": 1},
            {"c": 2},
            Rules(),
            Rules(),
            {
                "b": ParameterNode(type=str, value=1, edges={}),
                "c": ParameterNode(type=float, value=2, edges={}),
            },
        ],
        [
            {},
            {"b": 1, "b--connected_params": {"b.c": "c"}},
            Rules(),
            Rules(),
            {
                "b": ParameterNode(type=str, value=1, edges={"b.c": "c"}),
            },
        ],
        [
            {"c--type": "torch.optim.SGD", "c--connected_params": {"b.c": "c"}},
            {},
            Rules(),
            Rules(),
            {
                "c": ParameterNode(type=SGD, value=None, edges={"b.c": "c"}),
            },
        ],
        [
            {"c--type": "torch.optim.SGD", "c--creator": "func"},
            {},
            Rules(),
            Rules(),
            {
                "c": ParameterNode(type=SGD, value=None, edges={}, creator=func),
            },
        ],
        [
            {},
            {"c--type": "torch.optim.SGD", "c--creator": "tests.mock_module.utils.func"},
            Rules(),
            Rules(),
            {
                "c": ParameterNode(type=SGD, value=None, edges={}, creator=func),
            },
        ],
        [
            {},
            {},
            Rules(value_rules={re.compile(r"c$"): 12}),
            Rules(value_rules={re.compile(r"c$"): 11}),
            {"c": ParameterNode(type=float, value=11, edges={})},
        ],
        [
            {},
            {},
            Rules(type_rules={re.compile(r"c--type$"): MockA}),
            Rules(type_rules={re.compile(r"c--type$"): "torch.optim.SGD"}),
            {"c": ParameterNode(type=SGD, value=None, edges={})},
        ],
        [
            {},
            {},
            Rules(type_rules={re.compile(r"a--type$"): MockA}),
            Rules(value_rules={re.compile(r"c$"): "SGD"}),
            {
                "a": ParameterNode(type=MockA, value=None, edges={}),
                "c": ParameterNode(type=float, value="SGD", edges={}),
            },
        ],
        [
            {},
            {},
            Rules(value_rules={re.compile(r"a--const$"): MockA}),
            Rules(),
            {
                "a": ParameterNode(type=int, value=MockA, edges={}),
            },
        ],
        [
            {},
            {},
            Rules(
                value_rules={re.compile(r"b$"): 1},
                connected_params_rules={re.compile(r"b--connected_params$"): {"b.c": "c"}},
            ),
            Rules(),
            {
                "b": ParameterNode(type=str, value=1, edges={"b.c": "c"}),
            },
        ],
        [
            {},
            {},
            Rules(
                type_rules={re.compile("c--type"): "torch.optim.SGD"},
                connected_params_rules={re.compile("c--connected_params"): {"b.c": "c"}},
            ),
            Rules(),
            {
                "c": ParameterNode(type=SGD, value=None, edges={"b.c": "c"}),
            },
        ],
        [
            {},
            {},
            Rules(),
            Rules(
                type_rules={re.compile("c--type"): "torch.optim.SGD"},
                creator_rules={re.compile("c--creator"): "tests.mock_module.utils.func"},
            ),
            {
                "c": ParameterNode(type=SGD, value=None, edges={}, creator=func),
            },
        ],
        [
            {},
            {},
            Rules(),
            Rules(
                value_rules={re.compile("c--const"): "torch.optim.SGD"},
            ),
            {
                "c": ParameterNode(type=float, value=SGD, edges={}, creator=None),
            },
        ],
    ],
)
def test__needed_parameters_for_creation__check_rules_configs_and_default(
    default_config, config, default_rules, rules, expected
):
    # Act
    result = needed_parameters_for_calling(
        MockC, None, default_config, config, default_rules, rules, mock_module, True
    )

    # Assert
    assert result == expected


def test__needed_parameters_for_creation__check_annotation_doesnt_create_class():
    # Arrange
    class MockClass:
        def __init__(self, a: MockA):
            pass

    # Act
    result = needed_parameters_for_calling(
        MockClass, None, {}, {}, Rules(), Rules(), mock_module, True
    )

    # Assert
    assert result == {}


def test__needed_parameters_for_creation__warning_for_unmatching_value_and_type():
    # Arrange
    key_value_config = {"a": 12}
    regex_config = Rules(type_rules={re.compile(r"a--type$"): str})
    logger = MagicMock()

    # Act
    needed_parameters_for_calling(
        MockA,
        None,
        {},
        key_value_config,
        Rules(),
        regex_config,
        mock_module,
        True,
        logger=logger,
    )

    # Assert
    logger.warning.assert_called_once()


def test__needed_parameters_for_creation__warning_fur_multiple_matching_rules():
    # Arrange
    regex_config = Rules(value_rules={re.compile(r".*\.a$"): 12, re.compile(r"^b\.a$"): MockA})
    logger = MagicMock()

    # Act
    needed_parameters_for_calling(
        MockC,
        "func_name",
        {},
        {"b--init": True},
        Rules(),
        regex_config,
        mock_module,
        True,
        logger=logger,
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
                CliParam(type=str, multiple=False, default=None, name="a--type"),
                CliParam(type=str, multiple=True, default=None, name="a--connected_params"),
                CliParam(type=str, multiple=False, default=None, name="a--creator"),
                CliParam(type=int, multiple=False, default=None, name="a"),
                CliParam(type=str, multiple=False, default=None, name="a--const"),
                CliParam(type=str, multiple=False, default=None, name="b--type"),
                CliParam(type=str, multiple=True, default=None, name="b--connected_params"),
                CliParam(type=str, multiple=False, default=None, name="b--creator"),
                CliParam(type=str, multiple=False, default=None, name="b"),
                CliParam(type=str, multiple=False, default=None, name="b--const"),
            ],
        ],
        [
            MockC,
            "func_name",
            True,
            [
                CliParam(type=str, multiple=False, default=None, name="e--type"),
                CliParam(type=str, multiple=True, default=None, name="e--connected_params"),
                CliParam(type=str, multiple=False, default=None, name="e--creator"),
                CliParam(type=int, multiple=False, default=None, name="e"),
                CliParam(type=str, multiple=False, default=None, name="e--const"),
                CliParam(type=str, multiple=False, default=None, name="f--type"),
                CliParam(type=str, multiple=True, default=None, name="f--connected_params"),
                CliParam(type=str, multiple=False, default=None, name="f--creator"),
                CliParam(type=str, multiple=False, default=None, name="f"),
                CliParam(type=str, multiple=False, default=None, name="f--const"),
                CliParam(type=str, multiple=False, default=None, name="a--type"),
                CliParam(type=str, multiple=True, default=None, name="a--connected_params"),
                CliParam(type=str, multiple=False, default=None, name="a--creator"),
                CliParam(type=int, multiple=False, default=None, name="a"),
                CliParam(type=str, multiple=False, default=None, name="a--const"),
                CliParam(type=str, multiple=False, default=None, name="b--type"),
                CliParam(type=str, multiple=True, default=None, name="b--connected_params"),
                CliParam(type=str, multiple=False, default=None, name="b--creator"),
                CliParam(type=bool, multiple=False, default=None, name="b--init", flag=True),
                CliParam(type=str, multiple=False, default=None, name="b.a--type"),
                CliParam(type=str, multiple=True, default=None, name="b.a--connected_params"),
                CliParam(type=str, multiple=False, default=None, name="b.a--creator"),
                CliParam(type=int, multiple=False, default=None, name="b.a"),
                CliParam(type=str, multiple=False, default=None, name="b.a--const"),
                CliParam(type=str, multiple=False, default=None, name="b.aa--type"),
                CliParam(type=str, multiple=True, default=None, name="b.aa--connected_params"),
                CliParam(type=str, multiple=False, default=None, name="b.aa--creator"),
                CliParam(type=str, multiple=False, default=None, name="b.aa"),
                CliParam(type=str, multiple=False, default=None, name="b.aa--const"),
                CliParam(type=str, multiple=False, default=None, name="c--type"),
                CliParam(type=str, multiple=True, default=None, name="c--connected_params"),
                CliParam(type=str, multiple=False, default=None, name="c--creator"),
                CliParam(type=float, multiple=False, default=None, name="c"),
                CliParam(type=str, multiple=False, default=None, name="c--const"),
            ],
        ],
        [
            MockG,
            "func_name",
            True,
            [
                CliParam(type=str, multiple=False, default=None, name="opt--type"),
                CliParam(type=str, multiple=True, default=None, name="opt--connected_params"),
                CliParam(type=str, multiple=False, default=None, name="opt--creator"),
                CliParam(type=bool, multiple=False, default=None, name="opt--init", flag=True),
                CliParam(type=str, multiple=False, default=None, name="opt.params--type"),
                CliParam(
                    type=str, multiple=True, default=None, name="opt.params--connected_params"
                ),
                CliParam(type=str, multiple=False, default=None, name="opt.params--creator"),
                CliParam(type=None, multiple=False, default=None, name="opt.params"),
                CliParam(type=str, multiple=False, default=None, name="opt.params--const"),
                CliParam(type=str, multiple=False, default=None, name="opt.lr--type"),
                CliParam(
                    type=str, multiple=True, default=None, name="opt.lr--connected_params"
                ),
                CliParam(type=str, multiple=False, default=None, name="opt.lr--creator"),
                CliParam(type=None, multiple=False, default=None, name="opt.lr"),
                CliParam(type=str, multiple=False, default=None, name="opt.lr--const"),
                CliParam(type=str, multiple=False, default=None, name="opt.momentum--type"),
                CliParam(
                    type=str,
                    multiple=True,
                    default=None,
                    name="opt.momentum--connected_params",
                ),
                CliParam(type=str, multiple=False, default=None, name="opt.momentum--creator"),
                CliParam(type=None, multiple=False, default=None, name="opt.momentum"),
                CliParam(type=str, multiple=False, default=None, name="opt.momentum--const"),
                CliParam(type=str, multiple=False, default=None, name="opt.dampening--type"),
                CliParam(
                    type=str,
                    multiple=True,
                    default=None,
                    name="opt.dampening--connected_params",
                ),
                CliParam(
                    type=str, multiple=False, default=None, name="opt.dampening--creator"
                ),
                CliParam(type=None, multiple=False, default=None, name="opt.dampening"),
                CliParam(type=str, multiple=False, default=None, name="opt.dampening--const"),
                CliParam(type=str, multiple=False, default=None, name="opt.weight_decay--type"),
                CliParam(
                    type=str,
                    multiple=True,
                    default=None,
                    name="opt.weight_decay--connected_params",
                ),
                CliParam(
                    type=str, multiple=False, default=None, name="opt.weight_decay--creator"
                ),
                CliParam(type=None, multiple=False, default=None, name="opt.weight_decay"),
                CliParam(type=str, multiple=False, default=None, name="opt.weight_decay--const"),
                CliParam(type=str, multiple=False, default=None, name="opt.nesterov--type"),
                CliParam(
                    type=str,
                    multiple=True,
                    default=None,
                    name="opt.nesterov--connected_params",
                ),
                CliParam(type=str, multiple=False, default=None, name="opt.nesterov--creator"),
                CliParam(type=None, multiple=False, default=None, name="opt.nesterov"),
                CliParam(type=str, multiple=False, default=None, name="opt.nesterov--const"),
                CliParam(type=str, multiple=False, default=None, name="opt.maximize--type"),
                CliParam(
                    type=str,
                    multiple=True,
                    default=None,
                    name="opt.maximize--connected_params",
                ),
                CliParam(type=str, multiple=False, default=None, name="opt.maximize--creator"),
                CliParam(type=bool, multiple=False, default=None, name="opt.maximize"),
                CliParam(type=str, multiple=False, default=None, name="opt.maximize--const"),
                CliParam(type=str, multiple=False, default=None, name="opt.foreach--type"),
                CliParam(
                    type=str, multiple=True, default=None, name="opt.foreach--connected_params"
                ),
                CliParam(type=str, multiple=False, default=None, name="opt.foreach--creator"),
                CliParam(type=None, multiple=False, default=None, name="opt.foreach"),
                CliParam(type=str, multiple=False, default=None, name="opt.foreach--const"),
                CliParam(
                    type=str, multiple=False, default=None, name="opt.differentiable--type"
                ),
                CliParam(
                    type=str,
                    multiple=True,
                    default=None,
                    name="opt.differentiable--connected_params",
                ),
                CliParam(
                    type=str, multiple=False, default=None, name="opt.differentiable--creator"
                ),
                CliParam(type=bool, multiple=False, default=None, name="opt.differentiable"),
                CliParam(type=str, multiple=False, default=None, name="opt.differentiable--const"),
                CliParam(type=str, multiple=False, default=None, name="eps--type"),
                CliParam(type=str, multiple=True, default=None, name="eps--connected_params"),
                CliParam(type=str, multiple=False, default=None, name="eps--creator"),
                CliParam(type=None, multiple=False, default=None, name="eps"),
                CliParam(type=str, multiple=False, default=None, name="eps--const"),
            ],
        ],
        [
            MockF,
            "func_name",
            True,
            [
                CliParam(type=str, multiple=False, default=None, name="opt--type"),
                CliParam(type=str, multiple=True, default=None, name="opt--connected_params"),
                CliParam(type=str, multiple=False, default=None, name="opt--creator"),
                CliParam(type=None, multiple=False, default=None, name="opt"),
                CliParam(type=str, multiple=False, default=None, name="opt--const"),
            ],
        ],
    ],
)
def test__cli_parameters_for_calling__sanity(klass, signature_name, outside_classes, expected):
    # Act
    results = cli_parameters_for_calling(klass, signature_name, outside_classes, mock_module)

    # Assert
    assert results == expected
