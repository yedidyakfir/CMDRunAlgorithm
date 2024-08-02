import inspect
import re
from unittest.mock import MagicMock

import pytest
import torch
from torch.optim import Adam, SGD

from runner.object_creation import ParameterNode
from runner.parameters_analysis import (
    need_params_for_signature,
    get_full_signature_parameters,
    needed_parameters_for_calling,
    cli_parameters_for_calling,
    CliParam,
    Rules,
    find_missing_vertaxes,
)
from tests import mock_module
from tests.conftest import EXPECTED_GRAPH
from tests.mock_module.a import MockA, MockB, MockD
from tests.mock_module.sub_mock_module.b import (
    MockC,
    MockE,
    MockG,
    MockF,
    BasicNet,
    MockH,
)
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
def test__need_params_for_signature__sanity(
    obj, add_options_from_outside_packages, expected
):
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
        "a": inspect.Parameter(
            "a", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=int
        ),
        "b": inspect.Parameter(
            "b", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str
        ),
        "c": inspect.Parameter(
            "c", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=float
        ),
    }


def test__get_full_signature_parameters__new_function():
    # Act
    result = get_full_signature_parameters(MockC, MockC, "func_name")

    # Assert
    assert result == {
        "self": inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        "args": inspect.Parameter("args", inspect.Parameter.VAR_POSITIONAL),
        "kwargs": inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD),
        "a": inspect.Parameter(
            "a", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=int
        ),
        "b": inspect.Parameter(
            "b", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=MockA
        ),
        "c": inspect.Parameter(
            "c", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=float, default=0.2
        ),
    }


def test__get_full_signature_parameters__stops_at_class_with_no_signature_name():
    # Arrange
    expected = {
        "dd": inspect.Parameter(
            "dd", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=int
        ),
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
        "a__type": "MockB",
        "a": {
            "b__type": "torch.optim.SGD",
            "a": "None",
            "b__connected_params": ["c->module"],
            "b__creator": "create_opt",
        },
        "b__type": str,
        "c__type": BasicNet,
        "b": "bbb",
    }
    regex_config = Rules(
        value_rules={re.compile(r".*\.a$"): 12},
        type_rules={re.compile(r"^f__type$"): MockD},
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
            {
                "a__type": "MockB",
                "a": {"b__type": "torch.optim.SGD", "b": {"lr": 0.001}},
                "b": {"a__const": "0.002"},
            },
            {
                "b__type": "MockH",
                "b": {
                    "opt__connected_params": {"a.b": "a"},
                    "module__const": "torch.optim.SGD",
                },
            },
            Rules(
                type_rules={re.compile(r"b__type$"): "MockA"},
                value_rules={
                    re.compile(r"b\.a\.lr$"): 0.01,
                    re.compile(r"b\.eps$"): "final",
                },
            ),
            Rules(
                connected_params_rules={
                    re.compile(r"a.a__connected_params$"): {"b.opt": "opt"}
                }
            ),
            {
                "a": ParameterNode(
                    type=MockB,
                    value=None,
                    edges={"a.b": "b", "a.b.lr": "lr"},
                    creator=None,
                ),
                "a.b": ParameterNode(
                    type=torch.optim.SGD,
                    value=None,
                    edges={"a.b.lr": "lr"},
                    creator=None,
                ),
                "a.b.lr": ParameterNode(type=None, value=0.001, edges={}, creator=None),
                "b": ParameterNode(
                    type=MockH,
                    value=None,
                    edges={"b.eps": "eps", "b.module": "module", "b.opt": "opt"},
                    creator=None,
                ),
                "b.eps": ParameterNode(
                    type=None, value="final", edges={}, creator=None
                ),
                "b.module": ParameterNode(
                    type=BasicNet,
                    value=torch.optim.SGD,
                    edges={},
                    creator=None,
                ),
                "b.opt": ParameterNode(
                    type=torch.optim.SGD,
                    value=None,
                    edges={"a.b": "b"},
                    creator=None,
                ),
            },
        ],
        [
            {"a__type": "torch.optim.SGD"},
            {"a__type": "MockB"},
            Rules(),
            Rules(),
            {"a": ParameterNode(type=MockB, value=None, edges={})},
        ],
        [
            {"a__const": "torch.optim.SGD"},
            {"a__type": "MockB"},
            Rules(),
            Rules(),
            {"a": ParameterNode(type=MockB, value=SGD, edges={})},
        ],
        [
            {},
            {"a__const": "torch.optim.SGD"},
            Rules(),
            Rules(),
            {"a": ParameterNode(type=int, value=SGD, edges={})},
        ],
        [
            {},
            {"b__type": "float", "b": "10"},
            Rules(),
            Rules(),
            {"b": ParameterNode(type=float, value="10", edges={})},
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
            {"b": 1, "b__connected_params": {"b.c": "c"}},
            Rules(),
            Rules(),
            {
                "b": ParameterNode(type=str, value=1, edges={"b.c": "c"}),
            },
        ],
        [
            {"c__type": "torch.optim.SGD", "c__connected_params": {"b.c": "c"}},
            {},
            Rules(),
            Rules(),
            {
                "c": ParameterNode(type=SGD, value=None, edges={"b.c": "c"}),
            },
        ],
        [
            {"c__type": "torch.optim.SGD", "c__creator": "func"},
            {},
            Rules(),
            Rules(),
            {
                "c": ParameterNode(type=SGD, value=None, edges={}, creator=func),
            },
        ],
        [
            {},
            {
                "c__type": "torch.optim.SGD",
                "c__creator": "tests.mock_module.utils.func",
            },
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
            Rules(type_rules={re.compile(r"c__type$"): MockA}),
            Rules(type_rules={re.compile(r"c__type$"): "torch.optim.SGD"}),
            {"c": ParameterNode(type=SGD, value=None, edges={})},
        ],
        [
            {},
            {},
            Rules(type_rules={re.compile(r"a__type$"): MockA}),
            Rules(value_rules={re.compile(r"c$"): "SGD"}),
            {
                "a": ParameterNode(type=MockA, value=None, edges={}),
                "c": ParameterNode(type=float, value="SGD", edges={}),
            },
        ],
        [
            {},
            {},
            Rules(value_rules={re.compile(r"a__const$"): MockA}),
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
                connected_params_rules={
                    re.compile(r"b__connected_params$"): {"b.c": "c"}
                },
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
                type_rules={re.compile("c__type"): "torch.optim.SGD"},
                connected_params_rules={
                    re.compile("c__connected_params"): {"b.c": "c"}
                },
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
                type_rules={re.compile("c__type"): "torch.optim.SGD"},
                creator_rules={
                    re.compile("c__creator"): "tests.mock_module.utils.func"
                },
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
                value_rules={re.compile("c__const"): "torch.optim.SGD"},
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
    regex_config = Rules(type_rules={re.compile(r"a__type$"): str})
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
    regex_config = Rules(
        value_rules={re.compile(r".*\.a$"): 12, re.compile(r"^b\.a$"): MockA}
    )
    logger = MagicMock()

    # Act
    needed_parameters_for_calling(
        MockC,
        "func_name",
        {},
        {"b__init": True},
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
                CliParam(type=str, multiple=False, default=None, name="a__type"),
                CliParam(
                    type=str, multiple=True, default=None, name="a__connected_params"
                ),
                CliParam(type=str, multiple=False, default=None, name="a__creator"),
                CliParam(type=int, multiple=False, default=None, name="a"),
                CliParam(type=str, multiple=False, default=None, name="a__const"),
                CliParam(type=str, multiple=False, default=None, name="b__type"),
                CliParam(
                    type=str, multiple=True, default=None, name="b__connected_params"
                ),
                CliParam(type=str, multiple=False, default=None, name="b__creator"),
                CliParam(type=str, multiple=False, default=None, name="b"),
                CliParam(type=str, multiple=False, default=None, name="b__const"),
            ],
        ],
        [
            MockC,
            "func_name",
            True,
            [
                CliParam(type=str, multiple=False, default=None, name="e__type"),
                CliParam(
                    type=str, multiple=True, default=None, name="e__connected_params"
                ),
                CliParam(type=str, multiple=False, default=None, name="e__creator"),
                CliParam(type=int, multiple=False, default=None, name="e"),
                CliParam(type=str, multiple=False, default=None, name="e__const"),
                CliParam(type=str, multiple=False, default=None, name="f__type"),
                CliParam(
                    type=str, multiple=True, default=None, name="f__connected_params"
                ),
                CliParam(type=str, multiple=False, default=None, name="f__creator"),
                CliParam(type=str, multiple=False, default=None, name="f"),
                CliParam(type=str, multiple=False, default=None, name="f__const"),
                CliParam(type=str, multiple=False, default=None, name="a__type"),
                CliParam(
                    type=str, multiple=True, default=None, name="a__connected_params"
                ),
                CliParam(type=str, multiple=False, default=None, name="a__creator"),
                CliParam(type=int, multiple=False, default=None, name="a"),
                CliParam(type=str, multiple=False, default=None, name="a__const"),
                CliParam(type=str, multiple=False, default=None, name="b__type"),
                CliParam(
                    type=str, multiple=True, default=None, name="b__connected_params"
                ),
                CliParam(type=str, multiple=False, default=None, name="b__creator"),
                CliParam(
                    type=bool, multiple=False, default=None, name="b__init", flag=True
                ),
                CliParam(type=str, multiple=False, default=None, name="b.a__type"),
                CliParam(
                    type=str, multiple=True, default=None, name="b.a__connected_params"
                ),
                CliParam(type=str, multiple=False, default=None, name="b.a__creator"),
                CliParam(type=int, multiple=False, default=None, name="b.a"),
                CliParam(type=str, multiple=False, default=None, name="b.a__const"),
                CliParam(type=str, multiple=False, default=None, name="b.aa__type"),
                CliParam(
                    type=str, multiple=True, default=None, name="b.aa__connected_params"
                ),
                CliParam(type=str, multiple=False, default=None, name="b.aa__creator"),
                CliParam(type=str, multiple=False, default=None, name="b.aa"),
                CliParam(type=str, multiple=False, default=None, name="b.aa__const"),
                CliParam(type=str, multiple=False, default=None, name="c__type"),
                CliParam(
                    type=str, multiple=True, default=None, name="c__connected_params"
                ),
                CliParam(type=str, multiple=False, default=None, name="c__creator"),
                CliParam(type=float, multiple=False, default=None, name="c"),
                CliParam(type=str, multiple=False, default=None, name="c__const"),
            ],
        ],
        [
            MockG,
            "func_name",
            True,
            [
                CliParam(type=str, multiple=False, default=None, name="opt__type"),
                CliParam(
                    type=str, multiple=True, default=None, name="opt__connected_params"
                ),
                CliParam(type=str, multiple=False, default=None, name="opt__creator"),
                CliParam(
                    type=bool, multiple=False, default=None, name="opt__init", flag=True
                ),
                CliParam(
                    type=str, multiple=False, default=None, name="opt.params__type"
                ),
                CliParam(
                    type=str,
                    multiple=True,
                    default=None,
                    name="opt.params__connected_params",
                ),
                CliParam(
                    type=str, multiple=False, default=None, name="opt.params__creator"
                ),
                CliParam(type=None, multiple=False, default=None, name="opt.params"),
                CliParam(
                    type=str, multiple=False, default=None, name="opt.params__const"
                ),
                CliParam(type=str, multiple=False, default=None, name="opt.lr__type"),
                CliParam(
                    type=str,
                    multiple=True,
                    default=None,
                    name="opt.lr__connected_params",
                ),
                CliParam(
                    type=str, multiple=False, default=None, name="opt.lr__creator"
                ),
                CliParam(type=None, multiple=False, default=None, name="opt.lr"),
                CliParam(type=str, multiple=False, default=None, name="opt.lr__const"),
                CliParam(
                    type=str, multiple=False, default=None, name="opt.momentum__type"
                ),
                CliParam(
                    type=str,
                    multiple=True,
                    default=None,
                    name="opt.momentum__connected_params",
                ),
                CliParam(
                    type=str, multiple=False, default=None, name="opt.momentum__creator"
                ),
                CliParam(type=None, multiple=False, default=None, name="opt.momentum"),
                CliParam(
                    type=str, multiple=False, default=None, name="opt.momentum__const"
                ),
                CliParam(
                    type=str, multiple=False, default=None, name="opt.dampening__type"
                ),
                CliParam(
                    type=str,
                    multiple=True,
                    default=None,
                    name="opt.dampening__connected_params",
                ),
                CliParam(
                    type=str,
                    multiple=False,
                    default=None,
                    name="opt.dampening__creator",
                ),
                CliParam(type=None, multiple=False, default=None, name="opt.dampening"),
                CliParam(
                    type=str, multiple=False, default=None, name="opt.dampening__const"
                ),
                CliParam(
                    type=str,
                    multiple=False,
                    default=None,
                    name="opt.weight_decay__type",
                ),
                CliParam(
                    type=str,
                    multiple=True,
                    default=None,
                    name="opt.weight_decay__connected_params",
                ),
                CliParam(
                    type=str,
                    multiple=False,
                    default=None,
                    name="opt.weight_decay__creator",
                ),
                CliParam(
                    type=None, multiple=False, default=None, name="opt.weight_decay"
                ),
                CliParam(
                    type=str,
                    multiple=False,
                    default=None,
                    name="opt.weight_decay__const",
                ),
                CliParam(
                    type=str, multiple=False, default=None, name="opt.nesterov__type"
                ),
                CliParam(
                    type=str,
                    multiple=True,
                    default=None,
                    name="opt.nesterov__connected_params",
                ),
                CliParam(
                    type=str, multiple=False, default=None, name="opt.nesterov__creator"
                ),
                CliParam(type=None, multiple=False, default=None, name="opt.nesterov"),
                CliParam(
                    type=str, multiple=False, default=None, name="opt.nesterov__const"
                ),
                CliParam(
                    type=str, multiple=False, default=None, name="opt.maximize__type"
                ),
                CliParam(
                    type=str,
                    multiple=True,
                    default=None,
                    name="opt.maximize__connected_params",
                ),
                CliParam(
                    type=str, multiple=False, default=None, name="opt.maximize__creator"
                ),
                CliParam(type=bool, multiple=False, default=None, name="opt.maximize"),
                CliParam(
                    type=str, multiple=False, default=None, name="opt.maximize__const"
                ),
                CliParam(
                    type=str, multiple=False, default=None, name="opt.foreach__type"
                ),
                CliParam(
                    type=str,
                    multiple=True,
                    default=None,
                    name="opt.foreach__connected_params",
                ),
                CliParam(
                    type=str, multiple=False, default=None, name="opt.foreach__creator"
                ),
                CliParam(type=None, multiple=False, default=None, name="opt.foreach"),
                CliParam(
                    type=str, multiple=False, default=None, name="opt.foreach__const"
                ),
                CliParam(
                    type=str,
                    multiple=False,
                    default=None,
                    name="opt.differentiable__type",
                ),
                CliParam(
                    type=str,
                    multiple=True,
                    default=None,
                    name="opt.differentiable__connected_params",
                ),
                CliParam(
                    type=str,
                    multiple=False,
                    default=None,
                    name="opt.differentiable__creator",
                ),
                CliParam(
                    type=bool, multiple=False, default=None, name="opt.differentiable"
                ),
                CliParam(
                    type=str,
                    multiple=False,
                    default=None,
                    name="opt.differentiable__const",
                ),
                CliParam(type=str, multiple=False, default=None, name="eps__type"),
                CliParam(
                    type=str, multiple=True, default=None, name="eps__connected_params"
                ),
                CliParam(type=str, multiple=False, default=None, name="eps__creator"),
                CliParam(type=None, multiple=False, default=None, name="eps"),
                CliParam(type=str, multiple=False, default=None, name="eps__const"),
            ],
        ],
        [
            MockF,
            "func_name",
            True,
            [
                CliParam(type=str, multiple=False, default=None, name="opt__type"),
                CliParam(
                    type=str, multiple=True, default=None, name="opt__connected_params"
                ),
                CliParam(type=str, multiple=False, default=None, name="opt__creator"),
                CliParam(type=None, multiple=False, default=None, name="opt"),
                CliParam(type=str, multiple=False, default=None, name="opt__const"),
            ],
        ],
    ],
)
def test__cli_parameters_for_calling__sanity(
    klass, signature_name, outside_classes, expected
):
    # Act
    results = cli_parameters_for_calling(
        klass, signature_name, outside_classes, mock_module
    )

    # Assert
    assert results == expected


@pytest.mark.parametrize(
    [
        "graph",
        "default_config",
        "config",
        "default_rules",
        "rules",
        "module",
        "add_options_from_outside_packages",
        "new_nodes",
    ],
    [
        [
            {
                "c": ParameterNode(
                    type=MockB, value=None, edges={"a.a": "a", "a.b": "b"}
                )
            },
            {},
            {},
            Rules(),
            Rules(),
            mock_module,
            True,
            {
                "a.a": ParameterNode(type=None, value=None, edges={}),
                "a.b": ParameterNode(type=None, value=None, edges={}),
            },
        ],        [
            {
                "c": ParameterNode(
                    type=MockB, value=None, edges={"a.a": "a", "a.b": "b"}
                )
            },
            {"a.a": 12},
            {"a": {"b__type": float}},
            Rules(),
            Rules(),
            mock_module,
            True,
            {
                "a.a": ParameterNode(type=int, value=12, edges={}),
                "a.b": ParameterNode(type=float, value=None, edges={}),
            },
        ],
    ],
)
def test__find_missing_vertaxes__sanity(
    graph,
    default_config,
    config,
    default_rules,
    rules,
    module,
    add_options_from_outside_packages,
    new_nodes,
):
    # Act
    results = find_missing_vertaxes(
        graph,
        default_config,
        config,
        default_rules,
        rules,
        module,
        add_options_from_outside_packages,
        MagicMock(),
    )

    # Assert
    assert results == new_nodes | graph
