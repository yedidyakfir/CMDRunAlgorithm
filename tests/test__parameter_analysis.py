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
        "a_type": "MockB",
        "a": {"b_type": "torch.optim.SGD", "a": "None"},
        "b_type": str,
        "c": 0.1,
        "b": "bbb",
    }
    regex_config = Rules(
        value_rules={re.compile(r".*\.a$"): 12}, type_rules={re.compile(r"^f_type$"): MockA}
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
            {"a_type": "torch.optim.SGD"},
            {"a_type": "MockB"},
            Rules(),
            Rules(),
            {"a": ParameterNode(type=MockB, value=None, edges={})},
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
            {"b": 1, "__b_connected_params": {"b.c": "c"}},
            Rules(),
            Rules(),
            {
                "b": ParameterNode(type=str, value=1, edges={"b.c": "c"}),
            },
        ],
        [
            {"c_type": "torch.optim.SGD", "__c_connected_params": {"b.c": "c"}},
            {},
            Rules(),
            Rules(),
            {
                "c": ParameterNode(type=SGD, value=None, edges={"b.c": "c"}),
            },
        ],
        [
            {"c_type": "torch.optim.SGD", "__c_creator": "func"},
            {},
            Rules(),
            Rules(),
            {
                "c": ParameterNode(type=SGD, value=None, edges={}, creator=func),
            },
        ],
        [
            {},
            {"c_type": "torch.optim.SGD", "__c_creator": "tests.mock_module.utils.func"},
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
            Rules(type_rules={re.compile(r"c_type$"): MockA}),
            Rules(type_rules={re.compile(r"c_type$"): "torch.optim.SGD"}),
            {"c": ParameterNode(type=SGD, value=None, edges={})},
        ],
        [
            {},
            {},
            Rules(type_rules={re.compile(r"a_type$"): MockA}),
            Rules(value_rules={re.compile(r"c$"): "SGD"}),
            {
                "a": ParameterNode(type=MockA, value=None, edges={}),
                "c": ParameterNode(type=float, value="SGD", edges={}),
            },
        ],
        [
            {},
            {},
            Rules(
                value_rules={re.compile(r"b$"): 1},
                connected_params_rules={re.compile(r"__b_connected_params$"): {"b.c": "c"}},
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
                type_rules={re.compile("c_type"): "torch.optim.SGD"},
                connected_params_rules={re.compile("__c_connected_params"): {"b.c": "c"}},
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
                type_rules={re.compile("c_type"): "torch.optim.SGD"},
                creator_rules={re.compile("__c_creator"): "tests.mock_module.utils.func"},
            ),
            {
                "c": ParameterNode(type=SGD, value=None, edges={}, creator=func),
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
    regex_config = Rules(type_rules={re.compile(r"a_type$"): str})
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
        {"__b_init": True},
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
                CliParam(type=str, multiple=False, default=None, name="a_type"),
                CliParam(type=str, multiple=True, default=None, name="__a_connected_params"),
                CliParam(type=str, multiple=False, default=None, name="__a_creator"),
                CliParam(type=int, multiple=False, default=None, name="a"),
                CliParam(type=str, multiple=False, default=None, name="a--const"),
                CliParam(type=str, multiple=False, default=None, name="b_type"),
                CliParam(type=str, multiple=True, default=None, name="__b_connected_params"),
                CliParam(type=str, multiple=False, default=None, name="__b_creator"),
                CliParam(type=str, multiple=False, default=None, name="b"),
                CliParam(type=str, multiple=False, default=None, name="b--const"),
            ],
        ],
        [
            MockC,
            "func_name",
            True,
            [
                CliParam(type=str, multiple=False, default=None, name="e_type"),
                CliParam(type=str, multiple=True, default=None, name="__e_connected_params"),
                CliParam(type=str, multiple=False, default=None, name="__e_creator"),
                CliParam(type=int, multiple=False, default=None, name="e"),
                CliParam(type=str, multiple=False, default=None, name="e--const"),
                CliParam(type=str, multiple=False, default=None, name="f_type"),
                CliParam(type=str, multiple=True, default=None, name="__f_connected_params"),
                CliParam(type=str, multiple=False, default=None, name="__f_creator"),
                CliParam(type=str, multiple=False, default=None, name="f"),
                CliParam(type=str, multiple=False, default=None, name="f--const"),
                CliParam(type=str, multiple=False, default=None, name="a_type"),
                CliParam(type=str, multiple=True, default=None, name="__a_connected_params"),
                CliParam(type=str, multiple=False, default=None, name="__a_creator"),
                CliParam(type=int, multiple=False, default=None, name="a"),
                CliParam(type=str, multiple=False, default=None, name="a--const"),
                CliParam(type=str, multiple=False, default=None, name="b_type"),
                CliParam(type=str, multiple=True, default=None, name="__b_connected_params"),
                CliParam(type=str, multiple=False, default=None, name="__b_creator"),
                CliParam(type=bool, multiple=False, default=None, name="__b_init", flag=True),
                CliParam(type=str, multiple=False, default=None, name="b.a_type"),
                CliParam(type=str, multiple=True, default=None, name="__b.a_connected_params"),
                CliParam(type=str, multiple=False, default=None, name="__b.a_creator"),
                CliParam(type=int, multiple=False, default=None, name="b.a"),
                CliParam(type=str, multiple=False, default=None, name="b.a--const"),
                CliParam(type=str, multiple=False, default=None, name="b.aa_type"),
                CliParam(type=str, multiple=True, default=None, name="__b.aa_connected_params"),
                CliParam(type=str, multiple=False, default=None, name="__b.aa_creator"),
                CliParam(type=str, multiple=False, default=None, name="b.aa"),
                CliParam(type=str, multiple=False, default=None, name="b.aa--const"),
                CliParam(type=str, multiple=False, default=None, name="c_type"),
                CliParam(type=str, multiple=True, default=None, name="__c_connected_params"),
                CliParam(type=str, multiple=False, default=None, name="__c_creator"),
                CliParam(type=float, multiple=False, default=None, name="c"),
                CliParam(type=str, multiple=False, default=None, name="c--const"),
            ],
        ],
        [
            MockG,
            "func_name",
            True,
            [
                CliParam(type=str, multiple=False, default=None, name="opt_type"),
                CliParam(type=str, multiple=True, default=None, name="__opt_connected_params"),
                CliParam(type=str, multiple=False, default=None, name="__opt_creator"),
                CliParam(type=bool, multiple=False, default=None, name="__opt_init", flag=True),
                CliParam(type=str, multiple=False, default=None, name="opt.params_type"),
                CliParam(
                    type=str, multiple=True, default=None, name="__opt.params_connected_params"
                ),
                CliParam(type=str, multiple=False, default=None, name="__opt.params_creator"),
                CliParam(type=None, multiple=False, default=None, name="opt.params"),
                CliParam(type=str, multiple=False, default=None, name="opt.params--const"),
                CliParam(type=str, multiple=False, default=None, name="opt.lr_type"),
                CliParam(
                    type=str, multiple=True, default=None, name="__opt.lr_connected_params"
                ),
                CliParam(type=str, multiple=False, default=None, name="__opt.lr_creator"),
                CliParam(type=None, multiple=False, default=None, name="opt.lr"),
                CliParam(type=str, multiple=False, default=None, name="opt.lr--const"),
                CliParam(type=str, multiple=False, default=None, name="opt.momentum_type"),
                CliParam(
                    type=str,
                    multiple=True,
                    default=None,
                    name="__opt.momentum_connected_params",
                ),
                CliParam(type=str, multiple=False, default=None, name="__opt.momentum_creator"),
                CliParam(type=None, multiple=False, default=None, name="opt.momentum"),
                CliParam(type=str, multiple=False, default=None, name="opt.momentum--const"),
                CliParam(type=str, multiple=False, default=None, name="opt.dampening_type"),
                CliParam(
                    type=str,
                    multiple=True,
                    default=None,
                    name="__opt.dampening_connected_params",
                ),
                CliParam(
                    type=str, multiple=False, default=None, name="__opt.dampening_creator"
                ),
                CliParam(type=None, multiple=False, default=None, name="opt.dampening"),
                CliParam(type=str, multiple=False, default=None, name="opt.dampening--const"),
                CliParam(type=str, multiple=False, default=None, name="opt.weight_decay_type"),
                CliParam(
                    type=str,
                    multiple=True,
                    default=None,
                    name="__opt.weight_decay_connected_params",
                ),
                CliParam(
                    type=str, multiple=False, default=None, name="__opt.weight_decay_creator"
                ),
                CliParam(type=None, multiple=False, default=None, name="opt.weight_decay"),
                CliParam(type=str, multiple=False, default=None, name="opt.weight_decay--const"),
                CliParam(type=str, multiple=False, default=None, name="opt.nesterov_type"),
                CliParam(
                    type=str,
                    multiple=True,
                    default=None,
                    name="__opt.nesterov_connected_params",
                ),
                CliParam(type=str, multiple=False, default=None, name="__opt.nesterov_creator"),
                CliParam(type=None, multiple=False, default=None, name="opt.nesterov"),
                CliParam(type=str, multiple=False, default=None, name="opt.nesterov--const"),
                CliParam(type=str, multiple=False, default=None, name="opt.maximize_type"),
                CliParam(
                    type=str,
                    multiple=True,
                    default=None,
                    name="__opt.maximize_connected_params",
                ),
                CliParam(type=str, multiple=False, default=None, name="__opt.maximize_creator"),
                CliParam(type=bool, multiple=False, default=None, name="opt.maximize"),
                CliParam(type=str, multiple=False, default=None, name="opt.maximize--const"),
                CliParam(type=str, multiple=False, default=None, name="opt.foreach_type"),
                CliParam(
                    type=str, multiple=True, default=None, name="__opt.foreach_connected_params"
                ),
                CliParam(type=str, multiple=False, default=None, name="__opt.foreach_creator"),
                CliParam(type=None, multiple=False, default=None, name="opt.foreach"),
                CliParam(type=str, multiple=False, default=None, name="opt.foreach--const"),
                CliParam(
                    type=str, multiple=False, default=None, name="opt.differentiable_type"
                ),
                CliParam(
                    type=str,
                    multiple=True,
                    default=None,
                    name="__opt.differentiable_connected_params",
                ),
                CliParam(
                    type=str, multiple=False, default=None, name="__opt.differentiable_creator"
                ),
                CliParam(type=bool, multiple=False, default=None, name="opt.differentiable"),
                CliParam(type=str, multiple=False, default=None, name="opt.differentiable--const"),
                CliParam(type=str, multiple=False, default=None, name="eps_type"),
                CliParam(type=str, multiple=True, default=None, name="__eps_connected_params"),
                CliParam(type=str, multiple=False, default=None, name="__eps_creator"),
                CliParam(type=None, multiple=False, default=None, name="eps"),
                CliParam(type=str, multiple=False, default=None, name="eps--const"),
            ],
        ],
        [
            MockF,
            "func_name",
            True,
            [
                CliParam(type=str, multiple=False, default=None, name="opt_type"),
                CliParam(type=str, multiple=True, default=None, name="__opt_connected_params"),
                CliParam(type=str, multiple=False, default=None, name="__opt_creator"),
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
