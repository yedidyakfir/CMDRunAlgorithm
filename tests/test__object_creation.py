import torch
from torch.optim import SGD

from runner.object_creation import create_objects, ParameterNode
from tests.conftest import EXPECTED_GRAPH
from tests.mock_module.a import MockB, MockD
from tests.mock_module.sub_mock_module.b import MockH, BasicNet, MockC
from tests.mock_module.utils import create_opt


def test__create_objects__non_hierarchical_graph():
    # Arrange
    eps = [1, 2, 3]
    graph = {
        "runner": ParameterNode(
            type=MockH,
            value=None,
            edges={"runner.opt": "opt", "runner.eps": "eps", "runner.module": "module"},
        ),
        "runner.opt": ParameterNode(
            type=SGD, value=None, edges={"runner.module": "module"}, creator=create_opt
        ),
        "runner.eps": ParameterNode(type=list, value=eps, edges={}),
        "runner.module": ParameterNode(type=BasicNet, value=None, edges={}),
        "runner2": ParameterNode(
            type=MockH,
            value=None,
            edges={
                "runner.opt": "opt",
                "runner.eps[1]": "eps",
                "runner.module.linear": "module",
            },
        ),
    }

    # Act
    result = create_objects(graph)

    # Assert
    assert result["runner.eps"] == eps
    assert isinstance(result["runner.module"], BasicNet)
    assert isinstance(result["runner.opt"], SGD)
    for opt_param, param in zip(
        list(result["runner.module"].parameters()),
        result["runner.opt"].param_groups[0]["params"],
    ):
        if isinstance(opt_param, torch.Tensor):
            assert torch.equal(opt_param, param)
        else:
            assert opt_param == param
    assert result["runner2"].module == result["runner"].module.linear
    assert result["runner2"].eps == eps[1]


def test__can_create_expected_graph():
    # Act
    result = create_objects(EXPECTED_GRAPH)

    # Assert
    assert isinstance(result["a"], MockB)
    assert result["a"].a is None
    assert isinstance(result["b"], str)
    assert isinstance(result["c"], BasicNet)
    assert result["b"] == "bbb"
    assert isinstance(result["a"].b, SGD)
    assert isinstance(result["f"], MockD)


def test__created__from_additional_objects():
    # Act
    graph = {
        "runner": ParameterNode(
            type=MockH,
            value=None,
            edges={"external": "opt", "runner.eps": "eps", "module": "module"},
        ),
        "runner.eps": ParameterNode(
            type=MockC,
            value=None,
            edges={"new.a": "a", "new.b": "b", "new.c": "c"},
        ),
    }
    new = MockC(1, "2", 3.0)
    additional_objects = {"new": new, "external": "SGD", "module": "123"}

    # Act
    result = create_objects(graph, additional_objects)

    # Assert
    assert result["runner"].opt == "SGD"
    assert result["runner"].eps.a == 1
    assert result["runner"].eps.b == "2"
    assert result["runner"].eps.c == 3.0
    assert result["runner"].module == "123"
