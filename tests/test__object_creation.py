import torch
from torch.optim import SGD

from runner.object_creation import create_objects, ParameterNode
from tests.conftest import EXPECTED_GRAPH
from tests.mock_module.a import MockB, MockA, MockD
from tests.mock_module.sub_mock_module.b import MockH, BasicNet
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
            edges={"runner.opt": "opt", "runner.eps[1]": "eps", "runner.module.linear": "module"},
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
