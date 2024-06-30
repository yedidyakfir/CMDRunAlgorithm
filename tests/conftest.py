from torch.optim import SGD

from runner.object_creation import ParameterNode
from tests.mock_module.a import MockA, MockB

EXPECTED_GRAPH = {
    "a": ParameterNode(
        type=MockB,
        value=None,
        edges={
            "a.a": "a",
            "a.b": "b",
            "a.b.dampening": "dampening",
            "a.b.defaults": "defaults",
            "a.b.differentiable": "differentiable",
            "a.b.foreach": "foreach",
            "a.b.lr": "lr",
            "a.b.maximize": "maximize",
            "a.b.momentum": "momentum",
            "a.b.nesterov": "nesterov",
            "a.b.params": "params",
            "a.b.weight_decay": "weight_decay",
        },
    ),
    "a.a": ParameterNode(type=int, value=None, edges={}),
    "a.b": ParameterNode(
        type=SGD,
        value=None,
        edges={
            "a.b.dampening": "dampening",
            "a.b.defaults": "defaults",
            "a.b.differentiable": "differentiable",
            "a.b.foreach": "foreach",
            "a.b.lr": "lr",
            "a.b.maximize": "maximize",
            "a.b.momentum": "momentum",
            "a.b.nesterov": "nesterov",
            "a.b.params": "params",
            "a.b.weight_decay": "weight_decay",
        },
    ),
    "a.b.dampening": ParameterNode(type=int, value=0, edges={}),
    "a.b.defaults": ParameterNode(type=None, value=None, edges={}),
    "a.b.differentiable": ParameterNode(type=bool, value=False, edges={}),
    "a.b.foreach": ParameterNode(type=None, value=None, edges={}),
    "a.b.lr": ParameterNode(type=float, value=0.001, edges={}),
    "a.b.maximize": ParameterNode(type=bool, value=False, edges={}),
    "a.b.momentum": ParameterNode(type=int, value=0, edges={}),
    "a.b.nesterov": ParameterNode(type=bool, value=False, edges={}),
    "a.b.params": ParameterNode(type=None, value=None, edges={}),
    "a.b.weight_decay": ParameterNode(type=int, value=0, edges={}),
    "b": ParameterNode(type=str, value="bbb", edges={}),
    "c": ParameterNode(type=float, value=0.1, edges={}),
    "e": ParameterNode(type=int, value=None, edges={}),
    "f": ParameterNode(type=MockA, value=None, edges={"f.a": "a", "f.aa": "aa"}),
    "f.a": ParameterNode(type=int, value=12, edges={}),
    "f.aa": ParameterNode(type=str, value=None, edges={}),
}
