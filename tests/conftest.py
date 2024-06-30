from torch.optim import SGD

from runner.object_creation import ParameterNode
from tests.mock_module.a import MockA, MockB

EXPECTED_GRAPH = {
    "a": ParameterNode(type=MockB, value=None, edges={"a.a": "a", "a.b": "b"}),
    "a.a": ParameterNode(type=int, value=None, edges={}),
    "a.b": ParameterNode(type=SGD, value=None, edges={}),
    "b": ParameterNode(type=str, value="bbb", edges={}),
    "c": ParameterNode(type=float, value=0.1, edges={}),
    "f": ParameterNode(type=MockA, value=None, edges={"f.a": "a"}),
    "f.a": ParameterNode(type=int, value=12, edges={}),
}
