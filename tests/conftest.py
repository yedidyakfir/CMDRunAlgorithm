from torch.optim import SGD

from runner.object_creation import ParameterNode
from tests.mock_module.a import MockB, MockD
from tests.mock_module.sub_mock_module.b import BasicNet


def create_opt(node, dependencies):
    module = dependencies.pop("module")
    return node.type(module.parameters(), **dependencies)


EXPECTED_GRAPH = {
    "a": ParameterNode(type=MockB, value=None, edges={"a.a": "a", "a.b": "b"}),
    "a.a": ParameterNode(type=None, value=None, edges={}),
    "a.b": ParameterNode(type=SGD, value=None, edges={"c": "module"}, creator=create_opt),
    "b": ParameterNode(type=str, value="bbb", edges={}),
    "c": ParameterNode(type=BasicNet, value=None, edges={}),
    "f": ParameterNode(type=MockD, value=None, edges={}),
    "f.a": ParameterNode(type=int, value=12, edges={}),
}
