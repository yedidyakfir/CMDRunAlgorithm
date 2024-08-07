import tests.mock_module.a
from torch.optim import SGD

from runner.object_creation import ParameterNode
from tests.mock_module.a import MockB, MockD
from tests.mock_module.sub_mock_module.b import BasicNet
from tests.mock_module.utils import create_opt


EXPECTED_GRAPH = {
    "a": ParameterNode(type=MockB, value=None, edges={"a.a": "a", "a.b": "b"}),
    "a.a": ParameterNode(type=None, value=None, edges={}),
    "a.b": ParameterNode(type=SGD, value=None, edges={"c": "module"}, creator=create_opt),
    "b": ParameterNode(type=str, value="bbb", edges={}),
    "c": ParameterNode(type=BasicNet, value=None, edges={}),
    "f": ParameterNode(type=MockD, value=None, edges={}),
}
