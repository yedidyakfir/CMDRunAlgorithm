import abc
from typing import List

from torch.nn import Module, Linear
from torch.optim import SGD

from tests.mock_module.a import MockB, MockA


class BasicNet(Module):
    def __init__(self):
        super().__init__()
        self.linear = Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


class MockC(MockB):
    def __init__(self, a: int, b: str, c: float):
        pass

    def func_name(self, a: int, b: MockA, c: float = 0.2, *args, **kwargs):
        pass


class MockE(MockA):
    @abc.abstractmethod
    def start(self):
        pass

    def func_name(self, dd: int):
        pass


class MockG:
    def start(self):
        pass

    def func_name(self, opt: SGD, eps: List[str]):
        pass


class MockF:
    def func_name(self, opt: List[SGD]):
        pass


class MockH:
    def __init__(self, opt: SGD, eps: List[int], module: BasicNet):
        self.opt = opt
        self.eps = eps
        self.module = module

    def __eq__(self, other):
        return self.opt == other.opt and self.eps == other.eps and self.module == other.module

    def func(self, t: int):
        pass

