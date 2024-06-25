import abc
from typing import List

from torch.optim import SGD

from tests.mock_module.a import MockB, MockA


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
