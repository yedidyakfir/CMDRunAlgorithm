import abc
from abc import ABC


class MockBase(ABC):
    @abc.abstractmethod
    def func_name(self):
        pass


class MockA(MockBase, ABC):
    def __init__(self, a: int, aa: str = "aa"):
        pass


class MockB(MockBase):
    def __init__(self, a: int, b: str):
        pass

    def func_name(self, e: int, f: str):
        pass


class MockD:
    pass
