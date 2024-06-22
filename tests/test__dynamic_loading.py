from runner.dynamic_loading import find_subclasses, find_class_by_name
from tests import mock_module
from tests.mock_module.a import MockBase
from tests.mock_module.sub_mock_module.b import MockC


def test__find_subclasses__sanity():
    # Act
    result = find_subclasses(mock_module, MockBase)
    expected = [mock_module.a.MockB, MockC]

    # Assert
    assert set(result) == set(expected)


def test__find_class_by_name__sanity():
    # Act
    result = find_class_by_name(mock_module, "MockC")

    # Assert
    assert result == MockC
