from unittest.mock import MagicMock
import tests

import mock
from mock.mock import call

from runner.run import run


@mock.patch("runner.run.needed_parameters_for_calling")
@mock.patch("runner.run.create_objects")
@mock.patch("runner.run.find_class_by_name")
def test__run__sanity(find_class_by_name_mock, create_objects_mock, needed_parameters_for_calling_mock):
    # Arrange
    class_mock_h = "ClassMockH"
    nested_params = {"a.b": 4}
    call_param = {"1": 2, "2": "3"}
    find_class_by_name_mock.return_value = class_mock_h
    graph1 = MagicMock()
    graph2 = MagicMock()
    algorithm = MagicMock()
    needed_parameters_for_calling_mock.side_effect = [graph1, graph2]
    create_objects_mock.side_effect = [algorithm, call_param | nested_params]
    class_name = "MockH"
    func_name = "func"
    base_module = "tests.mock_module"
    default_config = {}
    config = {}
    default_rules = {}
    rules = {}
    add_options_from_outside_packages = True
    global_settings = {}
    use_config = None
    logger = MagicMock()

    # Act
    run(
        class_name,
        func_name,
        base_module,
        default_config,
        config,
        default_rules,
        rules,
        add_options_from_outside_packages,
        global_settings,
        use_config,
        logger=logger,
    )

    # Assert
    needed_parameters_for_calling_mock.assert_has_calls(
        [
            call(
                class_mock_h,
                None,
                default_config,
                config,
                default_rules,
                rules,
                tests,
                add_options_from_outside_packages,
                logger=logger,
            ),
            call(
                class_mock_h,
                func_name,
                default_config,
                config,
                default_rules,
                rules,
                tests,
                add_options_from_outside_packages,
                logger=logger,
            ),
        ]
    )
    create_objects_mock.assert_has_calls([call(graph1), call(graph2)])
    find_class_by_name_mock.assert_has_calls([call(tests, class_name)])
    algorithm.func.assert_called_once_with(**call_param)