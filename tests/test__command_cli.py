from runner.command_cli import RunCallableCLI
from click.testing import CliRunner
from tests.mock_module.a import MockB
from tests.mock_module.sub_mock_module.b import MockH
from unittest.mock import MagicMock
from tests import mock_module


def test__RunCLIAlgorithm__sanity():
    # Arrnage
    run = MagicMock(__click_params__=[], params=[])
    runner = CliRunner()
    default_config = "default_config"
    default_assign_value = "default_assign_value"
    default_assign_type = "default_assign_type"
    default_assign_creator = "default_assign_creator"
    default_assign_connection = "default_assign_connection"
    global_settings = "global_settings"
    command_name_a = "a"
    add_options_from_outside_packages = True
    func_name = "func_name"
    params = {
        "a": 1,
        "f": "2",
        "a__type": None,
        "a__connected_params": None,
        "a__creator": None,
        "a__const": None,
        "b__type": None,
        "b__connected_params": None,
        "b__creator": None,
        "b": None,
        "b__const": None,
        "e__type": None,
        "e__connected_params": None,
        "e__creator": None,
        "e": None,
        "e__const": None,
        "f__type": None,
        "f__connected_params": None,
        "f__creator": None,
        "f__const": None,
        "assign_value": {},
        "assign_type": {},
        "assign_creator": {},
        "assign_connection": {},
        "use_config": (),
    }

    cli = RunCallableCLI(
        {command_name_a: (MockB, func_name), "c": (MockH, "func")},
        run,
        add_options_from_outside_packages,
        mock_module,
        default_config,
        default_assign_value,
        default_assign_type,
        default_assign_creator,
        default_assign_connection,
        global_settings,
    )

    # Act
    runner.invoke(cli, [command_name_a, "--a", "1", "--f", 2])

    # Assert
    run.assert_called_once_with(
        class_name=command_name_a,
        func_name=func_name,
        base_module=mock_module,
        add_options_from_outside_packages=add_options_from_outside_packages,
        default_assign_value=default_assign_value,
        default_assign_type=default_assign_type,
        default_assign_creator=default_assign_creator,
        default_assign_connection=default_assign_connection,
        default_config=default_config,
        global_settings=global_settings,
        **params,
    )


def test__run_cli_callable__settings_and_config():
    pass


def test__run_cli_callable__use_config_check():
    pass


def test__cli_initialiation_by_other_classes__sanity():
    pass


def test__cli__nested_parameter_maniuplation():
    pass