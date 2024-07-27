import importlib.util
import json
import os
import sys

import functools
from types import ModuleType
from pathlib import Path
from typing import Callable, List, Optional, Dict, Tuple, Any

import click
from click import MultiCommand, Context, Command, Option

from runner.dynamic_loading import find_subclasses
from runner.parameters_analysis import cli_parameters_for_calling
from runner.run import run
from runner.utils.click import (
    convert_param_value,
    multiple_callbacks,
    ignore_emtpy_multiples,
    create_assigner_option,
    ParamTrueName,
    convert_click_dict_to_nested,
)
from runner.utils.regex import convert_str_keys_to_pattern

DEFAULT_CONFIG_JSON = "default_config.json"
DEFAULT_RULES_JSON = "default_rules.json"
DEFAULT_SETTINGS_JSON = "default_settings.json"


class RunCallableCLI(MultiCommand):
    def __init__(
        self,
        callables: Dict[str, Tuple[type, str]],
        command_runner: Callable,
        add_options_from_outside_packages: bool,
        module: ModuleType,
        default_config: Dict[str, Any] = None,
        default_assign_value: Dict[str, Any] = None,
        default_assign_type: Dict[str, Any] = None,
        default_assign_creator: Dict[str, Any] = None,
        default_assign_connection: Dict[str, Any] = None,
        global_settings: Dict[str, Any] = None,
        logger=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.callables = callables
        self.command_runner = command_runner
        self.logger = logger
        self.add_options_from_outside_packages = add_options_from_outside_packages
        self.module = module
        self.default_config = default_config or {}
        self.default_assign_value = default_assign_value or {}
        self.default_assign_type = default_assign_type or {}
        self.default_assign_creator = default_assign_creator or {}
        self.default_assign_connection = default_assign_connection or {}
        self.global_settings = global_settings or {}

    def list_commands(self, ctx: Context) -> List[str]:
        return list(self.callables.keys())

    def get_command(self, ctx: Context, cmd_name: str) -> Optional[Command]:
        if cmd_name in self.callables:
            klass, func_name = self.callables[cmd_name]
            alg_command = functools.partial(
                self.command_runner,
                class_name=cmd_name,
                func_name=func_name,
                base_module=self.module,
                add_options_from_outside_packages=self.add_options_from_outside_packages,
                default_assign_value=self.default_assign_value,
                default_assign_type=self.default_assign_type,
                default_assign_creator=self.default_assign_creator,
                default_assign_connection=self.default_assign_connection,
                default_config=self.default_config,
                global_settings=self.global_settings,
            )

            def convert_params_true_values_to_dict(*args, **kwargs):
                normal_command_config = {
                    key: value
                    for key, value in kwargs.items()
                    if not isinstance(value, ParamTrueName)
                }
                param_true_names_config = {
                    value.name: value.value
                    for value in kwargs.values()
                    if isinstance(value, ParamTrueName)
                }
                return alg_command(
                    *args,
                    **convert_click_dict_to_nested(param_true_names_config),
                    **normal_command_config,
                )

            init_params = cli_parameters_for_calling(
                klass,
                None,
                self.add_options_from_outside_packages,
                self.module,
                logger=self.logger,
            )
            func_params = cli_parameters_for_calling(
                klass,
                func_name,
                self.add_options_from_outside_packages,
                self.module,
                logger=self.logger,
            )
            parameters = init_params + func_params

            params = [
                Option(
                    ["--" + "-".join(param.name.split("."))],
                    type=param.type,
                    multiple=param.multiple,
                    default=param.default,
                    is_flag=param.flag,
                    callback=functools.partial(
                        multiple_callbacks,
                        callbacks=[convert_param_value, ignore_emtpy_multiples],
                    ),
                )
                for param in parameters
            ]
            params += [
                create_assigner_option("value"),
                create_assigner_option("type"),
                create_assigner_option("creator"),
                create_assigner_option("connection"),
                Option(
                    ["--use-config"],
                    type=str,
                    multiple=True,
                ),
            ]
            params += self.addtional_params()
            return Command(cmd_name, params=params, callback=convert_params_true_values_to_dict)

    def addtional_params(self):
        return []

    @classmethod
    def from_basic_settings(cls, *args, **kwargs):
        try:
            config = importlib.import_module("config")
            default_config_file_name = getattr(
                config, "DEFAULT_CONFIG_FILE_NAME", DEFAULT_CONFIG_JSON
            )
            default_rules_file_name = getattr(
                config, "DEFAULT_RULES_FILE_NAME", DEFAULT_RULES_JSON
            )
            default_settings_file_name = getattr(
                config, "DEFAULT_SETTINGS_FILE_NAME", DEFAULT_SETTINGS_JSON
            )
        except ModuleNotFoundError:
            default_config_file_name = DEFAULT_CONFIG_JSON
            default_rules_file_name = DEFAULT_RULES_JSON
            default_settings_file_name = DEFAULT_SETTINGS_JSON

        default_config = (
            json.loads(Path(default_config_file_name).read_text())
            if os.path.exists(default_config_file_name)
            else {}
        )
        default_rules = (
            json.loads(Path(default_rules_file_name).read_text())
            if os.path.exists(default_rules_file_name)
            else {}
        )
        global_settings = (
            json.loads(Path(default_settings_file_name).read_text())
            if os.path.exists(default_settings_file_name)
            else {}
        )

        default_rules = {
            rule_name: convert_str_keys_to_pattern(rules)
            for rule_name, rules in default_rules.items()
        }
        default_config = convert_click_dict_to_nested(default_config)
        return cls(
            *args,
            **kwargs,
            default_config=default_config,
            global_settings=global_settings,
            **default_rules,
        )


def run_class(*args, callback, **kwargs):
    callback(*args, runner=run, **kwargs)


class RunnerWithCLI(RunCallableCLI):
    def __init__(self, *args, command_runner, **kwargs):
        self.user_func = command_runner
        callback = functools.partial(run_class, callback=command_runner)
        super().__init__(*args, command_runner=callback, **kwargs)

    def addtional_params(self):
        params = super().addtional_params()
        params += getattr(self.user_func, "__click_params__", [])
        params += getattr(self.user_func, "params", [])
        return params


class RunCLIAlgorithm(RunnerWithCLI):
    def __init__(
        self,
        algorithms: Dict[str, type],
        func_name: str,
        *args,
        **kwargs,
    ):
        commands = {name: (alg, func_name) for name, alg in algorithms.items()}
        super().__init__(*args, callables=commands, **kwargs)


class RunCLIAlgorithmFromModule(RunCLIAlgorithm):
    def __init__(self, module: ModuleType, base_type: type, *args, **kwargs):
        algorithms = {klass.__name__: klass for klass in find_subclasses(module, base_type)}
        super().__init__(algorithms, *args, module=module, **kwargs)


class RunCLIClassFunctions(RunnerWithCLI):
    def __init__(self, klass: type, *args, **kwargs):
        callables = {
            name: (klass, name) for name in dir(klass) if callable(getattr(klass, name))
        }
        super().__init__(*args, callables=callables, **kwargs)
