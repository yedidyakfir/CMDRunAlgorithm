import functools
from types import ModuleType
from typing import Callable, List, Optional, Dict, Tuple

import click
from click import MultiCommand, Context, Command, Option

from runner.dynamic_loading import find_subclasses
from runner.parameters_analysis import cli_parameters_for_calling
from runner.run import run
from runner.utils.click import convert_assign_to_pattern


class RunCLIAlgorithm(MultiCommand):
    def __init__(
        self,
        callables: Dict[str, Tuple[type, str]],
        command_runner: Callable,
        add_options_from_outside_packages: bool,
        module: ModuleType,
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
                )
                for param in parameters
            ]
            params += [
                Option(
                    ["--assign"],
                    type=click.Tuple([str, str]),
                    multiple=True,
                    callback=convert_assign_to_pattern,
                ),
                Option(
                    ["--use-config"],
                    type=str,
                    multiple=True,
                ),
            ]
            params += self.addtional_params()
            return Command(cmd_name, params=params, callback=alg_command)

    def addtional_params(self):
        params = []
        params += getattr(self.command_runner, "__click_params__", [])
        params += getattr(self.command_runner, "params", [])
        return params


def run_class(*args, callback, **kwargs):
    callback(*args, runner=run, **kwargs)


class RunnerWithCLI(RunCLIAlgorithm):
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


# TODO - enable run functions from class
# TODO - enable run subclass of a certain function
# TODO - enable run file with parameters, while the user give you some of the parameters in his own way
# TODO - test this file
