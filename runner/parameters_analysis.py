import dataclasses
import importlib.util
import inspect
import logging
from logging import Logger
from typing import Dict, Pattern, Any, Optional

from runner.utils.python import PRIMITIVES
from runner.utils.regex import get_values_from_matching_patterns


@dataclasses.dataclass
class ParameterCLI:
    type: type
    default: Any
    requirements: Dict[str, "ParameterCLI"]


def need_params_for_signature(obj: Any, add_options_from_outside_packages: bool) -> bool:
    if not inspect.isclass(obj) or obj in PRIMITIVES:
        return False
    module = inspect.getmodule(obj).__name__
    current_module = inspect.currentframe().f_globals.get("__name__")
    if (
        not add_options_from_outside_packages
        and module.split(".")[0] != current_module.split(".")[0]
    ):
        return False
    return True


def get_full_signature_parameters(
    klass: type, base_klass: type, signature_name: str = None
) -> Dict[str, inspect.Parameter]:
    parameters = {}
    for parent_class in getattr(klass, "__bases__", []):
        if issubclass(parent_class, base_klass):
            parameters.update(
                get_full_signature_parameters(parent_class, base_klass, signature_name)
            )
    func = getattr(klass, signature_name) if signature_name else klass
    parameters.update(inspect.signature(func).parameters)
    return parameters


def needed_parameters_for_creation(
    klass: type,
    signature_name: Optional[str],
    key_value_config: dict,
    regex_config: Dict[Pattern, Any],
    add_options_from_outside_packages: bool,
    logger: Logger = None,
) -> dict:
    logger = logger or logging.getLogger(__name__)
    parameters = {}
    for param, value in get_full_signature_parameters(klass, klass, signature_name).items():
        if (
            value.kind == inspect.Parameter.VAR_POSITIONAL
            or value.kind == inspect.Parameter.VAR_KEYWORD
            or param == "self"
        ):
            continue
        param_type = get_values_from_matching_patterns(regex_config, f"{param}_type")
        if len(param_type) > 1:
            logger.warning(
                f"Multiple types found for {param} in {klass.__name__} signature. Using the first one."
            )
        default_type_from_secondary_option = key_value_config.get(param) or value.annotation
        param_type = param_type[0] if param_type else default_type_from_secondary_option
        if need_params_for_signature(param_type, add_options_from_outside_packages):
            klass_parameters = needed_parameters_for_creation(
                param_type,
                None,
                key_value_config.get(param),
                regex_config,
                add_options_from_outside_packages,
                logger,
            )
            parameters[param] = ParameterCLI(param_type, None, klass_parameters)
        if value.default == inspect.Parameter.empty:
            parameters[param] = ParameterCLI(value.annotation, None, {})
        else:
            parameters[param] = ParameterCLI(type(value.default), value.default, {})
    return parameters
