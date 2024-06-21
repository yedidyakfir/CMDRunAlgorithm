import dataclasses
import importlib.util
import inspect
import logging
from logging import Logger
from typing import Dict, Pattern, Any, Optional

from runner.utils.python import PRIMITIVES
from runner.utils.regex import (
    get_values_from_matching_patterns,
    get_first_value_for_matching_patterns,
)


@dataclasses.dataclass
class ParameterCLI:
    type: type
    default: Any
    requirements: Dict[str, Any]


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
    klass: type,
    base_klass: Optional[type],
    signature_name: str = None,
    add_options_from_outside_packages: bool = True,
) -> Dict[str, inspect.Parameter]:
    parameters = {}
    for parent_class in getattr(klass, "__bases__", []):
        needs_parent_class = (
            issubclass(parent_class, base_klass)
            if base_klass
            else need_params_for_signature(parent_class, add_options_from_outside_packages)
        )
        if needs_parent_class:
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
    initials: str = "",
    logger: Logger = None,
) -> dict:
    logger = logger or logging.getLogger(__name__)
    parameters = {}
    for param, value in get_full_signature_parameters(klass, None, signature_name).items():
        if (
            value.kind == inspect.Parameter.VAR_POSITIONAL
            or value.kind == inspect.Parameter.VAR_KEYWORD
            or param == "self"
        ):
            continue
        param_type_col_name = f"{initials}{param}_type"
        param_type = get_first_value_for_matching_patterns(
            regex_config, param_type_col_name, logger
        )
        default_type_from_secondary_option = (
            key_value_config.get(param_type_col_name)
            if isinstance(key_value_config, dict)
            else None
        )
        default_type_from_secondary_option = (
            default_type_from_secondary_option or value.annotation
        )
        param_type = param_type or default_type_from_secondary_option
        if need_params_for_signature(param_type, add_options_from_outside_packages):
            klass_parameters = needed_parameters_for_creation(
                param_type,
                None,
                key_value_config.get(param),
                regex_config,
                add_options_from_outside_packages,
                f"{initials}{param}.",
                logger,
            )
            final_parameter = ParameterCLI(param_type, None, klass_parameters)
        elif matching_rules_values := get_first_value_for_matching_patterns(
            regex_config, f"{initials}{param}", logger
        ):
            final_parameter = ParameterCLI(param_type, matching_rules_values, {})
        elif key_value_config and param in key_value_config:
            final_parameter = ParameterCLI(param_type, key_value_config.get(param), {})
        elif value.default == inspect.Parameter.empty:
            final_parameter = ParameterCLI(value.annotation, None, {})
        else:
            final_parameter = ParameterCLI(type(value.default), value.default, {})
        if (
            not isinstance(final_parameter.default, final_parameter.type)
            and final_parameter.default is not None
        ):
            logger.warning(
                f"Parameter {initials}{param} has a default value that is not of the same type as the parameter"
            )
        parameters[param] = final_parameter
    return parameters
