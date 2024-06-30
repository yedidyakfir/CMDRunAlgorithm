import dataclasses
import importlib
import inspect
import logging
import typing
from logging import Logger
from types import ModuleType
from typing import Dict, Pattern, Any, Optional, List

from runner.dynamic_loading import find_class_by_name, find_subclasses
from runner.object_creation import ParameterGraph, ParameterNode
from runner.utils.python import PRIMITIVES, notation_belong_to_typing
from runner.utils.regex import get_first_value_for_matching_patterns


@dataclasses.dataclass
class ParameterHierarchy:
    type: type  # This is needed only for creating the class, We need to understand what to do with typing module
    value: Any
    requirements: Dict[str, "ParameterHierarchy"]
    path_in_tree: str = ""


@dataclasses.dataclass
class CliParam:
    type: type
    multiple: bool
    default: Any
    name: str


ParameterType = Dict[str, ParameterHierarchy]


def create_type_parameter(parameter_name: str):
    return f"{parameter_name}_type"


def create_param_type(module: ModuleType, param_type: Any):
    if isinstance(param_type, str):
        if "." in param_type:
            param_type_path = param_type.split(".")
            class_name, module_name = param_type_path[-1], ".".join(param_type_path[:-1])
            class_type = getattr(importlib.import_module(module_name), class_name)
        else:
            class_type = find_class_by_name(module, param_type)
    else:
        class_type = param_type
    return class_type


def extract_type_from_annotation(annotation):
    if annotation == inspect.Parameter.empty:
        return None
    if notation_belong_to_typing(annotation):
        return None
    return annotation


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
    if signature_name and not hasattr(klass, signature_name):
        return {}
    func = getattr(klass, signature_name) if signature_name else klass
    parameters.update(inspect.signature(func).parameters)
    return parameters


def cli_parameters_for_calling(
    klass: type,
    signature_name: Optional[str],
    add_options_from_outside_packages: bool,
    base_module: ModuleType,
    initials: str = "",
    logger: Logger = None,
) -> List[CliParam]:
    parameters = []
    for param, value in get_full_signature_parameters(klass, None, signature_name).items():
        if (
            value.kind == inspect.Parameter.VAR_POSITIONAL
            or value.kind == inspect.Parameter.VAR_KEYWORD
            or param == "self"
        ):
            continue

        full_param_path = f"{initials}{param}"
        parameters.append(CliParam(str, False, None, create_type_parameter(full_param_path)))
        param_type = value.annotation
        if need_params_for_signature(param_type, add_options_from_outside_packages):
            sub_classes = find_subclasses(base_module, param_type)
            for sub_class in set(sub_classes + [param_type]):
                klass_parameters = cli_parameters_for_calling(
                    sub_class,
                    None,
                    add_options_from_outside_packages,
                    base_module,
                    f"{full_param_path}.",
                    logger,
                )
                parameters += klass_parameters
        elif typing.get_origin(param_type) == list and not need_params_for_signature(
            typing.get_args(param_type)[0], True
        ):
            parameters.append(
                CliParam(typing.get_args(param_type)[0], True, None, full_param_path)
            )
        else:
            param_type = str if notation_belong_to_typing(param_type) else param_type
            parameters.append(CliParam(param_type, False, None, full_param_path))
    return parameters


def needed_parameters_for_calling(
    klass: type,
    signature_name: Optional[str],
    key_value_config: dict,
    regex_config: Dict[Pattern, Any],
    base_module: ModuleType,
    add_options_from_outside_packages: bool,
    initials: str = "",
    logger: Logger = None,
) -> ParameterGraph:
    # TODO - need to split between key value and rules from parameters and from default. To create hierarchy of who is the winner
    logger = logger or logging.getLogger(__name__)
    parameters = {}
    for param, value in get_full_signature_parameters(klass, None, signature_name).items():
        if (
            value.kind == inspect.Parameter.VAR_POSITIONAL
            or value.kind == inspect.Parameter.VAR_KEYWORD
            or param == "self"
        ):
            continue
        full_param_path = f"{initials}{param}"
        param_type_name = create_type_parameter(param)
        param_type_rex_name = f"{initials}{param_type_name}"
        param_type = get_first_value_for_matching_patterns(
            regex_config, param_type_rex_name, logger
        )
        default_type_from_secondary_option = (
            key_value_config.get(param_type_name)
            if isinstance(key_value_config, dict)
            else None
        )
        annotation = extract_type_from_annotation(value.annotation)
        default_type_from_secondary_option = default_type_from_secondary_option or annotation
        param_type = param_type or default_type_from_secondary_option
        param_type = create_param_type(base_module, param_type)
        if key_value_config.get(param) == "None":
            final_parameter = ParameterNode(param_type, None, {})
        elif need_params_for_signature(param_type, add_options_from_outside_packages):
            klass_parameters = needed_parameters_for_calling(
                param_type,
                None,
                key_value_config.get(param, {}),
                regex_config,
                base_module,
                add_options_from_outside_packages,
                f"{full_param_path}.",
                logger,
            )
            parameters.update(klass_parameters)
            final_parameter = ParameterNode(
                param_type,
                None,
                {
                    full_sub_param_name: full_sub_param_name.split(".")[-1]
                    for full_sub_param_name, node in klass_parameters.items()
                },
            )
            logger.info(f"Parameter {full_param_path} is a {param_type}")
        elif key_value_config and param in key_value_config:
            final_parameter = ParameterNode(param_type, key_value_config.get(param), {})
            logger.info(
                f"Parameter {full_param_path} set to {key_value_config.get(param)} from a config"
            )
        elif matching_rules_values := get_first_value_for_matching_patterns(
            regex_config, f"{full_param_path}", logger
        ):
            final_parameter = ParameterNode(param_type, matching_rules_values, {})
            logger.info(
                f"Parameter {full_param_path} set to {matching_rules_values} from a rule"
            )
        elif value.default == inspect.Parameter.empty:
            final_parameter = ParameterNode(annotation, None, {})
            logger.info(
                f"Parameter {full_param_path} has no default value, set as {value.annotation}"
            )
        else:
            final_parameter = ParameterNode(
                type(value.default) if value.default is not None else None, value.default, {}
            )
            logger.info(
                f"Parameter {full_param_path} set to {value.default} from the signature"
            )
        if (
            final_parameter
            and final_parameter.type
            and not isinstance(final_parameter.value, final_parameter.type)
            and final_parameter.value is not None
        ):
            logger.warning(
                f"Parameter {full_param_path} has a default value that is not of the same type as the parameter"
            )
        parameters[full_param_path] = final_parameter
    return parameters
