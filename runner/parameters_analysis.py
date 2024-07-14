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


def create_param_creator_name(parameter_name: str):
    return f"__{parameter_name}_creator"


def create_param_connection_name(parameter_name: str):
    return f"__{parameter_name}_connected_params"


def create_type_from_name(module: ModuleType, param_type: Any, only_class: bool = True):
    if isinstance(param_type, str):
        if "." in param_type:
            param_type_path = param_type.split(".")
            class_name, module_name = param_type_path[-1], ".".join(param_type_path[:-1])
            class_type = getattr(importlib.import_module(module_name), class_name)
        else:
            class_type = find_class_by_name(module, param_type, only_class)
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
    try:
        inspect.signature(obj)
    except ValueError:
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
    needs_args: bool = True,
    needs_kwargs: bool = True,
) -> Dict[str, inspect.Parameter]:
    if signature_name and not hasattr(klass, signature_name):
        return {}
    func = getattr(klass, signature_name) if signature_name else klass

    parameters = {}
    needs_base_args = any(
        [
            parameter_type
            for parameter_type in inspect.signature(func).parameters.values()
            if parameter_type.kind == inspect.Parameter.VAR_POSITIONAL
        ]
    )
    needs_base_kwargs = any(
        [
            parameter_type
            for parameter_type in inspect.signature(func).parameters.values()
            if parameter_type.kind == inspect.Parameter.VAR_KEYWORD
        ]
    )
    for parent_class in getattr(klass, "__bases__", []):
        needs_parent_class = (
            issubclass(parent_class, base_klass)
            if base_klass
            else need_params_for_signature(parent_class, add_options_from_outside_packages)
        )
        if needs_parent_class:
            parameters.update(
                get_full_signature_parameters(
                    parent_class,
                    base_klass,
                    signature_name,
                    needs_args=needs_base_args,
                    needs_kwargs=needs_base_kwargs,
                )
            )
    if needs_args:
        parameters.update(
            {
                k: v
                for k, v in inspect.signature(func).parameters.items()
                if v.default == inspect.Parameter.empty
            }
        )
    if needs_kwargs:
        parameters.update(
            {
                k: v
                for k, v in inspect.signature(func).parameters.items()
                if v.default != inspect.Parameter.empty
            }
        )
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
        parameters.append(CliParam(str, True, None, create_param_connection_name(full_param_path)))
        parameters.append(CliParam(str, False, None, create_param_creator_name(full_param_path)))
        param_type = extract_type_from_annotation(value.annotation)
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


def extract_value_from_settings(
    param_name: str,
    initials: str,
    regex_config: Dict[Pattern, Any],
    default_regex: Dict[Pattern, Any],
    key_value_config: Dict[str, Any],
    key_value_config_default: Dict[str, Any],
    logger: Logger,
):
    full_param_name = f"{initials}{param_name}"
    value = key_value_config.get(param_name) if isinstance(key_value_config, dict) else None
    value = value or (
        key_value_config_default.get(param_name)
        if isinstance(key_value_config_default, dict)
        else None
    )
    value = value or get_first_value_for_matching_patterns(
        regex_config, full_param_name, logger
    )
    value = value or get_first_value_for_matching_patterns(
        default_regex, full_param_name, logger
    )
    return value


def needed_parameters_for_calling(
    klass: type,
    signature_name: Optional[str],
    key_value_config_default: dict,
    key_value_config: dict,
    regex_config_default: Dict[Pattern, Any],
    regex_config: Dict[Pattern, Any],
    base_module: ModuleType,
    add_options_from_outside_packages: bool,
    initials: str = "",
    logger: Logger = None,
) -> ParameterGraph:
    # TODO - need to think about a nested way to create parameter graph type, how to analyze the type
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

        param_type = extract_value_from_settings(
            param_type_name,
            initials,
            regex_config,
            regex_config_default,
            key_value_config,
            key_value_config_default,
            logger,
        )
        annotation = extract_type_from_annotation(value.annotation)
        param_type = param_type or annotation
        param_type = create_type_from_name(base_module, param_type)

        creator_name = create_param_creator_name(full_param_path)
        creator = extract_value_from_settings(
            creator_name,
            initials,
            regex_config,
            regex_config_default,
            key_value_config,
            key_value_config_default,
            logger,
        )
        creator = create_type_from_name(base_module, creator, False) if creator else None
        connected_params_name = create_param_connection_name(full_param_path)
        connected_params = extract_value_from_settings(
            connected_params_name,
            initials,
            regex_config,
            regex_config_default,
            key_value_config,
            key_value_config_default,
            logger,
        )
        if connected_params is None:
            connected_params = {}

        # Create the node for the parameter
        if key_value_config.get(param) == "None":
            final_parameter = ParameterNode(param_type, None, connected_params, creator)
        elif (
            key_value_config_default.get(param) == "None"
            and key_value_config.get(param) is None
        ):
            final_parameter = ParameterNode(param_type, None, connected_params, creator)
        elif need_params_for_signature(param_type, add_options_from_outside_packages):
            klass_parameters = needed_parameters_for_calling(
                param_type,
                None,
                key_value_config_default.get(param, {}),
                key_value_config.get(param, {}),
                regex_config_default,
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
                }
                | connected_params,
                creator,
            )
            logger.info(f"Parameter {full_param_path} is a {param_type}")
        elif key_value_config and param in key_value_config:
            final_parameter = ParameterNode(
                param_type, key_value_config.get(param), connected_params, creator
            )
            logger.info(
                f"Parameter {full_param_path} set to {key_value_config.get(param)} from a config"
            )
        elif matching_rules_values := (
            get_first_value_for_matching_patterns(regex_config, full_param_path, logger)
        ):
            final_parameter = ParameterNode(
                param_type, matching_rules_values, connected_params, creator
            )
            logger.info(
                f"Parameter {full_param_path} set to {matching_rules_values} from a rule"
            )
        elif key_value_config_default and param in key_value_config_default:
            final_parameter = ParameterNode(
                param_type, key_value_config_default.get(param), connected_params, creator
            )
            logger.info(
                f"Parameter {full_param_path} set to {key_value_config_default.get(param)} from a default config"
            )
        elif matching_rules_values := get_first_value_for_matching_patterns(
            regex_config_default, full_param_path, logger
        ):
            final_parameter = ParameterNode(
                param_type, matching_rules_values, connected_params, creator
            )
            logger.info(
                f"Parameter {full_param_path} set to {matching_rules_values} from a default rule"
            )
        else:
            final_parameter = None
        if (
            final_parameter
            and final_parameter.type
            and not isinstance(final_parameter.value, final_parameter.type)
            and final_parameter.value is not None
        ):
            logger.warning(
                f"Parameter {full_param_path} has a default value that is not of the same type as the parameter"
            )
        if final_parameter:
            parameters[full_param_path] = final_parameter
    return parameters
