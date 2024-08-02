import dataclasses
import importlib
import inspect
import logging
import typing
from collections import defaultdict
from dataclasses import field
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
    flag: bool = False


ParameterType = Dict[str, ParameterHierarchy]
RulesType = Dict[Pattern, Any]


@dataclasses.dataclass
class Rules:
    value_rules: RulesType = field(default_factory=lambda: defaultdict(dict))
    type_rules: RulesType = field(default_factory=lambda: defaultdict(dict))
    creator_rules: RulesType = field(default_factory=lambda: defaultdict(dict))
    connected_params_rules: RulesType = field(default_factory=lambda: defaultdict(dict))


def create_type_parameter(parameter_name: str):
    return f"{parameter_name}__type"


def create_param_creator_name(parameter_name: str):
    return f"{parameter_name}__creator"


def create_param_connection_name(parameter_name: str):
    return f"{parameter_name}__connected_params"


def create_param_initialize_command_name(parameter_name: str):
    return f"{parameter_name}__init"


def create_const_param_name(parameter_name: str):
    return f"{parameter_name}__const"


def create_edges_mapping_from_connection_params(connections: List[str]):
    return {
        full_sub_param_name.split("->")[0]: full_sub_param_name.split("->")[1]
        if "->" in full_sub_param_name
        else full_sub_param_name.split(".")[-1]
        for full_sub_param_name in connections
    }


def create_type_from_name(module: ModuleType, param_type: Any, only_class: bool = True):
    if isinstance(param_type, str):
        if "." in param_type:
            param_type_path = param_type.split(".")
            class_name, module_name = param_type_path[-1], ".".join(
                param_type_path[:-1]
            )
            try:
                class_type = getattr(importlib.import_module(module_name), class_name)
            except ModuleNotFoundError:
                class_type = None
        else:
            class_type = find_class_by_name(module, param_type, only_class)
        if class_type is None:
            class_type = eval(param_type)
    else:
        class_type = param_type
    return class_type


def extract_type_from_annotation(annotation):
    if annotation == inspect.Parameter.empty:
        return None
    if notation_belong_to_typing(annotation):
        return None
    return annotation


def need_params_for_signature(
    obj: Any, add_options_from_outside_packages: bool
) -> bool:
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
            else need_params_for_signature(
                parent_class, add_options_from_outside_packages
            )
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
    for param, value in get_full_signature_parameters(
        klass, None, signature_name
    ).items():
        if (
            value.kind == inspect.Parameter.VAR_POSITIONAL
            or value.kind == inspect.Parameter.VAR_KEYWORD
            or param == "self"
        ):
            continue

        full_param_path = f"{initials}{param}"
        parameters += [
            CliParam(str, False, None, create_type_parameter(full_param_path)),
            CliParam(str, True, None, create_param_connection_name(full_param_path)),
            CliParam(str, False, None, create_param_creator_name(full_param_path)),
        ]
        param_type = extract_type_from_annotation(value.annotation)
        if need_params_for_signature(param_type, add_options_from_outside_packages):
            parameters.append(
                CliParam(
                    bool,
                    False,
                    None,
                    create_param_initialize_command_name(full_param_path),
                    True,
                )
            )
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
            parameters += [
                CliParam(typing.get_args(param_type)[0], True, None, full_param_path),
                CliParam(str, True, None, create_const_param_name(full_param_path)),
            ]
        else:
            param_type = str if notation_belong_to_typing(param_type) else param_type
            parameters += [
                CliParam(param_type, False, None, full_param_path),
                CliParam(str, False, None, create_const_param_name(full_param_path)),
            ]
    return parameters


def extract_value_from_settings(
    param_name: str,
    initials: str,
    regex_config: RulesType,
    default_regex: RulesType,
    key_value_config: Dict[str, Any],
    key_value_config_default: Dict[str, Any],
    logger: Logger,
):
    full_param_name = f"{initials}{param_name}"
    value = (
        key_value_config.get(param_name) if isinstance(key_value_config, dict) else None
    )
    if value is not None:
        logger.info(f"Parameter {full_param_name} has a value of {value} from config")
        return value

    value = (
        key_value_config_default.get(param_name)
        if isinstance(key_value_config_default, dict)
        else None
    )
    if value is not None:
        logger.info(
            f"Parameter {full_param_name} has a value of {value} from default config"
        )
        return value

    value = get_first_value_for_matching_patterns(regex_config, full_param_name, logger)
    if value is not None:
        logger.info(f"Parameter {full_param_name} has a value of {value} from regex")
        return value

    value = get_first_value_for_matching_patterns(
        default_regex, full_param_name, logger
    )
    if value is not None:
        logger.info(
            f"Parameter {full_param_name} has a value of {value} from default regex"
        )
        return value
    return None


def extract_values_for_param(
    param: str,
    value: inspect.Parameter,
    key_value_config_default: dict,
    key_value_config: dict,
    regex_config_default: Rules,
    regex_config: Rules,
    base_module: ModuleType,
    add_options_from_outside_packages: bool,
    initials: str = "",
    logger: Logger = None,
):
    param_type_name = create_type_parameter(param)

    config_param_type = extract_value_from_settings(
        param_type_name,
        initials,
        regex_config.type_rules,
        regex_config_default.type_rules,
        key_value_config,
        key_value_config_default,
        logger,
    )
    annotation = extract_type_from_annotation(value.annotation)
    param_type = config_param_type or annotation
    param_type = create_type_from_name(base_module, param_type)

    creator_name = create_param_creator_name(param)
    creator = extract_value_from_settings(
        creator_name,
        initials,
        regex_config.creator_rules,
        regex_config_default.creator_rules,
        key_value_config,
        key_value_config_default,
        logger,
    )
    creator = create_type_from_name(base_module, creator, False) if creator else None
    connected_params_name = create_param_connection_name(param)
    connected_params = extract_value_from_settings(
        connected_params_name,
        initials,
        regex_config.connected_params_rules,
        regex_config_default.connected_params_rules,
        key_value_config,
        key_value_config_default,
        logger,
    )
    if connected_params is None:
        connected_params = {}
    else:
        connected_params = create_edges_mapping_from_connection_params(connected_params)

    param_value = extract_value_from_settings(
        param,
        initials,
        regex_config.value_rules,
        regex_config_default.value_rules,
        key_value_config,
        key_value_config_default,
        logger,
    )
    if param_value is None:
        param_const_value = extract_value_from_settings(
            create_const_param_name(param),
            initials,
            regex_config.value_rules,
            regex_config_default.value_rules,
            key_value_config,
            key_value_config_default,
            logger,
        )
        if param_const_value:
            param_value = create_type_from_name(base_module, param_const_value, False)

    # Create the node for the parameter
    init_value_name = create_param_initialize_command_name(param)
    init_value = extract_value_from_settings(
        init_value_name,
        initials,
        regex_config.value_rules,
        regex_config_default.value_rules,
        key_value_config,
        key_value_config_default,
        logger,
    )
    param_mentioned_by_user = (
        param_value or config_param_type or creator or connected_params or init_value
    )
    return (
        f"{initials}{param}",
        param_type,
        param_value,
        connected_params,
        creator,
        init_value,
        param_mentioned_by_user,
    )


def needed_parameters_for_calling(
    klass: type,
    signature_name: Optional[str],
    key_value_config_default: dict,
    key_value_config: dict,
    regex_config_default: Rules,
    regex_config: Rules,
    base_module: ModuleType,
    add_options_from_outside_packages: bool,
    initials: str = "",
    logger: Logger = None,
) -> ParameterGraph:
    logger = logger or logging.getLogger(__name__)
    parameters = {}
    for param, value in get_full_signature_parameters(
        klass, None, signature_name
    ).items():
        if (
            value.kind == inspect.Parameter.VAR_POSITIONAL
            or value.kind == inspect.Parameter.VAR_KEYWORD
            or param == "self"
        ):
            continue
        (
            full_param_path,
            param_type,
            param_value,
            connected_params,
            creator,
            init_value,
            param_mentioned_by_user,
        ) = extract_values_for_param(
            param,
            value,
            key_value_config_default,
            key_value_config,
            regex_config_default,
            regex_config,
            base_module,
            add_options_from_outside_packages,
            initials,
            logger,
        )

        if param_value == "None":
            final_parameter = ParameterNode(None, None, connected_params, creator)
        elif param_value is not None and not isinstance(param_value, dict):
            final_parameter = ParameterNode(
                param_type, param_value, connected_params, creator
            )
            logger.info(f"Parameter {full_param_path} has a value of {param_value}")
        elif (
            need_params_for_signature(param_type, add_options_from_outside_packages)
            and param_mentioned_by_user
        ):
            klass_parameters = needed_parameters_for_calling(
                param_type,
                None,
                key_value_config_default.get(param) or {},
                key_value_config.get(param) or {},
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
                create_edges_mapping_from_connection_params(klass_parameters.keys())
                | connected_params,
                creator,
            )
            logger.info(f"Parameter {full_param_path} is a {param_type}")
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
