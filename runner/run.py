import logging
import os
from logging import Logger
from typing import List, Optional, Dict, Pattern, Any

from runner.dynamic_loading import find_class_by_name
from runner.object_creation import (
    create_objects,
    only_creation_relevant_parameters_from_created,
)
from runner.parameters_analysis import (
    needed_parameters_for_calling,
    find_missing_vertaxes,
)
from runner.parameters_analysis import Rules


def run(
    class_name: str,
    func_name: str,
    base_module: str,
    default_config: dict,
    default_assign_value: Dict[Pattern, Any],
    default_assign_type: Dict[Pattern, Any],
    default_assign_creator: Dict[Pattern, Any],
    default_assign_connection: Dict[Pattern, Any],
    assign_value: Dict[Pattern, Any],
    assign_type: Dict[Pattern, Any],
    assign_creator: Dict[Pattern, Any],
    assign_connection: Dict[Pattern, Any],
    add_options_from_outside_packages: bool,
    global_settings: dict,
    use_config: Optional[List[str]],
    logger: Logger = None,
    **config,
):
    use_logger = logger is not None and isinstance(logger, Logger)
    logger = logger or logging.getLogger(__name__)
    if isinstance(base_module, str):
        module = __import__(base_module)
    else:
        module = base_module

    default_rules = Rules(
        value_rules=default_assign_value,
        type_rules=default_assign_type,
        creator_rules=default_assign_creator,
        connected_params_rules=default_assign_connection,
    )
    rules = Rules(
        value_rules=assign_value,
        type_rules=assign_type,
        creator_rules=assign_creator,
        connected_params_rules=assign_connection,
    )

    algorithm_class = find_class_by_name(module, class_name)

    if use_config:
        for config_name in use_config:
            config = config | global_settings[config_name]

    parameters_graph = needed_parameters_for_calling(
        algorithm_class,
        None,
        default_config,
        config,
        default_rules,
        rules,
        module,
        add_options_from_outside_packages,
        logger=logger,
    )
    parameters_graph = find_missing_vertaxes(
        parameters_graph,
        default_config,
        config,
        default_rules,
        rules,
        module,
        add_options_from_outside_packages,
        logger=logger,
    )
    if "logger" in parameters_graph and use_logger:
        parameters_graph["logger"].value = logger
    all_init_params = create_objects(parameters_graph)
    init_params = only_creation_relevant_parameters_from_created(all_init_params)
    algorithm = algorithm_class(**init_params)

    train_parameters_graph = needed_parameters_for_calling(
        algorithm_class,
        func_name,
        default_config,
        config,
        default_rules,
        rules,
        module,
        add_options_from_outside_packages,
        logger=logger,
    )
    train_parameters_graph = find_missing_vertaxes(
        train_parameters_graph,
        default_config,
        config,
        default_rules,
        rules,
        module,
        add_options_from_outside_packages,
        logger=logger,
    )
    run_parameters = create_objects(train_parameters_graph, init_params)
    func_parameters = only_creation_relevant_parameters_from_created(run_parameters)
    function = getattr(algorithm, func_name)

    logger.info(f"Start running with {algorithm}-{func_name}")
    logger.info(
        f"Train with {os.linesep.join([f'{key}={value}' for key, value in func_parameters.items()])}"
    )
    function(**func_parameters)
