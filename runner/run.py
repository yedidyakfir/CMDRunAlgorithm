import logging
import os
from logging import Logger
from typing import List, Optional, Dict, Pattern, Any

from runner.dynamic_loading import find_class_by_name
from runner.object_creation import (
    create_objects,
    only_creation_relevant_parameters_from_created,
)
from runner.parameters_analysis import needed_parameters_for_calling
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
    logger = logger or logging.getLogger(__name__)
    # TODO - how to get logger from user?
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
    # TODO - how to set consts? like space or env
    # TODO - how to set parameters from other parameters created? like lower bound and dims from space

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
    algorithm = create_objects(parameters_graph)
    # TODO - how to manipulate the class, like setting the start point?

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
    run_parameters = create_objects(train_parameters_graph)
    func_parameters = only_creation_relevant_parameters_from_created(run_parameters)
    function = getattr(algorithm, func_name)

    logger.info(f"Start running with {algorithm}-{func_name}")
    logger.info(
        f"Train with {os.linesep.join([f'{key}={value}' for key, value in func_parameters.items()])}"
    )
    function(**func_parameters)
    # TODO - how to add additional const parameters? like handlers
    # TODO - somtimes the creation of const is based on the parameters (like on trust region)
