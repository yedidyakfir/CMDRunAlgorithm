import dataclasses
import re
from typing import Any

import click
from click import Option
from runner.utils.regex import convert_str_keys_to_pattern


@dataclasses.dataclass
class ParamTrueName:
    name: str
    value: Any


def convert_assign_to_pattern(ctx, param, value):
    return convert_str_keys_to_pattern(dict(value))


def multiple_callbacks(ctx, param, value, callbacks):
    for callback in callbacks:
        value = callback(ctx, param, value)
    return value


def ignore_emtpy_multiples(ctx, param, value):
    if param.multiple:
        if isinstance(value, ParamTrueName):
            true_value = value.value
        else:
            true_value = value
        if not true_value:
            return None
    return value


# This enable me to find what is a name and what is a nested value
# (for example opt.lr will should be represented as opt-lr, not opt_lr)
def convert_param_value(ctx, param, value):
    return ParamTrueName(param.opts[-1].split("--")[-1], value)


def convert_click_dict_to_nested(click_values) -> dict:
    nested_dict = {}
    for key, value in click_values.items():
        current_dict = nested_dict
        keys = key.split("-")
        for key in keys[:-1]:
            if key not in current_dict:
                current_dict[key] = {}
            current_dict = current_dict[key]
        current_dict[keys[-1]] = value
    return nested_dict


def create_assigner_option(assign_type: str):
    return Option(
        [f"--assign-{assign_type}"],
        type=click.Tuple([str, str]),
        multiple=True,
        callback=convert_assign_to_pattern,
    )
