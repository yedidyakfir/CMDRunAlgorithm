import dataclasses
import re
import click
from typing import Any
from click import Option
from runner.utils.regex import convert_str_keys_to_pattern

@dataclasses.dataclass
class ParamTrueName:
    name: str
    value: Any


def convert_assign_to_pattern(ctx, param, value):
    return convert_str_keys_to_pattern(dict(value))


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
