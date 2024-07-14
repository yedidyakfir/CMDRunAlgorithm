import re


def convert_assign_to_pattern(ctx, param, value):
    return {re.compile(k): v for k, v in value}
