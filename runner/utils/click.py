import re


def convert_assign_to_pattern(ctx, param, value):
    return {re.compile(k): v for k, v in value}


def create_assigner_option(assign_type: str):
    return Option(
        [f"--assign-{assign_type}"],
        type=click.Tuple([str, str]),
        multiple=True,
        callback=convert_assign_to_pattern,
    )
