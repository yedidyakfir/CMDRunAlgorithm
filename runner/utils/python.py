PRIMITIVES = (
    bool,
    str,
    int,
    float,
    type(None),
    type,
    object,
    list,
    dict,
    tuple,
    set,
    frozenset,
    complex,
    bytes,
    bytearray,
    memoryview,
    range,
    slice,
    property,
    staticmethod,
    classmethod,
    super,
    NotImplemented,
    type,
    object,
    None,
)


def notation_belong_to_typing(annotation):
    return hasattr(annotation, "__module__") and annotation.__module__ == "typing"


def location_in_dict(data: dict, location: str):
    nested_location = location.split(".")
    for i in range(len(nested_location), 0, -1):
        inner_location = ".".join(nested_location[:i])
        if isinstance(data, dict) and inner_location in data:
            data = data[inner_location]
        else:
            return None
    return data
