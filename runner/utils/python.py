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
