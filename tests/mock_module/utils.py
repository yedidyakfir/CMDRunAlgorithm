def func():
    pass

def create_opt(node, dependencies):
    module = dependencies.pop("module")
    return node.type(module.parameters(), **dependencies)
