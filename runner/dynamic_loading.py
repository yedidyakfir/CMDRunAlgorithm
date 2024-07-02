import importlib
import inspect
import pkgutil
from types import ModuleType
from typing import List, Optional


def find_subclasses(module: ModuleType, base_class: type) -> List[type]:
    subclasses = []

    # Iterate over all modules in the package
    for loader, name, is_pkg in pkgutil.walk_packages(module.__path__):
        # Import the module
        full_name = module.__name__ + "." + name
        sub_module = importlib.import_module(full_name)
        if is_pkg:
            subclasses.extend(find_subclasses(sub_module, base_class))
        else:
            # Iterate over all objects in the module
            for obj_name, obj in inspect.getmembers(sub_module):
                # Check if the object is a class, is a subclass of the base class,
                # and is not the base class itself
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, base_class)
                    and not inspect.isabstract(obj)
                ):
                    subclasses.append(obj)
    return list(set(subclasses))


def find_class_by_name(module: ModuleType, class_name: str, only_class: bool = True) -> Optional[type]:
    for loader, name, is_pkg in pkgutil.walk_packages(module.__path__):
        full_name = module.__name__ + "." + name
        sub_module = importlib.import_module(full_name)
        if is_pkg:
            if klass := find_class_by_name(sub_module, class_name, only_class):
                return klass
        else:
            for obj_name, obj in inspect.getmembers(sub_module):
                if (
                    (inspect.isclass(obj) or not only_class)
                    and not inspect.isabstract(obj)
                    and class_name == obj_name
                ):
                    return obj
    return None
