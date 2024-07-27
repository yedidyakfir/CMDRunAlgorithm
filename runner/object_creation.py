import dataclasses
import functools
from collections import deque
from typing import List, Any, Dict, Callable


@dataclasses.dataclass
class ParameterNode:
    type: type  # This is needed only for creating the class, We need to understand what to do with typing module
    value: Any
    edges: Dict[str, str]  # Maps from edge path to parameter name as it is used in the class
    creator: Callable = None


ParameterGraph = Dict[str, ParameterNode]


def find_closes_edge_in_nested_from_mapping(mapping: Dict[str, Any], edge: str) -> str:
    nested_edge = edge.split(".")
    for i in range(len(nested_edge), 0, -1):
        inner_edge = ".".join(nested_edge[:i])
        if inner_edge in mapping:
            return inner_edge
    raise ValueError(f"Edge {edge} not found in mapping")


def get_value_from_created_objects(created: Dict[str, Any], edge: str) -> Any:
    base_edge = find_closes_edge_in_nested_from_mapping(created, edge)
    base_obj = created[base_edge]
    if base_edge == edge:
        return base_obj
    return eval(f"base_obj.{edge[len(base_edge) + 1:]}")


def topological_sort(graph: ParameterGraph) -> List[str]:
    in_degree = {node: 0 for node in graph}
    for node in graph.values():
        for neighbor in node.edges:
            neighbor = find_closes_edge_in_nested_from_mapping(graph, neighbor)
            in_degree[neighbor] += 1

    # Initialize queue with nodes that have in-degree 0
    queue = deque([node for node in graph if in_degree[node] == 0])

    topological_order = []
    while queue:
        node_key = queue.popleft()
        node_key = find_closes_edge_in_nested_from_mapping(graph, node_key)
        node = graph[node_key]
        topological_order.append(node_key)

        for neighbor in node.edges:
            neighbor = find_closes_edge_in_nested_from_mapping(graph, neighbor)
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(topological_order) != len(graph):
        raise ValueError("Graph has at least one cycle")

    return list(reversed(topological_order))


def create_objects(graph: ParameterGraph):
    order = topological_sort(graph)
    created_objects = {}

    for node_key in order:
        node = graph[node_key]
        dependencies = {
            node.edges[neighbor]: get_value_from_created_objects(created_objects, neighbor)
            for neighbor in node.edges
        }
        creator = node.creator or create_object
        created_objects[node_key] = creator(node, dependencies)

    return created_objects


def create_object(node: ParameterNode, dependencies: Dict[str, Any]):
    if node.value is None and node.type is None:
        return None
    if node.value is not None and isinstance(node.value, str) and node.type != str:
        try:
            return node.type(node.value)
        except ValueError:
            pass
    return node.value or node.type(**dependencies)


def only_creation_relevant_parameters_from_created(created: Dict[str, Any]):
    return {key: value for key, value in created.items() if "." not in key}
