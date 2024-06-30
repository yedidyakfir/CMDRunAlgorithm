import dataclasses
from collections import deque
from typing import List, Any, Dict, Callable


@dataclasses.dataclass
class ParameterNode:
    type: type  # This is needed only for creating the class, We need to understand what to do with typing module
    value: Any
    edges: Dict[str, str]  # Maps from edge path to parameter name as it is used in the class
    creator: Callable = None


ParameterGraph = Dict[str, ParameterNode]


def topological_sort(graph: ParameterGraph) -> List[str]:
    # in_degree = {key: len(node.edges) for key, node in graph.items()}
    in_degree = {node: 0 for node in graph}
    for node in graph.values():
        for neighbor in node.edges:
            in_degree[neighbor] += 1

    # Initialize queue with nodes that have in-degree 0
    queue = deque([node for node in graph if in_degree[node] == 0])

    topological_order = []
    while queue:
        node_key = queue.popleft()
        node = graph[node_key]
        topological_order.append(node_key)

        for neighbor in node.edges:
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
        dependencies = {neighbor: created_objects[neighbor] for neighbor in node.edges}
        created_objects[node_key] = create_object(node, dependencies)

    return created_objects


def create_object(node: ParameterNode, dependencies: Dict[str, Any]):
    creator = node.creator or node.type
    return node.value or creator(
        **{param_name: dependencies[edge] for edge, param_name in node.edges.items()}
    )
