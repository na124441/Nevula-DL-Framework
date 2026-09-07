from typing import Any, Optional, Set
from nevula.autograd.node import Node


def topological_sort(root_node: Node) -> list[Node]:
    """
    Computes a reverse-topological order of graph nodes starting from `root_node`.
    The resulting list starts with `root_node` and ends with the earliest predecessors,
    ensuring that each node's dependencies are fully accumulated before it executes backward.
    """
    visited: Set[Node] = set()
    post_order: list[Node] = []

    def dfs(node: Optional[Node]) -> None:
        if node is None or node in visited:
            return
        visited.add(node)
        for parent_node, _ in node.next_functions:
            if parent_node is not None:
                dfs(parent_node)
        post_order.append(node)

    dfs(root_node)
    return list(reversed(post_order))


def render_graph(root: Any) -> str:
    """
    Produces a human-readable ASCII representation of the computational graph.
    `root` can be a Tensor (with grad_fn) or a Node.
    """
    root_node = getattr(root, "grad_fn", root)
    if root_node is None:
        return "<Leaf Tensor with no grad_fn>"

    lines = []
    visited = set()

    def build_tree(node: Any, prefix: str = "", is_last: bool = True):
        connector = "└── " if is_last else "├── "
        
        if isinstance(node, Node):
            label = node.name
            lines.append(f"{prefix}{connector}{label}")
            child_prefix = prefix + ("    " if is_last else "│   ")
            
            if node in visited:
                lines.append(f"{child_prefix}└── (already expanded)")
                return
            visited.add(node)

            children = node.next_functions
            for i, (child_node, tensor_input) in enumerate(children):
                child_is_last = (i == len(children) - 1)
                if child_node is not None:
                    build_tree(child_node, child_prefix, child_is_last)
                else:
                    # Leaf input
                    t_shape = getattr(tensor_input, "shape", ())
                    req_grad = getattr(tensor_input, "requires_grad", False)
                    lines.append(f"{child_prefix}{'└── ' if child_is_last else '├── '}LeafTensor(shape={t_shape}, requires_grad={req_grad})")
        else:
            lines.append(f"{prefix}{connector}{node}")

    if isinstance(root_node, Node):
        lines.append(root_node.name)
        visited.add(root_node)
        children = root_node.next_functions
        for i, (child_node, tensor_input) in enumerate(children):
            child_is_last = (i == len(children) - 1)
            if child_node is not None:
                build_tree(child_node, "", child_is_last)
            else:
                t_shape = getattr(tensor_input, "shape", ())
                req_grad = getattr(tensor_input, "requires_grad", False)
                lines.append(f"{'└── ' if child_is_last else '├── '}LeafTensor(shape={t_shape}, requires_grad={req_grad})")
    else:
        lines.append(str(root_node))

    return "\n".join(lines)
