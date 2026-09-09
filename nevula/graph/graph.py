from typing import Any, Optional, Sequence, Set
from nevula.backend.device import Device
from nevula.graph.node import Node


class Graph:
    """
    Directed Acyclic Graph (DAG) representing a computational tensor program.
    Contains nodes, tracks input placeholders and target outputs, and provides
    topological sorting and rendering.
    """

    def __init__(self):
        self.nodes: list[Node] = []
        self.inputs: list[Node] = []
        self.outputs: list[Node] = []
        self._id_counter: int = 0

    def next_id(self) -> int:
        nid = self._id_counter
        self._id_counter += 1
        return nid

    def add_node(self, node: Node) -> Node:
        """Registers a node into the graph."""
        if node not in self.nodes:
            self.nodes.append(node)
            if node.id >= self._id_counter:
                self._id_counter = node.id + 1
            if node.is_input and node not in self.inputs:
                self.inputs.append(node)
        return node

    def create_node(
        self,
        op: str,
        inputs: Optional[Sequence[Node]] = None,
        name: Optional[str] = None,
        shape: tuple[int, ...] = (),
        dtype: str = "float64",
        device: Optional[Device] = None,
        value: Optional[Any] = None,
        kwargs: Optional[dict[str, Any]] = None,
    ) -> Node:
        """Factory helper creating and registering a new operation or value node."""
        nid = self.next_id()
        if name is None:
            name = f"{op}_{nid}"
        node = Node(
            node_id=nid,
            name=name,
            op=op,
            inputs=inputs,
            shape=shape,
            dtype=dtype,
            device=device,
            value=value,
            kwargs=kwargs,
        )
        return self.add_node(node)

    def create_input(
        self,
        name: str,
        shape: tuple[int, ...] = (),
        dtype: str = "float64",
        device: Optional[Device] = None,
        value: Optional[Any] = None,
    ) -> Node:
        """Creates an input node representing a model input or parameter."""
        node = self.create_node(
            op="input",
            inputs=[],
            name=name,
            shape=shape,
            dtype=dtype,
            device=device,
            value=value,
        )
        return node

    def create_constant(
        self,
        value: Any,
        name: Optional[str] = None,
        shape: tuple[int, ...] = (),
        dtype: str = "float64",
        device: Optional[Device] = None,
    ) -> Node:
        """Creates a compile-time constant node."""
        nid = self._id_counter
        c_name = name or f"const_{nid}"
        node = self.create_node(
            op="constant",
            inputs=[],
            name=c_name,
            shape=shape,
            dtype=dtype,
            device=device,
            value=value,
        )
        return node

    def topological_sort(self) -> list[Node]:
        """
        Computes dependency-resolved forward evaluation order.
        Guarantees that each node appears only after all of its input dependencies.
        """
        in_degree: dict[int, int] = {node.id: len(node.inputs) for node in self.nodes}
        queue: list[Node] = [node for node in self.nodes if in_degree[node.id] == 0]
        ordered: list[Node] = []

        # Kahn's algorithm
        while queue:
            curr = queue.pop(0)
            ordered.append(curr)
            for consumer in curr.outputs:
                if consumer.id in in_degree:
                    in_degree[consumer.id] -= 1
                    if in_degree[consumer.id] == 0:
                        queue.append(consumer)

        if len(ordered) != len(self.nodes):
            # Fallback to DFS post-order if disconnected or cycles exist
            visited: Set[int] = set()
            ordered = []

            def dfs(n: Node):
                if n.id in visited:
                    return
                visited.add(n.id)
                for in_n in n.inputs:
                    dfs(in_n)
                ordered.append(n)

            if self.outputs:
                for out_n in self.outputs:
                    dfs(out_n)
            for n in self.nodes:
                dfs(n)

        return ordered

    def render(self) -> str:
        """Generates a human-readable ASCII representation of the graph."""
        lines = ["Graph Architecture:"]
        lines.append("=" * 50)
        ordered = self.topological_sort()

        for node in ordered:
            role = "Input" if node.is_input else ("Constant" if node.is_constant else "Operation")
            in_desc = [f"Node {n.id} ({n.name})" for n in node.inputs]
            inputs_str = f" <- [{', '.join(in_desc)}]" if in_desc else ""
            val_str = f" = {node.value}" if (node.is_constant or node.is_input) and node.value is not None else ""
            lines.append(f"Node {node.id}: {role:9s} | {node.name:12s} | op={node.op:8s} | shape={str(node.shape):12s}{inputs_str}{val_str}")

        lines.append("=" * 50)
        out_desc = [f"Node {n.id} ({n.name})" for n in self.outputs]
        lines.append(f"Outputs ({len(self.outputs)}): [{', '.join(out_desc)}]")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Graph(nodes={len(self.nodes)}, inputs={len(self.inputs)}, outputs={len(self.outputs)})"
