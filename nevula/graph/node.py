from typing import Any, Optional, Sequence
from nevula.backend.device import Device


class Node:
    """
    Represents an operation or value in a computational graph.
    Maintains edges to input predecessors and output consumer nodes.
    """

    def __init__(
        self,
        node_id: int,
        name: str,
        op: str,
        inputs: Optional[Sequence['Node']] = None,
        shape: tuple[int, ...] = (),
        dtype: str = "float64",
        device: Optional[Device] = None,
        value: Optional[Any] = None,
        kwargs: Optional[dict[str, Any]] = None,
    ):
        self.id: int = node_id
        self.name: str = name
        self.op: str = op.lower()
        self.inputs: list['Node'] = list(inputs or [])
        self.outputs: list['Node'] = []
        self.shape: tuple[int, ...] = tuple(shape)
        self.dtype: str = dtype
        self.device: Device = device if device is not None else Device("cpu")
        self.value: Optional[Any] = value
        self.kwargs: dict[str, Any] = dict(kwargs or {})

        # Wire consumer connections
        for in_node in self.inputs:
            in_node.add_output(self)

    def add_output(self, consumer: 'Node') -> None:
        """Registers a consumer node dependent on this node's output."""
        if consumer not in self.outputs:
            self.outputs.append(consumer)

    def remove_output(self, consumer: 'Node') -> None:
        """Deregisters a consumer node."""
        if consumer in self.outputs:
            self.outputs.remove(consumer)

    @property
    def is_input(self) -> bool:
        """Checks if node is an external graph input or parameter."""
        return self.op == "input"

    @property
    def is_constant(self) -> bool:
        """Checks if node is a compile-time constant."""
        return self.op == "constant"

    @property
    def is_op(self) -> bool:
        """Checks if node represents a computable operator."""
        return self.op not in ("input", "constant")

    def __repr__(self) -> str:
        in_names = [f"Node {n.id} ({n.name})" for n in self.inputs]
        in_str = f", inputs=[{', '.join(in_names)}]" if in_names else ""
        val_str = f", value={self.value}" if self.value is not None else ""
        return f"Node(id={self.id}, name='{self.name}', op='{self.op}', shape={self.shape}{in_str}{val_str})"
