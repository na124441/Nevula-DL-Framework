from typing import Any, Optional, Sequence
from nevula.graph.graph import Graph
from nevula.graph.node import Node


class GraphCapture:
    """
    Global tracker for active computational graph recording.
    """
    _active_graph: Optional[Graph] = None

    @classmethod
    def is_active(cls) -> bool:
        return cls._active_graph is not None

    @classmethod
    def get_current_graph(cls) -> Optional[Graph]:
        return cls._active_graph

    @classmethod
    def set_active_graph(cls, graph: Optional[Graph]) -> None:
        cls._active_graph = graph

    @classmethod
    def record_op(cls, op_name: str, inputs: Sequence[Any], output: Any, **kwargs) -> Optional[Node]:
        """
        Records an operation into the currently active graph.
        """
        graph = cls.get_current_graph()
        if graph is None:
            return None

        # Ensure all inputs have nodes in graph
        input_nodes: list[Node] = []
        for inp in inputs:
            node = cls._get_or_create_node(graph, inp)
            input_nodes.append(node)

        # Output shape and device
        out_shape = getattr(output, "shape", ())
        out_device = getattr(output, "device", None)
        out_dtype = "float64"

        op_node = graph.create_node(
            op=op_name,
            inputs=input_nodes,
            shape=out_shape,
            dtype=out_dtype,
            device=out_device,
            kwargs=kwargs,
        )

        # Map output tensor to this operation node
        if hasattr(graph, "_tensor_to_node"):
            graph._tensor_to_node[id(output)] = op_node

        # Update graph outputs: any input consumed is no longer a terminal output
        for in_n in input_nodes:
            if in_n in graph.outputs:
                graph.outputs.remove(in_n)
        if op_node not in graph.outputs:
            graph.outputs.append(op_node)

        return op_node

    @classmethod
    def _get_or_create_node(cls, graph: Graph, val: Any) -> Node:
        from nevula.core.tensor import Tensor

        if not hasattr(graph, "_tensor_to_node"):
            graph._tensor_to_node = {}

        if isinstance(val, Node):
            return val

        if isinstance(val, Tensor):
            if id(val) in graph._tensor_to_node:
                return graph._tensor_to_node[id(val)]

            name = getattr(val, "name", None) or getattr(val, "_graph_name", None)
            if name is None:
                name = f"input_{len(graph.inputs)}"

            node = graph.create_input(
                name=name,
                shape=val.shape,
                dtype="float64",
                device=val.device,
                value=val,
            )
            graph._tensor_to_node[id(val)] = node
            if node not in graph.outputs:
                graph.outputs.append(node)
            return node

        # Literal constant (int, float, etc.)
        const_val = Tensor(val)
        node = graph.create_constant(
            value=const_val,
            name=f"const_{val}",
            shape=const_val.shape,
            device=const_val.device,
        )
        return node


class graph_capture:
    """
    Context manager that traces and captures Tensor operations into a Graph.
    
    Example:
        x = Tensor([2.0])
        w = Tensor([3.0])
        b = Tensor([1.0])

        with graph_capture() as graph:
            y = x * w
            z = y + b
    """

    def __init__(self, graph: Optional[Graph] = None):
        self.graph = graph if graph is not None else Graph()
        self.prev: Optional[Graph] = None

    def __enter__(self) -> Graph:
        self.prev = GraphCapture.get_current_graph()
        GraphCapture.set_active_graph(self.graph)
        return self.graph

    def __exit__(self, exc_type, exc_val, exc_tb):
        GraphCapture.set_active_graph(self.prev)
