from typing import Any, Optional, Sequence, Set
from nevula.graph.graph import Graph
from nevula.graph.node import Node


class GraphPass:
    """Base class for all graph optimization passes."""

    def run(self, graph: Graph) -> Graph:
        raise NotImplementedError


class ConstantFoldingPass(GraphPass):
    """
    Evaluates subtrees consisting entirely of compile-time constants.
    Replaces static operation nodes with folded constant nodes.
    Example:
        Add(Constant(2), Constant(3)) -> Constant(5)
    """

    def run(self, graph: Graph) -> Graph:
        ordered = graph.topological_sort()
        from nevula.backend.registry import get_backend
        from nevula.core.tensor import Tensor

        for node in ordered:
            if not node.is_op:
                continue

            # Check if all inputs are compile-time constants
            if len(node.inputs) > 0 and all(inp.is_constant and inp.value is not None for inp in node.inputs):
                backend = get_backend(node.device.type)
                inp_vals = [inp.value if isinstance(inp.value, Tensor) else Tensor(inp.value, device=node.device) for inp in node.inputs]

                try:
                    result = self._evaluate_op(node.op, inp_vals, node.kwargs, backend)
                except Exception:
                    # If evaluation fails for any reason, skip folding this node
                    continue

                # Transform node into a constant
                node.op = "constant"
                node.value = result
                node.shape = result.shape
                node.device = result.device

                # Disconnect from parent inputs
                for parent in node.inputs:
                    parent.remove_output(node)
                node.inputs.clear()

        return graph

    def _evaluate_op(self, op: str, inputs: list[Any], kwargs: dict[str, Any], backend: Any) -> Any:
        from nevula.core.tensor import Tensor

        if op == "add":
            return inputs[0] + inputs[1]
        elif op == "sub":
            return inputs[0] - inputs[1]
        elif op == "mul":
            return inputs[0] * inputs[1]
        elif op == "div":
            return inputs[0] / inputs[1]
        elif op == "pow":
            return inputs[0] ** (inputs[1].item() if hasattr(inputs[1], "item") else inputs[1])
        elif op == "neg":
            return -inputs[0]
        elif op == "matmul":
            return inputs[0] @ inputs[1]
        elif op == "relu":
            return inputs[0].relu()
        elif op == "sigmoid":
            return inputs[0].sigmoid()
        elif op == "tanh":
            return inputs[0].tanh()
        elif op == "sum":
            return inputs[0].sum(axis=kwargs.get("axis"), keepdims=kwargs.get("keepdims", False))
        elif op == "mean":
            return inputs[0].mean(axis=kwargs.get("axis"), keepdims=kwargs.get("keepdims", False))
        elif op == "reshape":
            return inputs[0].reshape(kwargs.get("shape", ()))
        elif op == "transpose":
            return inputs[0].transpose(kwargs.get("axis1", -2), kwargs.get("axis2", -1))
        else:
            raise NotImplementedError(f"Op '{op}' not supported in constant folding")


class DeadCodeEliminationPass(GraphPass):
    """
    Removes operations and dead branches that do not contribute to graph outputs.
    """

    def run(self, graph: Graph) -> Graph:
        if not graph.outputs:
            return graph

        # Traverse backwards from output nodes to discover all live dependencies
        live_node_ids: Set[int] = set()
        queue: list[Node] = list(graph.outputs)

        while queue:
            curr = queue.pop(0)
            if curr.id in live_node_ids:
                continue
            live_node_ids.add(curr.id)
            for parent in curr.inputs:
                if parent.id not in live_node_ids:
                    queue.append(parent)

        # Prune dead nodes
        dead_nodes = [n for n in graph.nodes if n.id not in live_node_ids]
        for dead in dead_nodes:
            # Unlink dead node from its inputs
            for parent in dead.inputs:
                parent.remove_output(dead)
            graph.nodes.remove(dead)
            if dead in graph.inputs:
                graph.inputs.remove(dead)

        return graph


class GraphOptimizer:
    """
    Extensible optimization manager chaining multiple graph optimization passes.
    """

    def __init__(self, passes: Optional[Sequence[GraphPass]] = None):
        if passes is None:
            self.passes: list[GraphPass] = [
                ConstantFoldingPass(),
                DeadCodeEliminationPass(),
            ]
        else:
            self.passes = list(passes)

    def add_pass(self, opt_pass: GraphPass) -> 'GraphOptimizer':
        self.passes.append(opt_pass)
        return self

    def optimize(self, graph: Graph) -> Graph:
        """Runs registered optimization passes sequentially on the graph."""
        for opt_pass in self.passes:
            graph = opt_pass.run(graph)
        return graph
