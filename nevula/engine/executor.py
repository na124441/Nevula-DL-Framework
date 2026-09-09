from typing import Any, Optional, Union
from nevula.backend.device import Device
from nevula.backend.registry import get_backend
from nevula.core.tensor import Tensor
from nevula.graph.graph import Graph
from nevula.graph.node import Node
from nevula.engine.plan import ExecutionPlan, Planner


class Executor:
    """
    Execution engine that takes a computational Graph or ExecutionPlan
    and runs its operations in dependency order on the target Backend.
    """

    def execute(
        self,
        target: Union[Graph, ExecutionPlan],
        feed_dict: Optional[dict[Any, Any]] = None,
    ) -> Any:
        """
        Executes the graph or plan, substituting dynamic inputs from feed_dict.
        
        Args:
            target: A Graph or pre-compiled ExecutionPlan.
            feed_dict: Optional mapping of Nodes, Tensors, or names to runtime Tensors.
            
        Returns:
            The evaluated output Tensor(s).
        """
        if isinstance(target, Graph):
            plan = Planner.build_plan(target)
        elif isinstance(target, ExecutionPlan):
            plan = target
        else:
            raise TypeError(f"Expected Graph or ExecutionPlan, got {type(target).__name__}")

        # Resolve feed_dict keys to node IDs
        resolved_feeds: dict[int, Tensor] = {}
        if feed_dict:
            for k, v in feed_dict.items():
                tensor_v = v if isinstance(v, Tensor) else Tensor(v)
                if isinstance(k, Node):
                    resolved_feeds[k.id] = tensor_v
                elif isinstance(k, Tensor):
                    # Check if tensor was mapped during capture
                    if hasattr(plan.graph, "_tensor_to_node") and id(k) in plan.graph._tensor_to_node:
                        resolved_feeds[plan.graph._tensor_to_node[id(k)].id] = tensor_v
                    else:
                        # Find input with matching shape/name
                        for inp in plan.inputs:
                            if inp.shape == k.shape:
                                resolved_feeds[inp.id] = tensor_v
                                break
                elif isinstance(k, str):
                    for inp in plan.inputs:
                        if inp.name == k:
                            resolved_feeds[inp.id] = tensor_v
                            break

        # Memory workspace for evaluated node outputs
        memory: dict[int, Tensor] = {}

        for step in plan.steps:
            node = step.node

            if node.is_constant:
                val = node.value if isinstance(node.value, Tensor) else Tensor(node.value, device=node.device)
                memory[step.output_id] = val
                continue

            if node.is_input:
                if step.output_id in resolved_feeds:
                    val = resolved_feeds[step.output_id]
                elif node.value is not None:
                    val = node.value if isinstance(node.value, Tensor) else Tensor(node.value, device=node.device)
                else:
                    raise ValueError(f"Missing runtime value for input Node {node.id} ('{node.name}')")
                memory[step.output_id] = val
                continue

            # Retrieve input operands
            args = [memory[in_id] for in_id in step.input_ids]

            # Execute operation on backend
            res = self._execute_op(step.op, args, step.kwargs, step.device)
            memory[step.output_id] = res

        # Collect target outputs
        if not plan.outputs:
            if plan.steps:
                return memory[plan.steps[-1].output_id]
            return None

        if len(plan.outputs) == 1:
            return memory[plan.outputs[0].id]

        return tuple(memory[out_node.id] for out_node in plan.outputs)

    def _execute_op(
        self,
        op: str,
        inputs: list[Tensor],
        kwargs: dict[str, Any],
        device: Device,
    ) -> Tensor:
        backend = get_backend(device.type)

        if op == "add":
            return inputs[0] + inputs[1]
        elif op == "sub":
            return inputs[0] - inputs[1]
        elif op == "mul":
            return inputs[0] * inputs[1]
        elif op == "div":
            return inputs[0] / inputs[1]
        elif op == "pow":
            p = inputs[1].item() if hasattr(inputs[1], "item") else inputs[1]
            return inputs[0] ** p
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
            raise NotImplementedError(f"Operation '{op}' not supported by Executor")
