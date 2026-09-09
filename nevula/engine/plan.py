from typing import Any, Optional, Sequence
from nevula.backend.device import Device
from nevula.graph.graph import Graph
from nevula.graph.node import Node


class ExecutionStep:
    """
    A single instruction in an execution plan.
    """

    def __init__(
        self,
        step_id: int,
        node: Node,
        op: str,
        input_ids: Sequence[int],
        output_id: int,
        device: Device,
        kwargs: Optional[dict[str, Any]] = None,
    ):
        self.step_id: int = step_id
        self.node: Node = node
        self.op: str = op
        self.input_ids: list[int] = list(input_ids)
        self.output_id: int = output_id
        self.device: Device = device
        self.kwargs: dict[str, Any] = dict(kwargs or {})

    def __repr__(self) -> str:
        in_str = f"inputs={self.input_ids}" if self.input_ids else "leaf"
        return f"Step {self.step_id}: {self.op.upper()} ({in_str}) -> Node {self.output_id} [{self.device}]"


class ExecutionPlan:
    """
    Pre-compiled static execution sequence for a computational graph.
    Specifies dependencies, instruction steps, memory footprint, and target device.
    """

    def __init__(
        self,
        graph: Graph,
        steps: Sequence[ExecutionStep],
        inputs: Sequence[Node],
        outputs: Sequence[Node],
        device: Optional[Device] = None,
    ):
        self.graph: Graph = graph
        self.steps: list[ExecutionStep] = list(steps)
        self.inputs: list[Node] = list(inputs)
        self.outputs: list[Node] = list(outputs)
        self.device: Device = device if device is not None else Device("cpu")

    def memory_summary(self) -> dict[str, Any]:
        """Calculates buffer count, node footprints, and total step metrics."""
        total_steps = len(self.steps)
        op_counts: dict[str, int] = {}
        for s in self.steps:
            op_counts[s.op] = op_counts.get(s.op, 0) + 1

        return {
            "total_steps": total_steps,
            "inputs_count": len(self.inputs),
            "outputs_count": len(self.outputs),
            "device": str(self.device),
            "operations": op_counts,
        }

    def render(self) -> str:
        """Renders formatted execution plan."""
        lines = ["Execution Plan:"]
        lines.append("=" * 60)
        for s in self.steps:
            lines.append(f"  {repr(s)}")
        lines.append("=" * 60)
        summary = self.memory_summary()
        lines.append(f"Summary: {summary['total_steps']} steps, {summary['inputs_count']} inputs, {summary['outputs_count']} outputs on {summary['device']}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"ExecutionPlan(steps={len(self.steps)}, inputs={len(self.inputs)}, outputs={len(self.outputs)})"


class Planner:
    """
    Translates a high-level computational Graph into an optimized ExecutionPlan.
    """

    @classmethod
    def build_plan(cls, graph: Graph) -> ExecutionPlan:
        ordered_nodes = graph.topological_sort()
        steps: list[ExecutionStep] = []

        primary_device = Device("cpu")
        if graph.outputs and hasattr(graph.outputs[0], "device"):
            primary_device = graph.outputs[0].device

        for step_id, node in enumerate(ordered_nodes):
            input_ids = [inp.id for inp in node.inputs]
            step = ExecutionStep(
                step_id=step_id,
                node=node,
                op=node.op,
                input_ids=input_ids,
                output_id=node.id,
                device=node.device,
                kwargs=node.kwargs,
            )
            steps.append(step)

        return ExecutionPlan(
            graph=graph,
            steps=steps,
            inputs=graph.inputs,
            outputs=graph.outputs,
            device=primary_device,
        )
