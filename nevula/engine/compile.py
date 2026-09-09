from typing import Any, Callable, Optional, Union
from nevula.core.tensor import Tensor
from nevula.graph.capture import graph_capture
from nevula.graph.optimizer import GraphOptimizer
from nevula.engine.plan import Planner, ExecutionPlan
from nevula.engine.executor import Executor


class CompiledModel:
    """
    A callable wrapper around a compiled computational graph and execution plan.
    Bypasses dynamic dispatch to execute an optimized execution plan.
    """

    def __init__(self, target: Any, optimizer: Optional[GraphOptimizer] = None):
        self.target: Any = target
        self.optimizer: GraphOptimizer = optimizer if optimizer is not None else GraphOptimizer()
        self.executor: Executor = Executor()
        self.graph = None
        self.plan: Optional[ExecutionPlan] = None
        self._input_nodes = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.plan is None:
            self._compile(*args, **kwargs)

        # Prepare feed_dict for dynamic inputs
        feed_dict = {}
        for i, arg in enumerate(args):
            if i < len(self._input_nodes):
                feed_dict[self._input_nodes[i]] = arg

        return self.executor.execute(self.plan, feed_dict=feed_dict)

    def _compile(self, *sample_args: Any, **kwargs: Any) -> None:
        """Traces the computation graph, runs optimizations, and generates the execution plan."""
        # Annotate parameter names if target is a Module
        from nevula.nn.module import Module
        if isinstance(self.target, Module):
            for name, param in self.target.named_parameters():
                param._graph_name = name

        # Annotate sample inputs
        for i, arg in enumerate(sample_args):
            if isinstance(arg, Tensor) and not hasattr(arg, "_graph_name"):
                arg._graph_name = f"input_{i}"

        # Capture computational graph
        with graph_capture() as captured_graph:
            if callable(self.target):
                _ = self.target(*sample_args, **kwargs)
            else:
                raise TypeError(f"Target {type(self.target)} is not callable.")

        # Remember which nodes correspond to positional inputs (excluding model parameters)
        param_ids = set()
        if isinstance(self.target, Module):
            param_ids = {id(p) for p in self.target.parameters()}

        self._input_nodes = []
        for arg in sample_args:
            if isinstance(arg, Tensor) and id(arg) in captured_graph._tensor_to_node:
                self._input_nodes.append(captured_graph._tensor_to_node[id(arg)])

        # Run compiler optimization passes
        self.graph = self.optimizer.optimize(captured_graph)

        # Build execution plan
        self.plan = Planner.build_plan(self.graph)


def compile(model_or_fn: Any, optimizer: Optional[GraphOptimizer] = None) -> CompiledModel:
    """
    High-level compiler entry point for Nevula programs.
    
    Transforms a neural network Module or Python function into an optimized,
    statically planned executable.
    
    Example:
        model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
        compiled = nevula.compile(model)
        output = compiled(x)
    """
    return CompiledModel(model_or_fn, optimizer=optimizer)
