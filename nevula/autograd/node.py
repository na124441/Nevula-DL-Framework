from typing import Any, Optional, Sequence


class Context:
    """
    Execution context for storing tensors and metadata during the forward pass,
    to be retrieved during the backward pass.
    """
    def __init__(self):
        self._saved_tensors: tuple = ()

    def save_for_backward(self, *tensors: Any) -> None:
        """Saves tensors required for gradient computation in backward."""
        self._saved_tensors = tensors

    @property
    def saved_tensors(self) -> tuple:
        """Retrieves tensors saved during the forward pass."""
        return self._saved_tensors


class Node:
    """
    A node in the computation graph representing an operation.
    Holds references to parent nodes and the backward execution logic.
    """
    def __init__(
        self,
        name: str,
        ctx: Optional[Context] = None,
        next_functions: Optional[Sequence[tuple[Optional['Node'], Any]]] = None,
        backward_fn: Optional[Any] = None
    ):
        self.name = name
        self.ctx = ctx if ctx is not None else Context()
        # List of (grad_fn, input_tensor) pairs
        self.next_functions: list[tuple[Optional['Node'], Any]] = list(next_functions or [])
        self._backward_fn = backward_fn

    def backward(self, *grad_outputs: Any) -> tuple:
        """
        Executes the backward computation for this node.
        Returns a tuple of gradients corresponding to each input in next_functions.
        """
        if self._backward_fn is not None:
            return self._backward_fn(self.ctx, *grad_outputs)
        raise NotImplementedError(f"Node '{self.name}' does not implement backward.")

    def __repr__(self) -> str:
        return f"<{self.name}>"


class AccumulateGrad(Node):
    """
    A leaf node in the computation graph representing gradient accumulation
    into a leaf Tensor.
    """
    def __init__(self, variable: Any):
        super().__init__(name="AccumulateGrad")
        self.variable = variable

    def backward(self, grad_output: Any) -> tuple:
        """Accumulates gradient into the leaf variable."""
        if self.variable.grad is None:
            self.variable.grad = grad_output
        else:
            self.variable.grad = self.variable.grad + grad_output
        return (grad_output,)

    def __repr__(self) -> str:
        return f"<AccumulateGrad for Tensor(shape={getattr(self.variable, 'shape', ())})>"
