from typing import Any, Iterable, Optional
from nevula.core.tensor import Tensor


class Optimizer:
    """
    Base class for all optimizers in Nevula.

    Args:
        parameters: Iterable of Parameters to optimize.
        lr: Learning rate (default: 1e-3).
    """

    def __init__(self, parameters: Iterable[Any], lr: float = 1e-3):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        self.parameters = list(parameters)
        self.lr = float(lr)
        self.state: dict[Any, dict[str, Any]] = {}

    def zero_grad(self) -> None:
        """
        Resets the gradients of all managed parameters.
        """
        for parameter in self.parameters:
            parameter.grad = None

    def step(self) -> None:
        """
        Performs a single optimization step (parameter update).
        Must be implemented by subclasses.
        """
        raise NotImplementedError(f"Optimizer {type(self).__name__} must implement step()")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lr={self.lr}, num_parameters={len(self.parameters)})"
