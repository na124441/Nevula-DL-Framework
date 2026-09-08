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

    def state_dict(self) -> dict[str, Any]:
        """
        Returns the state of the optimizer as a dict.
        Contains entries for the optimizer state (e.g. step counts, moving averages)
        and hyperparameters like learning rate.
        """
        param_states = {}
        for i, p in enumerate(self.parameters):
            p_id = id(p)
            if p_id in self.state:
                param_states[i] = dict(self.state[p_id])

        return {
            "state": param_states,
            "lr": self.lr,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Loads the optimizer state.
        """
        if "lr" in state_dict:
            self.lr = float(state_dict["lr"])

        saved_states = state_dict.get("state", {})
        for i, p in enumerate(self.parameters):
            if i in saved_states:
                self.state[id(p)] = dict(saved_states[i])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lr={self.lr}, num_parameters={len(self.parameters)})"
