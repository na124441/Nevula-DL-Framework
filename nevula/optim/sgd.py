from typing import Any, Iterable
from nevula.optim.optimizer import Optimizer


class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer with optional momentum and L2 weight decay.

    Update rule:
        v_t = momentum * v_{t-1} + g_t
        theta_t = theta_{t-1} - lr * v_t

    Args:
        parameters: Iterable of parameters to optimize.
        lr: Learning rate (default: 0.01).
        momentum: Momentum factor (default: 0.0).
        weight_decay: Weight decay (L2 penalty) factor (default: 0.0).
    """

    def __init__(
        self,
        parameters: Iterable[Any],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        super().__init__(parameters, lr=lr)
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)

    def step(self) -> None:
        """
        Performs a single optimization step.
        Updates parameter values in-place without constructing computation graph nodes.
        """
        for parameter in self.parameters:
            if parameter.grad is None:
                continue

            grad = parameter.grad
            p_id = id(parameter)

            # Initialize momentum state if necessary
            if self.momentum > 0.0 and p_id not in self.state:
                self.state[p_id] = {
                    "velocity": [0.0] * len(parameter.data)
                }

            velocity = self.state[p_id]["velocity"] if self.momentum > 0.0 else None

            # Fast path: contiguous arrays
            if parameter.is_contiguous() and parameter.offset == 0 and grad.is_contiguous() and grad.offset == 0:
                for i in range(len(parameter.data)):
                    g = grad.data[i]

                    if self.weight_decay != 0.0:
                        g += self.weight_decay * parameter.data[i]

                    if self.momentum > 0.0:
                        velocity[i] = self.momentum * velocity[i] + g
                        delta = velocity[i]
                    else:
                        delta = g

                    parameter.data[i] -= self.lr * delta
            else:
                # Coordinate-wise general path
                for idx in parameter._indices_generator():
                    flat_idx = parameter._flat_index(idx)
                    g = grad[idx]

                    if self.weight_decay != 0.0:
                        g += self.weight_decay * parameter[idx]

                    if self.momentum > 0.0:
                        velocity[flat_idx] = self.momentum * velocity[flat_idx] + g
                        delta = velocity[flat_idx]
                    else:
                        delta = g

                    parameter[idx] = parameter[idx] - self.lr * delta
