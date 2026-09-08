import math
from typing import Any, Iterable
from nevula.optim.optimizer import Optimizer


class Adam(Optimizer):
    """
    Adaptive Moment Estimation (Adam) optimizer.

    Maintains moving averages of the gradients (first moment) and squared gradients
    (second moment), with bias correction for initial steps.

    Args:
        parameters: Iterable of parameters to optimize.
        lr: Learning rate (default: 1e-3).
        betas: Coefficients for computing running averages of gradient and its square (default: (0.9, 0.999)).
        eps: Term added to denominator to improve numerical stability (default: 1e-8).
        weight_decay: Weight decay (L2 penalty) factor (default: 0.0).
    """

    def __init__(
        self,
        parameters: Iterable[Any],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(parameters, lr=lr)
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        self.beta1, self.beta2 = betas
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)

    def step(self) -> None:
        """
        Performs a single optimization step.
        """
        for parameter in self.parameters:
            if parameter.grad is None:
                continue

            grad = parameter.grad
            p_id = id(parameter)

            if p_id not in self.state:
                self.state[p_id] = {
                    "step": 0,
                    "exp_avg": [0.0] * len(parameter.data),
                    "exp_avg_sq": [0.0] * len(parameter.data),
                }

            s = self.state[p_id]
            s["step"] += 1
            t = s["step"]
            exp_avg = s["exp_avg"]
            exp_avg_sq = s["exp_avg_sq"]

            bias_correction1 = 1.0 - (self.beta1 ** t)
            bias_correction2 = 1.0 - (self.beta2 ** t)

            # Contiguous fast path
            if parameter.is_contiguous() and parameter.offset == 0 and grad.is_contiguous() and grad.offset == 0:
                for i in range(len(parameter.data)):
                    g = grad.data[i]

                    if self.weight_decay != 0.0:
                        g += self.weight_decay * parameter.data[i]

                    # Moving average of gradient values
                    exp_avg[i] = self.beta1 * exp_avg[i] + (1.0 - self.beta1) * g
                    # Moving average of squared gradient values
                    exp_avg_sq[i] = self.beta2 * exp_avg_sq[i] + (1.0 - self.beta2) * (g * g)

                    denom = math.sqrt(exp_avg_sq[i] / bias_correction2) + self.eps
                    step_size = (exp_avg[i] / bias_correction1) / denom

                    parameter.data[i] -= self.lr * step_size
            else:
                # Coordinate-wise general path
                for idx in parameter._indices_generator():
                    flat_idx = parameter._flat_index(idx)
                    g = grad[idx]

                    if self.weight_decay != 0.0:
                        g += self.weight_decay * parameter[idx]

                    exp_avg[flat_idx] = self.beta1 * exp_avg[flat_idx] + (1.0 - self.beta1) * g
                    exp_avg_sq[flat_idx] = self.beta2 * exp_avg_sq[flat_idx] + (1.0 - self.beta2) * (g * g)

                    denom = math.sqrt(exp_avg_sq[flat_idx] / bias_correction2) + self.eps
                    step_size = (exp_avg[flat_idx] / bias_correction1) / denom

                    parameter[idx] = parameter[idx] - self.lr * step_size
