from typing import Any
from nevula.nn.module import Module
from nevula.autograd.functions.loss import CrossEntropy


class Loss(Module):
    """Base class for all loss modules."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Invalid reduction '{reduction}', expected 'mean', 'sum', or 'none'")
        self.reduction = reduction


class MSELoss(Loss):
    """
    Measures the mean squared error (squared L2 norm) between each element
    in the input and target:
        L = (input - target) ** 2
    """

    def forward(self, input: Any, target: Any) -> Any:
        diff = input - target
        loss = diff ** 2

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    def __repr__(self) -> str:
        return f"MSELoss(reduction='{self.reduction}')"


class CrossEntropyLoss(Loss):
    """
    Combines LogSoftmax and NLLLoss in one single differentiable class.
    Accepts raw unnormalized logits of shape (N, C) and targets of shape (N,)
    or one-hot distributions of shape (N, C).
    """

    def forward(self, input: Any, target: Any) -> Any:
        return CrossEntropy.apply(input, target, self.reduction)

    def __repr__(self) -> str:
        return f"CrossEntropyLoss(reduction='{self.reduction}')"
