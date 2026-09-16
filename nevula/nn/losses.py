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
    Computes the cross entropy loss between input logits and target labels.
    Combines LogSoftmax and NLLLoss in a single, numerically stable class.

    Parameters:
        weight: Manual rescaling weight given to each class (Tensor of shape C).
        ignore_index: Specifies a target value that is ignored and does not contribute to gradient.
        reduction: Specifies the reduction to apply to the output: 'mean', 'sum', 'none'.
        label_smoothing: Float in [0.0, 1.0] specifying amount of smoothing when computing loss.
    """

    def __init__(
        self,
        weight: Any = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__(reduction=reduction)
        self.weight = weight
        self.ignore_index = ignore_index
        self.label_smoothing = float(label_smoothing)

    def forward(self, input: Any, target: Any) -> Any:
        from nevula.autograd.functions.loss import CrossEntropy
        return CrossEntropy.apply(
            input, target, self.weight, self.ignore_index, self.reduction, self.label_smoothing
        )

    def __repr__(self) -> str:
        return f"CrossEntropyLoss(reduction='{self.reduction}', label_smoothing={self.label_smoothing})"


class NLLLoss(Loss):
    """
    The negative log likelihood loss.
    Accepts log-probabilities (e.g. from LogSoftmax) and class targets.
    """

    def __init__(
        self,
        weight: Any = None,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        super().__init__(reduction=reduction)
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input: Any, target: Any) -> Any:
        from nevula.autograd.functions.loss import NLLLoss as NLLLossFn
        return NLLLossFn.apply(
            input, target, self.weight, self.ignore_index, self.reduction
        )

    def __repr__(self) -> str:
        return f"NLLLoss(reduction='{self.reduction}')"


class BCELoss(Loss):
    """
    Measures the Binary Cross Entropy between target and input probabilities:
        L = - (target * log(input) + (1 - target) * log(1 - input))
    """

    def __init__(self, weight: Any = None, reduction: str = "mean"):
        super().__init__(reduction=reduction)
        self.weight = weight

    def forward(self, input: Any, target: Any) -> Any:
        from nevula.autograd.functions.loss import BinaryCrossEntropy
        return BinaryCrossEntropy.apply(input, target, self.weight, self.reduction)

    def __repr__(self) -> str:
        return f"BCELoss(reduction='{self.reduction}')"


class BCEWithLogitsLoss(Loss):
    """
    Combines a Sigmoid layer and the BCELoss in one single class for maximum numerical stability.
    Supports pos_weight for handling imbalanced classification datasets.
    """

    def __init__(self, weight: Any = None, pos_weight: Any = None, reduction: str = "mean"):
        super().__init__(reduction=reduction)
        self.weight = weight
        self.pos_weight = pos_weight

    def forward(self, input: Any, target: Any) -> Any:
        from nevula.autograd.functions.loss import BinaryCrossEntropyWithLogits
        return BinaryCrossEntropyWithLogits.apply(input, target, self.weight, self.pos_weight, self.reduction)

    def __repr__(self) -> str:
        return f"BCEWithLogitsLoss(reduction='{self.reduction}')"


class HingeLoss(Loss):
    """
    Measures the Hinge loss between predicted scores and binary target labels y in {-1, 1}:
        L = max(0, margin - target * input)
    or when squared=True (Squared Hinge / L2-SVM):
        L = max(0, margin - target * input) ** 2

    Parameters:
        margin: The margin threshold (default: 1.0).
        reduction: Specifies reduction: 'mean', 'sum', or 'none' (default: 'mean').
        squared: Whether to square the hinge penalty (default: False).
    """

    def __init__(self, margin: float = 1.0, reduction: str = "mean", squared: bool = False):
        super().__init__(reduction=reduction)
        self.margin = float(margin)
        self.squared = squared

    def forward(self, input: Any, target: Any) -> Any:
        diff = self.margin - target * input
        loss = diff.relu()
        if self.squared:
            loss = loss ** 2

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    def __repr__(self) -> str:
        return f"HingeLoss(margin={self.margin}, reduction='{self.reduction}', squared={self.squared})"

