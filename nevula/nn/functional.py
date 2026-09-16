"""
Functional interface for activations, losses, and tensor operations.
"""

from typing import Any, Optional, Union
import math
import numpy as np

from nevula.core.tensor import Tensor
from nevula.autograd.functions.loss import (
    CrossEntropy,
    BinaryCrossEntropy,
    BinaryCrossEntropyWithLogits,
)


def _to_tensor(x: Any) -> Tensor:
    if isinstance(x, Tensor):
        return x
    return Tensor(x)


def sigmoid(input: Any) -> Tensor:
    """
    Applies the element-wise sigmoid function:
        sigmoid(x) = 1 / (1 + exp(-x))
    """
    return _to_tensor(input).sigmoid()


def log_sigmoid(input: Any) -> Tensor:
    """
    Applies the element-wise log-sigmoid function:
        log_sigmoid(x) = log(1 / (1 + exp(-x)))
    """
    return _to_tensor(input).log_sigmoid()


def relu(input: Any) -> Tensor:
    """
    Applies the rectified linear unit function element-wise:
        relu(x) = max(0, x)
    """
    return _to_tensor(input).relu()


def tanh(input: Any) -> Tensor:
    """
    Applies the hyperbolic tangent function element-wise:
        tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    """
    return _to_tensor(input).tanh()


def softmax(input: Any, dim: int = -1) -> Tensor:
    """
    Applies the Softmax function to an n-dimensional input Tensor,
    rescaling them so that the elements of the n-dimensional output Tensor
    lie in the range [0, 1] and sum to 1.
    """
    return _to_tensor(input).softmax(dim=dim)


def log_softmax(input: Any, dim: int = -1) -> Tensor:
    """
    Applies the LogSoftmax function to an n-dimensional input Tensor:
        log_softmax(x) = log(softmax(x))
    """
    return _to_tensor(input).log_softmax(dim=dim)


def binary_cross_entropy(
    input: Any,
    target: Any,
    weight: Optional[Any] = None,
    reduction: str = "mean",
) -> Tensor:
    """
    Measures the Binary Cross Entropy between target and input probabilities:
        L = - (target * log(input) + (1 - target) * log(1 - input))
    """
    in_t = _to_tensor(input)
    tgt_t = _to_tensor(target)
    w_t = _to_tensor(weight) if weight is not None else None
    return BinaryCrossEntropy.apply(in_t, tgt_t, w_t, reduction)


def binary_cross_entropy_with_logits(
    input: Any,
    target: Any,
    weight: Optional[Any] = None,
    pos_weight: Optional[Any] = None,
    reduction: str = "mean",
) -> Tensor:
    """
    Measures Binary Cross Entropy with logits for maximum numerical stability:
        L = max(input, 0) - input * target + log(1 + exp(-|input|))
    Supports pos_weight for imbalanced classification.
    """
    in_t = _to_tensor(input)
    tgt_t = _to_tensor(target)
    w_t = _to_tensor(weight) if weight is not None else None
    pw_t = _to_tensor(pos_weight) if pos_weight is not None else None
    return BinaryCrossEntropyWithLogits.apply(in_t, tgt_t, w_t, pw_t, reduction)


def cross_entropy(
    input: Any,
    target: Any,
    weight: Optional[Any] = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> Tensor:
    """
    Computes the cross entropy loss between input logits and target labels.
    """
    in_t = _to_tensor(input)
    tgt_t = _to_tensor(target)
    w_t = _to_tensor(weight) if weight is not None else None
    return CrossEntropy.apply(in_t, tgt_t, w_t, ignore_index, reduction, label_smoothing)


def nll_loss(
    input: Any,
    target: Any,
    weight: Optional[Any] = None,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> Tensor:
    """
    The negative log likelihood loss.
    Accepts log-probabilities (e.g. from log_softmax) and target class labels.
    """
    from nevula.autograd.functions.loss import NLLLoss as NLLLossFn
    in_t = _to_tensor(input)
    tgt_t = _to_tensor(target)
    w_t = _to_tensor(weight) if weight is not None else None
    return NLLLossFn.apply(in_t, tgt_t, w_t, ignore_index, reduction)


def mse_loss(
    input: Any,
    target: Any,
    reduction: str = "mean",
) -> Tensor:
    """
    Measures the mean squared error (squared L2 norm) between each element
    in the input and target.
    """
    in_t = _to_tensor(input)
    tgt_t = _to_tensor(target)
    diff = in_t - tgt_t
    loss = diff ** 2
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Invalid reduction '{reduction}', expected 'mean', 'sum', or 'none'")


def hinge_loss(
    input: Any,
    target: Any,
    margin: float = 1.0,
    reduction: str = "mean",
    squared: bool = False,
) -> Tensor:
    """
    Measures the Hinge loss between predictions and binary targets y in {-1, 1}:
        L = max(0, margin - target * input)
    or if squared=True:
        L = max(0, margin - target * input) ** 2
    """
    in_t = _to_tensor(input)
    tgt_t = _to_tensor(target)
    diff = float(margin) - tgt_t * in_t
    loss = diff.relu()
    if squared:
        loss = loss ** 2

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Invalid reduction '{reduction}', expected 'mean', 'sum', or 'none'")


__all__ = [
    "sigmoid",
    "log_sigmoid",
    "relu",
    "tanh",
    "softmax",
    "log_softmax",
    "binary_cross_entropy",
    "binary_cross_entropy_with_logits",
    "cross_entropy",
    "nll_loss",
    "mse_loss",
    "hinge_loss",
]
