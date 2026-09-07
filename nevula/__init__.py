from nevula.core.tensor import Tensor
from nevula.autograd.engine import backward, no_grad, GradMode
from nevula import autograd
from nevula import core
from nevula import nn
from nevula import losses


def tensor(data, requires_grad: bool = False, shape=None):
    """Factory function for creating a Nevula Tensor."""
    return Tensor(data, shape=shape, requires_grad=requires_grad)


__all__ = [
    "Tensor",
    "tensor",
    "backward",
    "no_grad",
    "GradMode",
    "autograd",
    "core",
    "nn",
    "losses",
]
