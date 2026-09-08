from nevula.core.tensor import Tensor
from nevula.autograd.engine import backward, no_grad, GradMode
from nevula import autograd
from nevula import core
from nevula import nn
from nevula import losses
from nevula import optim
from nevula import data
from nevula import utils
from nevula.utils.serialization import save, load, save_checkpoint, load_checkpoint


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
    "optim",
    "data",
    "utils",
    "save",
    "load",
    "save_checkpoint",
    "load_checkpoint",
]
