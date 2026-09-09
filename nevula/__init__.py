from nevula.core.tensor import Tensor
from nevula.autograd.engine import backward, no_grad, GradMode
from nevula.backend.device import Device, cuda_available
from nevula.graph.capture import graph_capture
from nevula.engine.compile import compile
from nevula import autograd
from nevula import backend
from nevula import core
from nevula import graph
from nevula import engine
from nevula import nn
from nevula import losses
from nevula import optim
from nevula import data
from nevula import utils
from nevula.utils.serialization import save, load, save_checkpoint, load_checkpoint


def tensor(data, requires_grad: bool = False, shape=None, device="cpu"):
    """Factory function for creating a Nevula Tensor."""
    return Tensor(data, shape=shape, requires_grad=requires_grad, device=device)


__all__ = [
    "Tensor",
    "tensor",
    "Device",
    "cuda_available",
    "graph_capture",
    "compile",
    "backward",
    "no_grad",
    "GradMode",
    "autograd",
    "backend",
    "core",
    "graph",
    "engine",
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

