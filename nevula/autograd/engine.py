from typing import Any, Optional
from nevula.autograd.graph import topological_sort
from nevula.autograd.node import Node


class GradMode:
    """
    Global control flag for automatic differentiation.
    When disabled, forward passes do not construct computation graphs.
    """
    _enabled: bool = True

    @classmethod
    def is_enabled(cls) -> bool:
        return cls._enabled

    @classmethod
    def set_enabled(cls, mode: bool) -> None:
        cls._enabled = mode


class no_grad:
    """
    Context-manager and decorator that disables gradient calculation.
    Disabling gradient calculation is useful for inference, evaluation,
    or updating parameters without tracking history.
    """
    def __init__(self):
        self.prev = True

    def __enter__(self):
        self.prev = GradMode.is_enabled()
        GradMode.set_enabled(False)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        GradMode.set_enabled(self.prev)

    def __call__(self, fn):
        def wrapper(*args, **kwargs):
            with self:
                return fn(*args, **kwargs)
        return wrapper


def unbroadcast(grad: Any, target_shape: tuple[int, ...]) -> Any:
    """
    Reduces broadcasted dimensions of `grad` so that its shape matches `target_shape`.
    """
    if not hasattr(grad, "shape") or grad.shape == target_shape:
        return grad

    grad_ndim = len(grad.shape)
    target_ndim = len(target_shape)
    leading = grad_ndim - target_ndim
    padded_target = (1,) * leading + target_shape

    axes_to_sum = []
    for i in range(grad_ndim):
        if grad.shape[i] > padded_target[i]:
            axes_to_sum.append(i)

    res = grad
    for axis in reversed(axes_to_sum):
        res = res.sum(axis=axis, keepdims=True)

    if res.shape != target_shape:
        res = res.reshape(target_shape)
    return res


def backward(
    tensor: Any,
    grad: Optional[Any] = None,
    retain_graph: bool = False
) -> None:
    """
    Computes the sum of gradients of given tensor with respect to graph leaves.
    The graph is differentiated using reverse-mode automatic differentiation.

    Args:
        tensor: The root tensor to differentiate (usually a scalar loss).
        grad: Initial gradient w.r.t. the root tensor. If None and tensor is scalar,
              defaults to Tensor(1.0). If tensor is not scalar, defaults to ones_like(tensor).
        retain_graph: If False, saved tensors and graph references are freed after backward.
    """
    # Import Tensor lazily to avoid circular imports
    from nevula.core.tensor import Tensor

    if tensor.grad_fn is None:
        if tensor.requires_grad:
            if grad is None:
                tensor.grad = Tensor(1.0) if tensor.shape == () else Tensor.ones_like(tensor)
            else:
                tensor.grad = grad
        return

    if grad is None:
        if tensor.shape == ():
            grad = Tensor(1.0)
        else:
            grad = Tensor.ones_like(tensor)

    # 1. Topological sort of the computational graph
    nodes = topological_sort(tensor.grad_fn)

    # 2. Map of accumulated incoming gradients for each node
    node_grads: dict[Node, Any] = {tensor.grad_fn: grad}

    # 3. Traverse nodes in reverse-topological order
    for node in nodes:
        curr_grad = node_grads.get(node)
        if curr_grad is None:
            continue

        input_grads = node.backward(curr_grad)
        if not isinstance(input_grads, (tuple, list)):
            input_grads = (input_grads,)

        for (parent_fn, input_tensor), in_grad in zip(node.next_functions, input_grads):
            if in_grad is None or not isinstance(input_tensor, Tensor):
                continue

            reduced_grad = unbroadcast(in_grad, input_tensor.shape)

            # Route to parent node if intermediate
            if parent_fn is not None:
                if parent_fn in node_grads:
                    node_grads[parent_fn] = node_grads[parent_fn] + reduced_grad
                else:
                    node_grads[parent_fn] = reduced_grad

            # Accumulate into leaf tensor's grad attribute
            if input_tensor.is_leaf and input_tensor.requires_grad:
                if input_tensor.grad is None:
                    input_tensor.grad = reduced_grad
                else:
                    input_tensor.grad = input_tensor.grad + reduced_grad

            # Handle non-leaf tensors with retain_grad() enabled
            elif getattr(input_tensor, "_retains_grad", False):
                if input_tensor.grad is None:
                    input_tensor.grad = reduced_grad
                else:
                    input_tensor.grad = input_tensor.grad + reduced_grad

        # Free saved tensors if graph is not retained
        if not retain_graph:
            node.ctx._saved_tensors = ()
