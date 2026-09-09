from typing import Any, Optional
from nevula.autograd.functions.base import Function
from nevula.autograd.node import Context


def _prod(shape: tuple[int, ...]) -> int:
    res = 1
    for x in shape:
        res *= x
    return res


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, a: Any, axis: Optional[Any] = None, keepdims: bool = False) -> Any:
        ctx.in_shape = a.shape
        ctx.axis = axis
        ctx.keepdims = keepdims
        return a._sum_raw(axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> tuple:
        from nevula.core.tensor import Tensor

        in_shape = ctx.in_shape
        axis = ctx.axis
        keepdims = ctx.keepdims

        if axis is None:
            # Reduced all dimensions to scalar
            grad_expanded = grad_output + Tensor.zeros(in_shape, device=grad_output.device)
        else:
            # Normalize axis to tuple of non-negative ints
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            ndim = len(in_shape)
            normalized_axes = {ax if ax >= 0 else ax + ndim for ax in axes}

            if not keepdims:
                # Re-insert 1s at reduced dimensions
                target_shape = []
                grad_idx = 0
                for i in range(ndim):
                    if i in normalized_axes:
                        target_shape.append(1)
                    else:
                        target_shape.append(grad_output.shape[grad_idx])
                        grad_idx += 1
                grad_reshaped = grad_output.reshape(tuple(target_shape))
            else:
                grad_reshaped = grad_output

            grad_expanded = grad_reshaped + Tensor.zeros(in_shape, device=grad_output.device)

        return grad_expanded, None, None


class Mean(Function):
    @staticmethod
    def forward(ctx: Context, a: Any, axis: Optional[Any] = None, keepdims: bool = False) -> Any:
        ctx.in_shape = a.shape
        ctx.axis = axis
        ctx.keepdims = keepdims
        out = a._mean_raw(axis=axis, keepdims=keepdims)
        ctx.out_shape = out.shape
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> tuple:
        from nevula.core.tensor import Tensor

        in_shape = ctx.in_shape
        axis = ctx.axis
        keepdims = ctx.keepdims
        
        num_elements = _prod(in_shape) // max(1, _prod(ctx.out_shape))
        scaled_grad = grad_output / num_elements

        if axis is None:
            grad_expanded = scaled_grad + Tensor.zeros(in_shape, device=grad_output.device)
        else:
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            ndim = len(in_shape)
            normalized_axes = {ax if ax >= 0 else ax + ndim for ax in axes}

            if not keepdims:
                target_shape = []
                grad_idx = 0
                for i in range(ndim):
                    if i in normalized_axes:
                        target_shape.append(1)
                    else:
                        target_shape.append(scaled_grad.shape[grad_idx])
                        grad_idx += 1
                grad_reshaped = scaled_grad.reshape(tuple(target_shape))
            else:
                grad_reshaped = scaled_grad

            grad_expanded = grad_reshaped + Tensor.zeros(in_shape, device=grad_output.device)

        return grad_expanded, None, None

