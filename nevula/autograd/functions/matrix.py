from typing import Any
from nevula.autograd.functions.base import Function
from nevula.autograd.node import Context


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, a: Any, b: Any) -> Any:
        ctx.save_for_backward(a, b)
        return a._matmul_raw(b)

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> tuple:
        a, b = ctx.saved_tensors
        # grad_a = grad @ b.T
        # grad_b = a.T @ grad
        b_t = b.transpose(-2, -1) if len(b.shape) >= 2 else b
        a_t = a.transpose(-2, -1) if len(a.shape) >= 2 else a
        
        grad_a = grad_output.matmul(b_t)
        grad_b = a_t.matmul(grad_output)
        return grad_a, grad_b


class Transpose(Function):
    @staticmethod
    def forward(ctx: Context, a: Any, axis1: int, axis2: int) -> Any:
        ctx.axis1 = axis1
        ctx.axis2 = axis2
        return a.transpose(axis1, axis2)

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> tuple:
        return grad_output.transpose(ctx.axis1, ctx.axis2), None, None
