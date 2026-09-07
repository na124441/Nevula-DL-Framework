from typing import Any
from nevula.autograd.functions.base import Function
from nevula.autograd.node import Context


class Reshape(Function):
    @staticmethod
    def forward(ctx: Context, a: Any, shape: tuple[int, ...]) -> Any:
        ctx.in_shape = a.shape
        return a._reshape_raw(shape)

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> tuple:
        return grad_output.reshape(ctx.in_shape), None
