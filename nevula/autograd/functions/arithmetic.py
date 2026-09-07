from typing import Any
from nevula.autograd.functions.base import Function
from nevula.autograd.node import Context


class Add(Function):
    @staticmethod
    def forward(ctx: Context, a: Any, b: Any) -> Any:
        ctx.save_for_backward(a, b)
        return a._add_raw(b)

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> tuple:
        return grad_output, grad_output


class Sub(Function):
    @staticmethod
    def forward(ctx: Context, a: Any, b: Any) -> Any:
        ctx.save_for_backward(a, b)
        return a._sub_raw(b)

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> tuple:
        return grad_output, -grad_output


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, a: Any, b: Any) -> Any:
        ctx.save_for_backward(a, b)
        return a._mul_raw(b)

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> tuple:
        a, b = ctx.saved_tensors
        grad_a = grad_output * b
        grad_b = grad_output * a
        return grad_a, grad_b


class Div(Function):
    @staticmethod
    def forward(ctx: Context, a: Any, b: Any) -> Any:
        ctx.save_for_backward(a, b)
        return a._div_raw(b)

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> tuple:
        a, b = ctx.saved_tensors
        grad_a = grad_output / b
        grad_b = -grad_output * a / (b * b)
        return grad_a, grad_b


class Pow(Function):
    @staticmethod
    def forward(ctx: Context, a: Any, p: Any) -> Any:
        ctx.save_for_backward(a)
        ctx.p = p
        return a._pow_raw(p)

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> tuple:
        a, = ctx.saved_tensors
        p = ctx.p
        p_val = p.to_list() if hasattr(p, "to_list") else p
        grad_a = grad_output * p_val * (a ** (p_val - 1))
        return grad_a, None


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, a: Any) -> Any:
        return a._neg_raw()

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> tuple:
        return (-grad_output,)
