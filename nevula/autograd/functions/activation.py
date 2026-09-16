import math
from typing import Any
from nevula.autograd.functions.base import Function
from nevula.autograd.node import Context


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, a: Any) -> Any:
        from nevula.core.tensor import Tensor
        ctx.save_for_backward(a)
        
        # Apply ReLU elementwise over raw flat data
        out_data = [max(0.0, float(a[idx])) for idx in a._indices_generator()]
        return Tensor(out_data, shape=a.shape, device=a.device)

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> tuple:
        from nevula.core.tensor import Tensor
        a, = ctx.saved_tensors
        
        mask_data = [1.0 if float(a[idx]) > 0.0 else 0.0 for idx in a._indices_generator()]
        mask = Tensor(mask_data, shape=a.shape, device=a.device)
        return grad_output * mask,


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, a: Any) -> Any:
        from nevula.core.tensor import Tensor
        
        def sig(x):
            val = float(x)
            if val < -700:
                return 0.0
            if val > 700:
                return 1.0
            return 1.0 / (1.0 + math.exp(-val))

        out_data = [sig(a[idx]) for idx in a._indices_generator()]
        out = Tensor(out_data, shape=a.shape, device=a.device)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> tuple:
        out, = ctx.saved_tensors
        grad_a = grad_output * out * (1.0 - out)
        return grad_a,


class Tanh(Function):
    @staticmethod
    def forward(ctx: Context, a: Any) -> Any:
        from nevula.core.tensor import Tensor
        out_data = [math.tanh(float(a[idx])) for idx in a._indices_generator()]
        out = Tensor(out_data, shape=a.shape, device=a.device)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> tuple:
        out, = ctx.saved_tensors
        grad_a = grad_output * (1.0 - out * out)
        return grad_a,


class Abs(Function):
    @staticmethod
    def forward(ctx: Context, a: Any) -> Any:
        from nevula.core.tensor import Tensor
        ctx.save_for_backward(a)
        out_data = [abs(float(a[idx])) for idx in a._indices_generator()]
        return Tensor(out_data, shape=a.shape, device=a.device)

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> tuple:
        from nevula.core.tensor import Tensor
        a, = ctx.saved_tensors

        def sign_fn(x):
            val = float(x)
            if val > 0.0:
                return 1.0
            elif val < 0.0:
                return -1.0
            return 0.0

        sign_data = [sign_fn(a[idx]) for idx in a._indices_generator()]
        sign = Tensor(sign_data, shape=a.shape, device=a.device)
        return grad_output * sign,

