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


class LogSigmoid(Function):
    """
    Numerically stable elementwise LogSigmoid:
        LogSigmoid(x) = log(1 / (1 + exp(-x))) = -softplus(-x)
    """

    @staticmethod
    def forward(ctx: Context, a: Any) -> Any:
        from nevula.core.tensor import Tensor

        def log_sig(x):
            val = float(x)
            if val >= 0:
                return -math.log1p(math.exp(-val))
            else:
                return val - math.log1p(math.exp(val))

        out_data = [log_sig(a[idx]) for idx in a._indices_generator()]
        out = Tensor(out_data, shape=a.shape, device=a.device)
        ctx.save_for_backward(a)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> tuple:
        from nevula.core.tensor import Tensor
        a, = ctx.saved_tensors

        def d_log_sig(x):
            val = float(x)
            # Derivative is 1 - sigmoid(x) = sigmoid(-x)
            if val < -700:
                return 1.0
            if val > 700:
                return 0.0
            return 1.0 / (1.0 + math.exp(val))

        grad_data = [d_log_sig(a[idx]) for idx in a._indices_generator()]
        grad_tensor = Tensor(grad_data, shape=a.shape, device=a.device)
        return grad_output * grad_tensor,


class Softmax(Function):
    """
    Applies the Softmax function to an n-dimensional input Tensor,
    rescaling elements along dim so that they lie in range [0, 1] and sum to 1:
        Softmax(z_i) = exp(z_i) / sum_j exp(z_j)
    """

    @staticmethod
    def forward(ctx: Context, a: Any, dim: int = -1) -> Any:
        from nevula.core.tensor import Tensor
        import numpy as np

        a_np = np.array(a.to_list(), dtype=np.float64)
        # Normalize dimension
        ndim = a_np.ndim
        norm_dim = dim if dim >= 0 else ndim + dim

        # Subtract max along axis for numerical stability
        max_vals = np.max(a_np, axis=norm_dim, keepdims=True)
        exp_a = np.exp(a_np - max_vals)
        sum_exp = np.sum(exp_a, axis=norm_dim, keepdims=True)
        probs_np = exp_a / (sum_exp + 1e-15)

        out = Tensor(probs_np, shape=a.shape, device=a.device)
        ctx.save_for_backward(out)
        ctx.dim = norm_dim
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> tuple:
        from nevula.core.tensor import Tensor
        import numpy as np

        out, = ctx.saved_tensors
        dim = ctx.dim

        out_np = np.array(out.to_list(), dtype=np.float64)
        grad_np = np.array(grad_output.to_list(), dtype=np.float64)

        # Vector-Jacobian product:
        # d_loss / d_z = p * (g - sum(g * p, axis=dim, keepdims=True))
        sum_gp = np.sum(grad_np * out_np, axis=dim, keepdims=True)
        grad_input_np = out_np * (grad_np - sum_gp)

        grad_input = Tensor(grad_input_np, shape=out.shape, device=out.device)
        return grad_input, None


class LogSoftmax(Function):
    """
    Applies the LogSoftmax function to an n-dimensional input Tensor:
        LogSoftmax(z_i) = log(exp(z_i) / sum_j exp(z_j))
    """

    @staticmethod
    def forward(ctx: Context, a: Any, dim: int = -1) -> Any:
        from nevula.core.tensor import Tensor
        import numpy as np

        a_np = np.array(a.to_list(), dtype=np.float64)
        ndim = a_np.ndim
        norm_dim = dim if dim >= 0 else ndim + dim

        max_vals = np.max(a_np, axis=norm_dim, keepdims=True)
        shifted = a_np - max_vals
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=norm_dim, keepdims=True) + 1e-15)
        log_probs_np = shifted - log_sum_exp

        # Also precompute softmax probabilities for backward
        probs_np = np.exp(log_probs_np)

        out = Tensor(log_probs_np, shape=a.shape, device=a.device)
        probs_t = Tensor(probs_np, shape=a.shape, device=a.device)
        ctx.save_for_backward(probs_t)
        ctx.dim = norm_dim
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> tuple:
        from nevula.core.tensor import Tensor
        import numpy as np

        probs_t, = ctx.saved_tensors
        dim = ctx.dim

        probs_np = np.array(probs_t.to_list(), dtype=np.float64)
        grad_np = np.array(grad_output.to_list(), dtype=np.float64)

        # Vector-Jacobian product:
        # d_loss / d_z = g - p * sum(g, axis=dim, keepdims=True)
        sum_g = np.sum(grad_np, axis=dim, keepdims=True)
        grad_input_np = grad_np - probs_np * sum_g

        grad_input = Tensor(grad_input_np, shape=probs_t.shape, device=probs_t.device)
        return grad_input, None


