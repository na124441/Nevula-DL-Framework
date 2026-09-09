import math
from typing import Any
from nevula.autograd.functions.base import Function
from nevula.autograd.node import Context


class CrossEntropy(Function):
    """
    Differentiable numerically-stable Softmax + Cross-Entropy Loss.
    Supports input logits of shape (N, C) and target class indices of shape (N,)
    or one-hot distributions of shape (N, C).
    """

    @staticmethod
    def forward(ctx: Context, logits: Any, target: Any, reduction: str = "mean") -> Any:
        from nevula.core.tensor import Tensor

        if len(logits.shape) != 2:
            raise ValueError(f"CrossEntropy expects 2D logits (N, C), got shape {logits.shape}")

        n, c = logits.shape

        # Calculate numerically stable softmax probabilities
        probs_data = []
        losses = []

        is_one_hot = (len(target.shape) == 2 and target.shape == (n, c))

        for i in range(n):
            row = [logits[i, j] for j in range(c)]
            max_val = max(row)
            exp_row = [math.exp(val - max_val) for val in row]
            sum_exp = sum(exp_row)
            p_row = [e / sum_exp for e in exp_row]
            probs_data.extend(p_row)

            if is_one_hot:
                loss_i = 0.0
                for j in range(c):
                    t_val = target[i, j]
                    if t_val > 0:
                        loss_i -= t_val * math.log(max(p_row[j], 1e-15))
                losses.append(loss_i)
            else:
                target_idx = int(target[i] if len(target.shape) == 1 else target[()])
                p_target = max(p_row[target_idx], 1e-15)
                losses.append(-math.log(p_target))

        probs = Tensor(probs_data, shape=(n, c), device=logits.device)
        ctx.save_for_backward(probs, target)
        ctx.reduction = reduction
        ctx.is_one_hot = is_one_hot

        if reduction == "mean":
            return Tensor(sum(losses) / n, device=logits.device)
        elif reduction == "sum":
            return Tensor(sum(losses), device=logits.device)
        else:
            return Tensor(losses, shape=(n,), device=logits.device)

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> tuple:
        from nevula.core.tensor import Tensor

        probs, target = ctx.saved_tensors
        reduction = ctx.reduction
        is_one_hot = ctx.is_one_hot

        n, c = probs.shape
        grad_data = []

        scale = 1.0 / n if reduction == "mean" else 1.0
        grad_val = grad_output.item() if hasattr(grad_output, "item") and len(grad_output.shape) == 0 else 1.0

        for i in range(n):
            g_factor = grad_output[i] if reduction == "none" else grad_val
            target_idx = -1 if is_one_hot else int(target[i] if len(target.shape) == 1 else target[()])

            for j in range(c):
                p_ij = probs[i, j]
                t_ij = target[i, j] if is_one_hot else (1.0 if j == target_idx else 0.0)
                d_logits = (p_ij - t_ij) * scale * g_factor
                grad_data.append(d_logits)

        return Tensor(grad_data, shape=(n, c), device=probs.device), None, None

