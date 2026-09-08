import math
from typing import Any, Iterable, Union
from nevula.core.tensor import Tensor


def clip_grad_norm_(
    parameters: Union[Iterable[Any], Any],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
) -> float:
    """
    Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters: An iterable of Tensors or a single Tensor that will have gradients normalized.
        max_norm: Max norm of the gradients.
        norm_type: Type of the used p-norm. Currently supports L2 norm (2.0) and Linf norm (inf).
        error_if_nonfinite: If True, an error is thrown if the total norm is NaN or Inf.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, Tensor):
        parameters = [parameters]

    params = [p for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)

    if len(params) == 0:
        return 0.0

    if norm_type == float("inf"):
        total_norm = max(max(abs(g) for g in p.grad.data) for p in params)
    elif norm_type == 2.0:
        total_norm = math.sqrt(sum(sum(g * g for g in p.grad.data) for p in params))
    else:
        total_norm = sum(sum(abs(g) ** norm_type for g in p.grad.data) for p in params) ** (1.0 / norm_type)

    if math.isnan(total_norm) or math.isinf(total_norm):
        if error_if_nonfinite:
            raise RuntimeError(
                f"The total norm of order {norm_type} for gradients from "
                "`parameters` is non-finite, so it cannot be clipped."
            )

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in params:
            for i in range(len(p.grad.data)):
                p.grad.data[i] *= clip_coef

    return total_norm
