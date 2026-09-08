import math
from typing import Any, Iterable, Union
from nevula.core.tensor import Tensor


def check_for_nan(tensor: Tensor) -> bool:
    """Returns True if any element in tensor is NaN."""
    for val in tensor.data:
        if math.isnan(val):
            return True
    return False


def check_for_inf(tensor: Tensor) -> bool:
    """Returns True if any element in tensor is infinite (+inf or -inf)."""
    for val in tensor.data:
        if math.isinf(val):
            return True
    return False


def inspect_gradients(model_or_parameters: Any) -> dict[str, dict[str, Any]]:
    """
    Computes diagnostic statistics on gradients of model parameters.

    Returns a dictionary mapping parameter names to stats:
        shape, has_grad, has_nan, has_inf, norm, mean, std, min, max.
    """
    if hasattr(model_or_parameters, "named_parameters"):
        named_params = list(model_or_parameters.named_parameters())
    elif isinstance(model_or_parameters, (list, tuple)):
        named_params = [(f"param_{i}", p) for i, p in enumerate(model_or_parameters)]
    else:
        named_params = [("param", model_or_parameters)]

    diagnostics = {}

    for name, p in named_params:
        stats: dict[str, Any] = {
            "shape": p.shape,
            "has_grad": p.grad is not None,
        }

        if p.grad is not None:
            g_data = p.grad.data
            stats["has_nan"] = any(math.isnan(g) for g in g_data)
            stats["has_inf"] = any(math.isinf(g) for g in g_data)

            finite_vals = [g for g in g_data if not math.isnan(g) and not math.isinf(g)]
            if len(finite_vals) > 0:
                stats["norm"] = math.sqrt(sum(g * g for g in finite_vals))
                stats["mean"] = sum(finite_vals) / len(finite_vals)
                variance = sum((g - stats["mean"]) ** 2 for g in finite_vals) / len(finite_vals)
                stats["std"] = math.sqrt(variance)
                stats["min"] = min(finite_vals)
                stats["max"] = max(finite_vals)
            else:
                stats["norm"] = float("nan")
                stats["mean"] = float("nan")
                stats["std"] = float("nan")
                stats["min"] = float("nan")
                stats["max"] = float("nan")
        else:
            stats["has_nan"] = False
            stats["has_inf"] = False
            stats["norm"] = 0.0
            stats["mean"] = 0.0
            stats["std"] = 0.0
            stats["min"] = 0.0
            stats["max"] = 0.0

        diagnostics[name] = stats

    return diagnostics


def print_gradient_diagnostics(model_or_parameters: Any) -> None:
    """
    Prints a formatted, human-readable report of gradient statistics.
    """
    diag = inspect_gradients(model_or_parameters)

    print("\nGradient Diagnostics")
    print("─" * 45)
    for name, stats in diag.items():
        print(f"{name} (shape={stats['shape']})")
        if not stats["has_grad"]:
            print("  Status: No gradient computed")
        else:
            print(f"  norm : {stats['norm']:.6f}")
            print(f"  mean : {stats['mean']:.6f}")
            print(f"  std  : {stats['std']:.6f}")
            print(f"  min  : {stats['min']:.6f}")
            print(f"  max  : {stats['max']:.6f}")
            print(f"  NaN  : {stats['has_nan']}")
            print(f"  Inf  : {stats['has_inf']}")
        print()
