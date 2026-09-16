import math
from typing import Any, Optional
import numpy as np
from nevula.autograd.functions.base import Function
from nevula.autograd.node import Context


class CrossEntropy(Function):
    """
    Differentiable, numerically stable Softmax + Cross-Entropy Loss.
    Supports:
      - 2D logits of shape (N, C)
      - Target class indices of shape (N,) or one-hot/probabilistic targets of shape (N, C)
      - Optional class weights of shape (C,)
      - Label smoothing
      - ignore_index
      - Reduction: 'mean', 'sum', 'none'
    """

    @staticmethod
    def forward(
        ctx: Context,
        logits: Any,
        target: Any,
        weight: Any = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> Any:
        from nevula.core.tensor import Tensor

        if len(logits.shape) != 2:
            raise ValueError(f"CrossEntropy expects 2D logits (N, C), got shape {logits.shape}")

        n, c = logits.shape
        logits_np = np.array(logits.to_list(), dtype=np.float64)
        target_np = np.array(target.to_list(), dtype=np.float64)

        has_weight = weight is not None
        weight_np = np.array(weight.to_list(), dtype=np.float64) if has_weight else None

        # Subtract max along class dimension for numerical stability
        max_logits = np.max(logits_np, axis=1, keepdims=True)
        exp_logits = np.exp(logits_np - max_logits)
        sum_exp = np.sum(exp_logits, axis=1, keepdims=True)
        probs_np = exp_logits / (sum_exp + 1e-15)
        log_probs_np = (logits_np - max_logits) - np.log(sum_exp + 1e-15)

        is_one_hot = (target_np.ndim == 2 and target_np.shape == (n, c))

        # Convert target to probability distribution if label_smoothing is active or if integer targets
        if not is_one_hot:
            target_indices = target_np.astype(int).ravel()
            target_dist = np.zeros((n, c), dtype=np.float64)
            valid_mask = np.ones(n, dtype=bool)

            for i in range(n):
                idx = target_indices[i]
                if idx == ignore_index:
                    valid_mask[i] = False
                elif 0 <= idx < c:
                    if label_smoothing > 0.0:
                        target_dist[i, :] = label_smoothing / c
                        target_dist[i, idx] = (1.0 - label_smoothing) + (label_smoothing / c)
                    else:
                        target_dist[i, idx] = 1.0
                else:
                    raise ValueError(f"Target index {idx} out of bounds for {c} classes.")
        else:
            target_dist = target_np
            valid_mask = np.ones(n, dtype=bool)
            if label_smoothing > 0.0:
                target_dist = (1.0 - label_smoothing) * target_dist + (label_smoothing / c)

        # Compute loss per sample
        losses_np = -np.sum(target_dist * log_probs_np, axis=1)

        # Apply class weights if provided
        weights_sum = 0.0
        sample_weights = np.ones(n, dtype=np.float64)
        for i in range(n):
            if not valid_mask[i]:
                losses_np[i] = 0.0
                sample_weights[i] = 0.0
            elif has_weight:
                if not is_one_hot:
                    idx = target_indices[i]
                    w = weight_np[idx]
                else:
                    w = np.sum(target_dist[i] * weight_np)
                losses_np[i] *= w
                sample_weights[i] = w
                weights_sum += w
            else:
                weights_sum += 1.0

        ctx.save_for_backward(logits, target)
        ctx.probs_np = probs_np
        ctx.target_dist = target_dist
        ctx.valid_mask = valid_mask
        ctx.sample_weights = sample_weights
        ctx.weights_sum = weights_sum
        ctx.reduction = reduction
        ctx.has_weight = has_weight
        ctx.weight_np = weight_np

        if reduction == "mean":
            denom = max(weights_sum, 1e-12)
            total = float(np.sum(losses_np) / denom)
            return Tensor(total, device=logits.device)
        elif reduction == "sum":
            return Tensor(float(np.sum(losses_np)), device=logits.device)
        else:
            return Tensor(losses_np, shape=(n,), device=logits.device)

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> tuple:
        from nevula.core.tensor import Tensor

        probs_np = ctx.probs_np
        target_dist = ctx.target_dist
        valid_mask = ctx.valid_mask
        sample_weights = ctx.sample_weights
        weights_sum = ctx.weights_sum
        reduction = ctx.reduction

        n, c = probs_np.shape
        grad_np = np.array(grad_output.to_list(), dtype=np.float64)

        denom = max(weights_sum, 1e-12) if reduction == "mean" else 1.0
        scale = 1.0 / denom

        # Gradient of cross-entropy w.r.t logits: (probs - target_dist) * weight * scale * grad_output
        grad_logits = (probs_np - target_dist)

        for i in range(n):
            if not valid_mask[i]:
                grad_logits[i, :] = 0.0
            else:
                g = grad_np[i] if reduction == "none" else grad_np
                w = sample_weights[i]
                grad_logits[i, :] *= (w * scale * g)

        return (
            Tensor(grad_logits, shape=(n, c)),
            None,
            None,
            None,
            None,
            None,
        )


class NLLLoss(Function):
    """
    Negative Log Likelihood Loss.
    Takes log-probabilities (e.g. from LogSoftmax) and target class labels.
    """

    @staticmethod
    def forward(
        ctx: Context,
        log_probs: Any,
        target: Any,
        weight: Any = None,
        ignore_index: int = -100,
        reduction: str = "mean",
    ) -> Any:
        from nevula.core.tensor import Tensor

        if len(log_probs.shape) != 2:
            raise ValueError(f"NLLLoss expects 2D log-probabilities (N, C), got shape {log_probs.shape}")

        n, c = log_probs.shape
        lp_np = np.array(log_probs.to_list(), dtype=np.float64)
        target_np = np.array(target.to_list(), dtype=np.float64)

        has_weight = weight is not None
        weight_np = np.array(weight.to_list(), dtype=np.float64) if has_weight else None

        is_one_hot = (target_np.ndim == 2 and target_np.shape == (n, c))
        losses_np = np.zeros(n, dtype=np.float64)
        valid_mask = np.ones(n, dtype=bool)
        sample_weights = np.ones(n, dtype=np.float64)
        weights_sum = 0.0

        for i in range(n):
            if not is_one_hot:
                idx = int(target_np[i])
                if idx == ignore_index:
                    valid_mask[i] = False
                    sample_weights[i] = 0.0
                elif 0 <= idx < c:
                    w = weight_np[idx] if has_weight else 1.0
                    losses_np[i] = -lp_np[i, idx] * w
                    sample_weights[i] = w
                    weights_sum += w
                else:
                    raise ValueError(f"Target index {idx} out of bounds for {c} classes.")
            else:
                w = np.sum(target_np[i] * weight_np) if has_weight else 1.0
                losses_np[i] = -np.sum(target_np[i] * lp_np[i]) * w
                sample_weights[i] = w
                weights_sum += w

        ctx.save_for_backward(log_probs, target)
        ctx.lp_np = lp_np
        ctx.target_np = target_np
        ctx.is_one_hot = is_one_hot
        ctx.valid_mask = valid_mask
        ctx.sample_weights = sample_weights
        ctx.weights_sum = weights_sum
        ctx.reduction = reduction
        ctx.ignore_index = ignore_index

        if reduction == "mean":
            denom = max(weights_sum, 1e-12)
            return Tensor(float(np.sum(losses_np) / denom), device=log_probs.device)
        elif reduction == "sum":
            return Tensor(float(np.sum(losses_np)), device=log_probs.device)
        else:
            return Tensor(losses_np, shape=(n,), device=log_probs.device)

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> tuple:
        from nevula.core.tensor import Tensor

        lp_np = ctx.lp_np
        target_np = ctx.target_np
        is_one_hot = ctx.is_one_hot
        valid_mask = ctx.valid_mask
        sample_weights = ctx.sample_weights
        weights_sum = ctx.weights_sum
        reduction = ctx.reduction
        ignore_index = ctx.ignore_index

        n, c = lp_np.shape
        grad_np = np.array(grad_output.to_list(), dtype=np.float64)

        denom = max(weights_sum, 1e-12) if reduction == "mean" else 1.0
        scale = 1.0 / denom

        grad_input = np.zeros((n, c), dtype=np.float64)
        for i in range(n):
            if valid_mask[i]:
                g = grad_np[i] if reduction == "none" else grad_np
                w = sample_weights[i]
                if not is_one_hot:
                    idx = int(target_np[i])
                    grad_input[i, idx] = -w * scale * g
                else:
                    grad_input[i, :] = -target_np[i] * w * scale * g

        return Tensor(grad_input, shape=(n, c)), None, None, None, None


class BinaryCrossEntropyWithLogits(Function):
    """
    Numerically stable Binary Cross-Entropy Loss with Logits.
    Combines a Sigmoid layer and the BCELoss in one single differentiable class.
    Supports optional pos_weight for imbalanced classification:
        loss_i = (1 - y_i) * log(1 + exp(z_i)) + y_i * pos_weight * log(1 + exp(-z_i))
    """

    @staticmethod
    def forward(
        ctx: Context,
        logits: Any,
        target: Any,
        weight: Any = None,
        pos_weight: Any = None,
        reduction: str = "mean",
    ) -> Any:
        from nevula.core.tensor import Tensor

        flat_logits = [float(logits[idx]) for idx in logits._indices_generator()]
        flat_target = [float(target[idx]) for idx in target._indices_generator()]

        if len(flat_logits) != len(flat_target):
            raise ValueError(f"Logits shape {logits.shape} does not match target shape {target.shape}")

        has_weight = weight is not None
        flat_weight = [float(weight[idx]) for idx in weight._indices_generator()] if has_weight else None

        has_pos_weight = pos_weight is not None
        flat_pos_weight = [float(pos_weight[idx]) for idx in pos_weight._indices_generator()] if has_pos_weight else None

        losses = []
        sigmoids = []

        for k, (z, y) in enumerate(zip(flat_logits, flat_target)):
            # Numerically stable calculation
            # If pos_weight is None:
            #   loss = max(z, 0) - z*y + log1p(exp(-abs(z)))
            # If pos_weight is specified:
            #   pw = pos_weight[k]
            #   loss = (1 - y)*log(1 + exp(z)) + pw*y*log(1 + exp(-z))
            #        = (1 - y)*(max(z, 0) + log1p(exp(-abs(z)))) + pw*y*(max(-z, 0) + log1p(exp(-abs(z))))
            abs_z = abs(z)
            log1p_exp_neg_abs = math.log1p(math.exp(-abs_z))

            if flat_pos_weight is not None:
                pw = flat_pos_weight[k]
                term_neg = max(z, 0.0) + log1p_exp_neg_abs
                term_pos = max(-z, 0.0) + log1p_exp_neg_abs
                loss_val = (1.0 - y) * term_neg + pw * y * term_pos
            else:
                loss_val = max(z, 0.0) - z * y + log1p_exp_neg_abs

            if flat_weight is not None:
                loss_val *= flat_weight[k]
            losses.append(loss_val)

            # Compute sigmoid(z) for backward pass
            if z >= 0:
                sig_z = 1.0 / (1.0 + math.exp(-z))
            else:
                ez = math.exp(z)
                sig_z = ez / (1.0 + ez)
            sigmoids.append(sig_z)

        n = len(losses)
        ctx.save_for_backward(logits, target, weight, pos_weight)
        ctx.sigmoids = sigmoids
        ctx.reduction = reduction
        ctx.n = n

        if reduction == "mean":
            return Tensor(sum(losses) / max(n, 1), device=logits.device)
        elif reduction == "sum":
            return Tensor(sum(losses), device=logits.device)
        else:
            return Tensor(losses, shape=logits.shape, device=logits.device)

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> tuple:
        from nevula.core.tensor import Tensor

        logits, target, weight, pos_weight = ctx.saved_tensors
        sigmoids = ctx.sigmoids
        reduction = ctx.reduction
        n = ctx.n

        flat_target = [float(target[idx]) for idx in target._indices_generator()]
        flat_weight = [float(weight[idx]) for idx in weight._indices_generator()] if weight is not None else None
        flat_pos_weight = [float(pos_weight[idx]) for idx in pos_weight._indices_generator()] if pos_weight is not None else None

        scale = (1.0 / n) if reduction == "mean" else 1.0
        grad_val = grad_output.item() if hasattr(grad_output, "item") and len(grad_output.shape) == 0 else 1.0

        grad_logits = []
        for k, (sig_z, y) in enumerate(zip(sigmoids, flat_target)):
            g_factor = float(grad_output[k]) if reduction == "none" else grad_val

            if flat_pos_weight is not None:
                pw = flat_pos_weight[k]
                # d/dz = sig_z * (1 + (pw - 1)*y) - pw*y
                d_z = (sig_z * (1.0 + (pw - 1.0) * y) - pw * y) * scale * g_factor
            else:
                # d/dz = sig_z - y
                d_z = (sig_z - y) * scale * g_factor

            if flat_weight is not None:
                d_z *= flat_weight[k]
            grad_logits.append(d_z)

        return Tensor(grad_logits, shape=logits.shape, device=logits.device), None, None, None, None


class BinaryCrossEntropy(Function):
    """
    Standard Binary Cross-Entropy Loss on predicted probabilities p in (0, 1):
        loss_i = -(y_i * log(p_i) + (1 - y_i) * log(1 - p_i))
    """

    @staticmethod
    def forward(ctx: Context, probs: Any, target: Any, weight: Any = None, reduction: str = "mean") -> Any:
        from nevula.core.tensor import Tensor

        flat_probs = [float(probs[idx]) for idx in probs._indices_generator()]
        flat_target = [float(target[idx]) for idx in target._indices_generator()]

        if len(flat_probs) != len(flat_target):
            raise ValueError(f"Probs shape {probs.shape} does not match target shape {target.shape}")

        has_weight = weight is not None
        flat_weight = [float(weight[idx]) for idx in weight._indices_generator()] if has_weight else None

        eps = 1e-12
        losses = []
        clamped_probs = []

        for k, (p, y) in enumerate(zip(flat_probs, flat_target)):
            p_clamped = max(eps, min(1.0 - eps, p))
            clamped_probs.append(p_clamped)
            loss_val = -(y * math.log(p_clamped) + (1.0 - y) * math.log(1.0 - p_clamped))
            if flat_weight is not None:
                loss_val *= flat_weight[k]
            losses.append(loss_val)

        n = len(losses)
        ctx.save_for_backward(probs, target, weight)
        ctx.clamped_probs = clamped_probs
        ctx.reduction = reduction
        ctx.n = n

        if reduction == "mean":
            return Tensor(sum(losses) / max(n, 1), device=probs.device)
        elif reduction == "sum":
            return Tensor(sum(losses), device=probs.device)
        else:
            return Tensor(losses, shape=probs.shape, device=probs.device)

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> tuple:
        from nevula.core.tensor import Tensor

        probs, target, weight = ctx.saved_tensors
        clamped_probs = ctx.clamped_probs
        reduction = ctx.reduction
        n = ctx.n

        flat_target = [float(target[idx]) for idx in target._indices_generator()]
        flat_weight = [float(weight[idx]) for idx in weight._indices_generator()] if weight is not None else None

        scale = (1.0 / n) if reduction == "mean" else 1.0
        grad_val = grad_output.item() if hasattr(grad_output, "item") and len(grad_output.shape) == 0 else 1.0

        grad_probs = []
        for k, (p, y) in enumerate(zip(clamped_probs, flat_target)):
            g_factor = float(grad_output[k]) if reduction == "none" else grad_val
            # d_loss / d_p = (p - y) / (p * (1 - p))
            denom = max(p * (1.0 - p), 1e-12)
            d_p = ((p - y) / denom) * scale * g_factor
            if flat_weight is not None:
                d_p *= flat_weight[k]
            grad_probs.append(d_p)

        return Tensor(grad_probs, shape=probs.shape, device=probs.device), None, None, None
