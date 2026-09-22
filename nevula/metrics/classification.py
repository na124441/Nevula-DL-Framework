"""
Classification evaluation metrics for Nevula.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from nevula.core.tensor import Tensor


def _to_numpy_1d(data: Any) -> np.ndarray:
    if isinstance(data, Tensor):
        arr = np.array(data.to_list(), dtype=np.float64)
    elif isinstance(data, np.ndarray):
        arr = data.astype(np.float64)
    else:
        arr = np.array(data, dtype=np.float64)
    return arr.ravel()


def accuracy_score(y_true: Any, y_pred: Any) -> float:
    """
    Accuracy classification score.

    Computes the fraction of correctly predicted class labels:
        Accuracy = (1 / N) * sum(I(y_true_i == y_pred_i))
    """
    yt = _to_numpy_1d(y_true)
    yp = _to_numpy_1d(y_pred)
    if len(yt) != len(yp):
        raise ValueError(f"Length mismatch: y_true has {len(yt)}, y_pred has {len(yp)}")
    if len(yt) == 0:
        return 0.0
    return float(np.mean(yt == yp))


def confusion_matrix(y_true: Any, y_pred: Any, labels: Optional[List[int]] = None) -> np.ndarray:
    """
    Computes confusion matrix to evaluate the accuracy of a classification.
    Returns matrix C where C[i, j] is the number of observations known to be
    in group i and predicted to be in group j.
    """
    yt = _to_numpy_1d(y_true).astype(int)
    yp = _to_numpy_1d(y_pred).astype(int)
    if len(yt) != len(yp):
        raise ValueError(f"Length mismatch: y_true has {len(yt)}, y_pred has {len(yp)}")

    if labels is None:
        unique_labels = np.unique(np.concatenate([yt, yp]))
        unique_labels.sort()
    else:
        unique_labels = np.array(labels, dtype=int)

    n_labels = len(unique_labels)
    label_to_idx = {val: idx for idx, val in enumerate(unique_labels)}
    cm = np.zeros((n_labels, n_labels), dtype=int)

    for true_val, pred_val in zip(yt, yp):
        if true_val in label_to_idx and pred_val in label_to_idx:
            i = label_to_idx[true_val]
            j = label_to_idx[pred_val]
            cm[i, j] += 1

    return cm


def precision_score(
    y_true: Any,
    y_pred: Any,
    average: str = "binary",
    pos_label: int = 1,
    zero_division: float = 0.0,
) -> float:
    """
    Computes precision metric: tp / (tp + fp).

    Args:
        y_true: Ground truth target labels.
        y_pred: Predicted labels.
        average: 'binary', 'macro', 'micro', or 'weighted'.
        pos_label: Positive class label when average='binary'.
        zero_division: Value to return when precision denominator (tp + fp) is 0.

    Returns:
        float: Computed precision score.
    """
    yt = _to_numpy_1d(y_true).astype(int)
    yp = _to_numpy_1d(y_pred).astype(int)
    if len(yt) != len(yp):
        raise ValueError(f"Length mismatch: y_true has {len(yt)}, y_pred has {len(yp)}")

    if average == "binary":
        tp = int(np.sum((yt == pos_label) & (yp == pos_label)))
        fp = int(np.sum((yt != pos_label) & (yp == pos_label)))
        denom = tp + fp
        return float(tp / denom) if denom > 0 else float(zero_division)

    classes = np.unique(np.concatenate([yt, yp]))
    if average == "micro":
        total_tp = int(np.sum([np.sum((yt == c) & (yp == c)) for c in classes]))
        total_fp = int(np.sum([np.sum((yt != c) & (yp == c)) for c in classes]))
        denom = total_tp + total_fp
        return float(total_tp / denom) if denom > 0 else float(zero_division)

    precisions = []
    weights = []
    for c in classes:
        tp = int(np.sum((yt == c) & (yp == c)))
        fp = int(np.sum((yt != c) & (yp == c)))
        denom = tp + fp
        prec = float(tp / denom) if denom > 0 else float(zero_division)
        precisions.append(prec)
        weights.append(int(np.sum(yt == c)))

    if average == "weighted":
        total_w = sum(weights)
        return float(np.sum(np.array(precisions) * np.array(weights)) / total_w) if total_w > 0 else float(zero_division)

    # Default 'macro'
    return float(np.mean(precisions)) if precisions else float(zero_division)


def recall_score(
    y_true: Any,
    y_pred: Any,
    average: str = "binary",
    pos_label: int = 1,
    zero_division: float = 0.0,
) -> float:
    """
    Computes recall metric: tp / (tp + fn).

    Args:
        y_true: Ground truth target labels.
        y_pred: Predicted labels.
        average: 'binary', 'macro', 'micro', or 'weighted'.
        pos_label: Positive class label when average='binary'.
        zero_division: Value to return when recall denominator (tp + fn) is 0.

    Returns:
        float: Computed recall score.
    """
    yt = _to_numpy_1d(y_true).astype(int)
    yp = _to_numpy_1d(y_pred).astype(int)
    if len(yt) != len(yp):
        raise ValueError(f"Length mismatch: y_true has {len(yt)}, y_pred has {len(yp)}")

    if average == "binary":
        tp = int(np.sum((yt == pos_label) & (yp == pos_label)))
        fn = int(np.sum((yt == pos_label) & (yp != pos_label)))
        denom = tp + fn
        return float(tp / denom) if denom > 0 else float(zero_division)

    classes = np.unique(np.concatenate([yt, yp]))
    if average == "micro":
        total_tp = int(np.sum([np.sum((yt == c) & (yp == c)) for c in classes]))
        total_fn = int(np.sum([np.sum((yt == c) & (yp != c)) for c in classes]))
        denom = total_tp + total_fn
        return float(total_tp / denom) if denom > 0 else float(zero_division)

    recalls = []
    weights = []
    for c in classes:
        tp = int(np.sum((yt == c) & (yp == c)))
        fn = int(np.sum((yt == c) & (yp != c)))
        denom = tp + fn
        rec = float(tp / denom) if denom > 0 else float(zero_division)
        recalls.append(rec)
        weights.append(int(np.sum(yt == c)))

    if average == "weighted":
        total_w = sum(weights)
        return float(np.sum(np.array(recalls) * np.array(weights)) / total_w) if total_w > 0 else float(zero_division)

    # Default 'macro'
    return float(np.mean(recalls)) if recalls else float(zero_division)


def f1_score(
    y_true: Any,
    y_pred: Any,
    average: str = "binary",
    pos_label: int = 1,
    zero_division: float = 0.0,
) -> float:
    """
    Computes F1 score: harmonic mean of precision and recall.
        F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true: Ground truth target labels.
        y_pred: Predicted labels.
        average: 'binary', 'macro', 'micro', or 'weighted'.
        pos_label: Positive class label when average='binary'.
        zero_division: Value to return when precision + recall is 0.

    Returns:
        float: Computed F1 score.
    """
    yt = _to_numpy_1d(y_true).astype(int)
    yp = _to_numpy_1d(y_pred).astype(int)
    if len(yt) != len(yp):
        raise ValueError(f"Length mismatch: y_true has {len(yt)}, y_pred has {len(yp)}")

    if average in ("binary", "micro"):
        p = precision_score(yt, yp, average=average, pos_label=pos_label, zero_division=zero_division)
        r = recall_score(yt, yp, average=average, pos_label=pos_label, zero_division=zero_division)
        denom = p + r
        return float(2.0 * p * r / denom) if denom > 0.0 else float(zero_division)

    classes = np.unique(np.concatenate([yt, yp]))
    f1_list = []
    weights = []
    for c in classes:
        p_c = precision_score(yt, yp, average="binary", pos_label=c, zero_division=zero_division)
        r_c = recall_score(yt, yp, average="binary", pos_label=c, zero_division=zero_division)
        denom = p_c + r_c
        f1_c = float(2.0 * p_c * r_c / denom) if denom > 0.0 else float(zero_division)
        f1_list.append(f1_c)
        weights.append(int(np.sum(yt == c)))

    if average == "weighted":
        total_w = sum(weights)
        return float(np.sum(np.array(f1_list) * np.array(weights)) / total_w) if total_w > 0 else float(zero_division)

    # Default 'macro'
    return float(np.mean(f1_list)) if f1_list else float(zero_division)


def bce_loss(y_true: Any, y_prob: Any, eps: float = 1e-12) -> float:
    """
    Computes Binary Cross Entropy loss given true binary labels and predicted probabilities.
    """
    yt = _to_numpy_1d(y_true)
    yp = np.clip(_to_numpy_1d(y_prob), eps, 1.0 - eps)
    loss = -(yt * np.log(yp) + (1.0 - yt) * np.log(1.0 - yp))
    return float(np.mean(loss))


def hinge_loss_score(y_true: Any, pred_decision: Any, margin: float = 1.0) -> float:
    """
    Computes average Hinge loss given ground-truth labels and raw decision values.
    Converts {0, 1} labels to {-1, +1} if necessary.
    """
    yt = _to_numpy_1d(y_true)
    yp = _to_numpy_1d(pred_decision)
    if set(np.unique(yt)).issubset({0, 1}):
        yt = np.where(yt == 1, 1.0, -1.0)
    return float(np.mean(np.maximum(0.0, margin - yt * yp)))


def log_loss_score(y_true: Any, y_prob: Any, eps: float = 1e-15) -> float:
    """
    Computes Log Loss / Cross-Entropy loss for binary or multiclass classification.

    Args:
        y_true: Ground truth labels (N,) or one-hot (N, K).
        y_prob: Predicted probabilities of shape (N, K) or (N,).
        eps: Small epsilon to avoid log(0).

    Returns:
        float: Computed log loss.
    """
    if isinstance(y_prob, Tensor):
        prob = np.array(y_prob.to_list(), dtype=np.float64)
    elif isinstance(y_prob, np.ndarray):
        prob = y_prob.astype(np.float64)
    else:
        prob = np.array(y_prob, dtype=np.float64)

    if isinstance(y_true, Tensor):
        yt = np.array(y_true.to_list(), dtype=np.float64)
    elif isinstance(y_true, np.ndarray):
        yt = y_true.astype(np.float64)
    else:
        yt = np.array(y_true, dtype=np.float64)

    # Binary case where prob is 1D or (N, 1)
    if prob.ndim == 1 or (prob.ndim == 2 and prob.shape[1] == 1):
        yt_1d = yt.ravel()
        if len(yt_1d) != len(prob.ravel()):
            raise ValueError(f"Shape mismatch: y_true has {len(yt_1d)} elements, y_prob has {len(prob.ravel())}.")
        p1 = np.clip(prob.ravel(), eps, 1.0 - eps)
        loss = -(yt_1d * np.log(p1) + (1.0 - yt_1d) * np.log(1.0 - p1))
        return float(np.mean(loss))

    # Multiclass case: prob is (N, K)
    if prob.ndim != 2:
        raise ValueError(f"y_prob must be 1D or 2D array, got shape {prob.shape}")

    N, K = prob.shape
    prob = np.clip(prob, eps, 1.0 - eps)
    # Normalize rows to sum to 1
    prob = prob / np.sum(prob, axis=1, keepdims=True)

    if yt.ndim == 2 and yt.shape == (N, K):
        yt_mat = yt
    else:
        yt_1d = yt.ravel()
        if len(yt_1d) != N:
            raise ValueError(f"Shape mismatch: y_true has {len(yt_1d)} samples, y_prob has {N} samples.")
        int_labels = yt_1d.astype(int)
        one_hot = np.zeros((N, K), dtype=np.float64)
        for i, lab in enumerate(int_labels):
            if 0 <= lab < K:
                one_hot[i, lab] = 1.0
        yt_mat = one_hot

    loss = -np.sum(yt_mat * np.log(prob)) / N
    return float(loss)


def auc(x: Any, y: Any) -> float:
    """
    Computes Area Under the Curve (AUC) using the trapezoidal rule.

    Args:
        x: x coordinates (e.g. False Positive Rates), monotonically increasing.
        y: y coordinates (e.g. True Positive Rates).

    Returns:
        float: Computed area under the curve.
    """
    x_arr = _to_numpy_1d(x)
    y_arr = _to_numpy_1d(y)
    if len(x_arr) != len(y_arr):
        raise ValueError(f"x and y must have equal length, got {len(x_arr)} and {len(y_arr)}")
    if len(x_arr) < 2:
        return 0.0

    # Trapezoidal rule: sum((x[i+1] - x[i]) * (y[i+1] + y[i]) / 2)
    dx = np.diff(x_arr)
    # Area under each trapezoid segment
    area = np.sum(dx * (y_arr[:-1] + y_arr[1:]) / 2.0)
    return float(abs(area))


def roc_curve(
    y_true: Any,
    y_score: Any,
    pos_label: Optional[Union[int, float]] = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Receiver Operating Characteristic (ROC) curve coordinates.

    Args:
        y_true: True binary labels in {0, 1} or {-1, 1}.
        y_score: Target confidence scores, continuous probabilities, or margin decisions.
        pos_label: Label of the positive class (default: 1).

    Returns:
        Tuple of (fpr, tpr, thresholds):
            fpr: Increasing false positive rates array.
            tpr: Increasing true positive rates array.
            thresholds: Decreasing decision thresholds on y_score.
    """
    y_t = _to_numpy_1d(y_true)
    y_s = _to_numpy_1d(y_score)
    if len(y_t) != len(y_s):
        raise ValueError(f"Length mismatch: y_true has {len(y_t)}, y_score has {len(y_s)}")

    # Convert binary targets to boolean positive mask
    if pos_label is None:
        unique = np.unique(y_t)
        pos_label = unique[-1]

    y_bool = (y_t == pos_label)
    n_pos = int(np.sum(y_bool))
    n_neg = len(y_t) - n_pos

    if n_pos == 0:
        raise ValueError("y_true contains no positive samples.")
    if n_neg == 0:
        raise ValueError("y_true contains no negative samples.")

    # Sort scores descending
    desc_order = np.argsort(-y_s, kind="mergesort")
    y_s_sorted = y_s[desc_order]
    y_bool_sorted = y_bool[desc_order]

    # Find unique threshold cutoffs where score changes
    distinct_mask = np.empty(len(y_s_sorted), dtype=bool)
    distinct_mask[:-1] = y_s_sorted[:-1] != y_s_sorted[1:]
    distinct_mask[-1] = True
    distinct_indices = np.where(distinct_mask)[0]

    # Cumulative true positives and false positives
    tps = np.cumsum(y_bool_sorted, dtype=np.float64)[distinct_indices]
    fps = (1 + distinct_indices).astype(np.float64) - tps

    # Prepend starting point (0, 0) with threshold = max_score + 1
    tpr = np.r_[0.0, tps / n_pos]
    fpr = np.r_[0.0, fps / n_neg]
    thresholds = np.r_[y_s_sorted[0] + 1.0, y_s_sorted[distinct_indices]]

    return fpr, tpr, thresholds


def roc_auc_score(
    y_true: Any,
    y_score: Any,
    average: str = "macro",
    multi_class: str = "ovr",
) -> float:
    """
    Computes Area Under the Receiver Operating Characteristic Curve (ROC AUC).

    Supports both binary classification and multi-class classification (One-vs-Rest / OvR).

    Args:
        y_true: True class labels of shape (N,) or binary indicators.
        y_score: Target scores or probabilities.
                 - For binary: 1D array of positive class probabilities / scores.
                 - For multiclass: 2D array of shape (N, K) of class probabilities.
        average: 'macro' or 'weighted' for multiclass.
        multi_class: 'ovr' (One-vs-Rest, default).

    Returns:
        float: ROC AUC score in [0.0, 1.0].
    """
    if isinstance(y_score, Tensor):
        score_arr = np.array(y_score.to_list(), dtype=np.float64)
    elif isinstance(y_score, np.ndarray):
        score_arr = y_score.astype(np.float64)
    else:
        score_arr = np.array(y_score, dtype=np.float64)

    y_t = _to_numpy_1d(y_true)

    # Detect binary classification: 1D score array or 2D with 2 columns
    is_binary = score_arr.ndim == 1 or (score_arr.ndim == 2 and score_arr.shape[1] <= 2 and len(np.unique(y_t)) <= 2)

    if is_binary:
        if score_arr.ndim == 2 and score_arr.shape[1] == 2:
            scores = score_arr[:, 1]
        else:
            scores = score_arr.ravel()
        fpr, tpr, _ = roc_curve(y_t, scores)
        return float(auc(fpr, tpr))

    # Multiclass One-vs-Rest (OvR)
    if score_arr.ndim != 2:
        raise ValueError(f"For multiclass ROC-AUC, y_score must be 2D array (N, n_classes), got {score_arr.shape}")

    classes = np.unique(y_t)
    n_classes = len(classes)
    if score_arr.shape[1] != n_classes:
        # Check if classes are 0..K-1
        if score_arr.shape[1] > n_classes:
            classes = np.arange(score_arr.shape[1])
            n_classes = len(classes)
        else:
            raise ValueError(f"Class count mismatch: y_true has {n_classes} classes, y_score has {score_arr.shape[1]} columns")

    aucs = []
    weights = []
    for idx, c in enumerate(classes):
        y_binary = (y_t == c).astype(int)
        if len(np.unique(y_binary)) < 2:
            # Class not present in ground truth or only class present
            continue
        c_scores = score_arr[:, idx]
        fpr, tpr, _ = roc_curve(y_binary, c_scores, pos_label=1)
        auc_c = float(auc(fpr, tpr))
        aucs.append(auc_c)
        weights.append(int(np.sum(y_binary == 1)))

    if not aucs:
        return 0.0

    if average == "weighted":
        total_w = sum(weights)
        return float(np.sum(np.array(aucs) * np.array(weights)) / total_w) if total_w > 0 else 0.0

    # Default 'macro'
    return float(np.mean(aucs))


# Aliases
accuracy = accuracy_score
precision = precision_score
recall = recall_score
f1 = f1_score
roc_auc = roc_auc_score
log_loss = log_loss_score

__all__ = [
    "accuracy_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "confusion_matrix",
    "bce_loss",
    "hinge_loss_score",
    "log_loss_score",
    "auc",
    "roc_curve",
    "roc_auc_score",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "log_loss",
]

