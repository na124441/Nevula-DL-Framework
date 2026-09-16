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


def precision_score(y_true: Any, y_pred: Any, average: str = "binary", pos_label: int = 1) -> float:
    """
    Computes precision: tp / (tp + fp).
    """
    yt = _to_numpy_1d(y_true).astype(int)
    yp = _to_numpy_1d(y_pred).astype(int)

    if average == "binary":
        tp = int(np.sum((yt == pos_label) & (yp == pos_label)))
        fp = int(np.sum((yt != pos_label) & (yp == pos_label)))
        denom = tp + fp
        return float(tp / denom) if denom > 0 else 0.0

    classes = np.unique(np.concatenate([yt, yp]))
    precisions = []
    for c in classes:
        tp = int(np.sum((yt == c) & (yp == c)))
        fp = int(np.sum((yt != c) & (yp == c)))
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        precisions.append(prec)

    return float(np.mean(precisions)) if precisions else 0.0


def recall_score(y_true: Any, y_pred: Any, average: str = "binary", pos_label: int = 1) -> float:
    """
    Computes recall: tp / (tp + fn).
    """
    yt = _to_numpy_1d(y_true).astype(int)
    yp = _to_numpy_1d(y_pred).astype(int)

    if average == "binary":
        tp = int(np.sum((yt == pos_label) & (yp == pos_label)))
        fn = int(np.sum((yt == pos_label) & (yp != pos_label)))
        denom = tp + fn
        return float(tp / denom) if denom > 0 else 0.0

    classes = np.unique(np.concatenate([yt, yp]))
    recalls = []
    for c in classes:
        tp = int(np.sum((yt == c) & (yp == c)))
        fn = int(np.sum((yt == c) & (yp != c)))
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        recalls.append(rec)

    return float(np.mean(recalls)) if recalls else 0.0


def f1_score(y_true: Any, y_pred: Any, average: str = "binary", pos_label: int = 1) -> float:
    """
    Computes F1 score: harmonic mean of precision and recall.
        F1 = 2 * (precision * recall) / (precision + recall)
    """
    p = precision_score(y_true, y_pred, average=average, pos_label=pos_label)
    r = recall_score(y_true, y_pred, average=average, pos_label=pos_label)
    denom = p + r
    return float(2.0 * p * r / denom) if denom > 0.0 else 0.0


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


__all__ = [
    "accuracy_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "confusion_matrix",
    "bce_loss",
    "hinge_loss_score",
]
