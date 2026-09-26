"""
Utility functions for model selection, data indexing, estimator cloning, and scoring.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import copy
import inspect
import numpy as np

from nevula.core.tensor import Tensor
from nevula.data.dataset import Dataset, Subset
from nevula.metrics.regression import (
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
)
from nevula.metrics.classification import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss_score,
)


def to_numpy(data: Any) -> np.ndarray:
    """
    Converts Tensor, NumPy array, Python sequence, or scalar into a NumPy array.
    """
    if isinstance(data, Tensor):
        return data.numpy()
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, (list, tuple)):
        return np.array(data)
    elif hasattr(data, "to_numpy"):
        return data.to_numpy()
    else:
        return np.asarray(data)


def safe_indexing(X: Any, indices: Any) -> Any:
    """
    Safely indexes X by an array of integer indices, boolean mask, or slice.
    Supports Nevula Tensor, NumPy ndarray, Python lists/tuples, Datasets, and DataFrames.
    """
    if X is None:
        return None

    if isinstance(X, Tensor):
        arr = X.numpy()
        idx = indices.numpy().astype(int) if isinstance(indices, Tensor) else indices
        sliced_arr = arr[idx]
        return Tensor(sliced_arr, device=X.device, requires_grad=X.requires_grad)

    if isinstance(X, np.ndarray):
        idx = indices.numpy().astype(int) if isinstance(indices, Tensor) else indices
        return X[idx]

    if isinstance(X, (list, tuple)):
        idx = indices.tolist() if hasattr(indices, "tolist") else list(indices)
        selected = [X[i] for i in idx]
        return selected if isinstance(X, list) else tuple(selected)

    if isinstance(X, Dataset):
        idx = indices.tolist() if hasattr(indices, "tolist") else list(indices)
        return Subset(X, idx)

    if hasattr(X, "iloc"):
        idx = indices.tolist() if hasattr(indices, "tolist") else list(indices)
        return X.iloc[idx]

    idx = indices.tolist() if hasattr(indices, "tolist") else list(indices)
    return [X[i] for i in idx]


def is_classifier(estimator: Any) -> bool:
    """
    Determines whether an estimator is a classifier.
    """
    if estimator is None:
        return False

    if getattr(estimator, "_estimator_type", None) == "classifier":
        return True

    cls_name = estimator.__class__.__name__.lower()
    if "classifier" in cls_name or "svc" in cls_name or "logistic" in cls_name:
        return True

    try:
        from nevula.models.registry import _REGISTRY, _normalize_name
        canon = _normalize_name(estimator.__class__.__name__)
        if canon in _REGISTRY and _REGISTRY[canon]["category"] == "classification":
            return True
    except Exception:
        pass

    return False


def clone_estimator(estimator: Any) -> Any:
    """
    Creates an unfitted clone of an estimator with identical configuration / hyperparameters.
    """
    if estimator is None:
        return None

    # Check if estimator has get_config()
    if hasattr(estimator, "get_config"):
        try:
            config = copy.deepcopy(estimator.get_config())
            new_est = estimator.__class__(**config)
            return new_est
        except Exception:
            pass

    # Deepcopy fallback
    cloned = copy.deepcopy(estimator)
    if hasattr(cloned, "_is_fitted"):
        cloned._is_fitted = False
    if hasattr(cloned, "loss_history_"):
        cloned.loss_history_ = []
    return cloned


def get_scorer(
    scoring: Optional[Union[str, Callable[[Any, Any, Any], float], Callable[[Any, Any], float]]] = None,
    estimator: Any = None,
) -> Callable[[Any, Any, Any], float]:
    """
    Returns a callable scorer with signature:
        scorer(model, X, y) -> float
    """
    if callable(scoring):
        sig = inspect.signature(scoring)
        num_params = len(sig.parameters)
        if num_params == 3:
            # scoring(estimator, X, y)
            return scoring
        elif num_params == 2:
            # scoring(y_true, y_pred)
            def _metric_scorer(est: Any, X_val: Any, y_val: Any) -> float:
                preds = est.predict(X_val)
                return float(scoring(y_val, preds))
            return _metric_scorer
        else:
            def _generic_scorer(est: Any, X_val: Any, y_val: Any) -> float:
                return float(scoring(est, X_val, y_val))
            return _generic_scorer

    if scoring is None:
        if is_classifier(estimator):
            scoring_str = "accuracy"
        else:
            scoring_str = "r2"
    elif isinstance(scoring, str):
        scoring_str = scoring.lower().strip()
    else:
        raise ValueError(f"Invalid scoring specification: {scoring}")

    # Classification Metrics
    if scoring_str in ("accuracy", "acc", "accuracy_score"):
        def _acc_scorer(est: Any, X_val: Any, y_val: Any) -> float:
            preds = est.predict(X_val)
            return accuracy_score(y_val, preds)
        return _acc_scorer

    if scoring_str in ("f1", "f1_score", "f1_macro"):
        def _f1_scorer(est: Any, X_val: Any, y_val: Any) -> float:
            preds = est.predict(X_val)
            y_arr = to_numpy(y_val).ravel()
            n_classes = len(np.unique(y_arr))
            avg = "binary" if n_classes <= 2 else "macro"
            return f1_score(y_val, preds, average=avg)
        return _f1_scorer

    if scoring_str == "f1_weighted":
        def _f1w_scorer(est: Any, X_val: Any, y_val: Any) -> float:
            preds = est.predict(X_val)
            return f1_score(y_val, preds, average="weighted")
        return _f1w_scorer

    if scoring_str == "f1_micro":
        def _f1m_scorer(est: Any, X_val: Any, y_val: Any) -> float:
            preds = est.predict(X_val)
            return f1_score(y_val, preds, average="micro")
        return _f1m_scorer

    if scoring_str in ("precision", "precision_score", "precision_macro"):
        def _prec_scorer(est: Any, X_val: Any, y_val: Any) -> float:
            preds = est.predict(X_val)
            y_arr = to_numpy(y_val).ravel()
            n_classes = len(np.unique(y_arr))
            avg = "binary" if n_classes <= 2 else "macro"
            return precision_score(y_val, preds, average=avg)
        return _prec_scorer

    if scoring_str in ("recall", "recall_score", "recall_macro"):
        def _rec_scorer(est: Any, X_val: Any, y_val: Any) -> float:
            preds = est.predict(X_val)
            y_arr = to_numpy(y_val).ravel()
            n_classes = len(np.unique(y_arr))
            avg = "binary" if n_classes <= 2 else "macro"
            return recall_score(y_val, preds, average=avg)
        return _rec_scorer

    if scoring_str in ("roc_auc", "roc_auc_score"):
        def _auc_scorer(est: Any, X_val: Any, y_val: Any) -> float:
            if hasattr(est, "predict_proba"):
                scores = est.predict_proba(X_val)
                # If binary and shape (N, 2), take positive class probability
                scores_np = to_numpy(scores)
                if scores_np.ndim == 2 and scores_np.shape[1] == 2:
                    scores_np = scores_np[:, 1]
            elif hasattr(est, "decision_function"):
                scores_np = to_numpy(est.decision_function(X_val))
            else:
                scores_np = to_numpy(est.predict(X_val))
            return roc_auc_score(y_val, scores_np)
        return _auc_scorer

    if scoring_str in ("log_loss", "neg_log_loss", "bce"):
        def _log_loss_scorer(est: Any, X_val: Any, y_val: Any) -> float:
            if hasattr(est, "predict_proba"):
                probs = est.predict_proba(X_val)
            else:
                probs = est.predict(X_val)
            val = log_loss_score(y_val, probs)
            return -val if "neg" in scoring_str else val
        return _log_loss_scorer

    # Regression Metrics
    if scoring_str in ("r2", "r_squared", "r2_score"):
        def _r2_scorer(est: Any, X_val: Any, y_val: Any) -> float:
            preds = est.predict(X_val)
            return r2_score(y_val, preds)
        return _r2_scorer

    if scoring_str in ("mean_squared_error", "mse"):
        def _mse_scorer(est: Any, X_val: Any, y_val: Any) -> float:
            preds = est.predict(X_val)
            return mean_squared_error(y_val, preds)
        return _mse_scorer

    if scoring_str in ("neg_mean_squared_error", "neg_mse"):
        def _neg_mse_scorer(est: Any, X_val: Any, y_val: Any) -> float:
            preds = est.predict(X_val)
            return -mean_squared_error(y_val, preds)
        return _neg_mse_scorer

    if scoring_str in ("mean_absolute_error", "mae"):
        def _mae_scorer(est: Any, X_val: Any, y_val: Any) -> float:
            preds = est.predict(X_val)
            return mean_absolute_error(y_val, preds)
        return _mae_scorer

    if scoring_str in ("neg_mean_absolute_error", "neg_mae"):
        def _neg_mae_scorer(est: Any, X_val: Any, y_val: Any) -> float:
            preds = est.predict(X_val)
            return -mean_absolute_error(y_val, preds)
        return _neg_mae_scorer

    if scoring_str in ("root_mean_squared_error", "rmse"):
        def _rmse_scorer(est: Any, X_val: Any, y_val: Any) -> float:
            preds = est.predict(X_val)
            return root_mean_squared_error(y_val, preds)
        return _rmse_scorer

    if scoring_str in ("neg_root_mean_squared_error", "neg_rmse"):
        def _neg_rmse_scorer(est: Any, X_val: Any, y_val: Any) -> float:
            preds = est.predict(X_val)
            return -root_mean_squared_error(y_val, preds)
        return _neg_rmse_scorer

    # Fallback to model evaluate
    if hasattr(estimator, "evaluate"):
        def _evaluate_scorer(est: Any, X_val: Any, y_val: Any) -> float:
            return float(est.evaluate(X_val, y_val, metric=scoring_str))
        return _evaluate_scorer

    raise ValueError(
        f"Unknown scoring metric '{scoring_str}'. Supported options: "
        "'accuracy', 'f1', 'f1_macro', 'f1_weighted', 'precision', 'recall', "
        "'roc_auc', 'log_loss', 'neg_log_loss', 'r2', 'mse', 'neg_mean_squared_error', "
        "'mae', 'neg_mean_absolute_error', 'rmse', 'neg_root_mean_squared_error'."
    )
