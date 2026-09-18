from typing import Any, Union
import numpy as np
from nevula.core.tensor import Tensor


def _to_numpy(data: Any) -> np.ndarray:
    """Helper to convert Tensor, array-like, or scalar to numpy float64 ndarray."""
    if isinstance(data, Tensor):
        return np.array(data.to_list(), dtype=np.float64)
    if isinstance(data, np.ndarray):
        return data.astype(np.float64)
    return np.array(data, dtype=np.float64)


def mean_squared_error(y_true: Union[Tensor, Any], y_pred: Union[Tensor, Any]) -> float:
    """
    Computes the Mean Squared Error (MSE) between ground truth and predictions.

    Args:
        y_true: Ground truth target values (Tensor, ndarray, or sequence).
        y_pred: Predicted target values.

    Returns:
        float: Computed MSE score.
    """
    yt = _to_numpy(y_true).ravel()
    yp = _to_numpy(y_pred).ravel()
    if yt.size == 0 or yp.size == 0:
        raise ValueError("Cannot calculate MSE on empty arrays.")
    if yt.size != yp.size:
        raise ValueError(f"Shape mismatch: y_true has {yt.size} elements, y_pred has {yp.size}.")
    return float(np.mean((yt - yp) ** 2))


def mean_absolute_error(y_true: Union[Tensor, Any], y_pred: Union[Tensor, Any]) -> float:
    """
    Computes the Mean Absolute Error (MAE) between ground truth and predictions.

    Args:
        y_true: Ground truth target values (Tensor, ndarray, or sequence).
        y_pred: Predicted target values.

    Returns:
        float: Computed MAE score.
    """
    yt = _to_numpy(y_true).ravel()
    yp = _to_numpy(y_pred).ravel()
    if yt.size == 0 or yp.size == 0:
        raise ValueError("Cannot calculate MAE on empty arrays.")
    if yt.size != yp.size:
        raise ValueError(f"Shape mismatch: y_true has {yt.size} elements, y_pred has {yp.size}.")
    return float(np.mean(np.abs(yt - yp)))


def r2_score(y_true: Union[Tensor, Any], y_pred: Union[Tensor, Any]) -> float:
    """
    Computes the Coefficient of Determination (R^2 score).

    Args:
        y_true: Ground truth target values (Tensor, ndarray, or sequence).
        y_pred: Predicted target values.

    Returns:
        float: Computed R^2 score.
    """
    yt = _to_numpy(y_true).ravel()
    yp = _to_numpy(y_pred).ravel()
    if yt.size == 0 or yp.size == 0:
        raise ValueError("Cannot calculate R^2 score on empty arrays.")
    if yt.size != yp.size:
        raise ValueError(f"Shape mismatch: y_true has {yt.size} elements, y_pred has {yp.size}.")

    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)

    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0

    return float(1.0 - (ss_res / ss_tot))


def root_mean_squared_error(y_true: Union[Tensor, Any], y_pred: Union[Tensor, Any]) -> float:
    """
    Computes the Root Mean Squared Error (RMSE) between ground truth and predictions.

    Args:
        y_true: Ground truth target values (Tensor, ndarray, or sequence).
        y_pred: Predicted target values.

    Returns:
        float: Computed RMSE score.
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# Aliases
mse_score = mean_squared_error
mae_score = mean_absolute_error
rmse_score = root_mean_squared_error
r_squared_score = r2_score

__all__ = [
    "mean_squared_error",
    "mean_absolute_error",
    "root_mean_squared_error",
    "r2_score",
    "mse_score",
    "mae_score",
    "rmse_score",
    "r_squared_score",
]
