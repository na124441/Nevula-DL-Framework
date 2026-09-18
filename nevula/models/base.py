from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Sequence, Union
import numpy as np

from nevula.core.tensor import Tensor
from nevula.nn.module import Module
from nevula.metrics.regression import mean_squared_error, mean_absolute_error, r2_score


class BaseModel(Module, ABC):
    """
    Abstract base class for all machine learning and deep learning models in Nevula.

    Extends `nevula.nn.Module` to inherit parameter registration, state_dict
    serialization, gradient management, and device tracking, while standardizing
    the high-level model lifecycle:

        model = Model(...)
        model.fit(X, y)
        predictions = model.predict(X_test)
        score = model.evaluate(X_test, y_test, metric="mse")

    Subclasses must implement:
        - forward(X)
        - fit(X, y, **kwargs)
        - predict(X)
        - get_config()
    """

    def __init__(self):
        super().__init__()
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """Returns whether the model has been fitted."""
        return self._is_fitted

    @abstractmethod
    def forward(self, X: Tensor) -> Tensor:
        """
        Defines the mathematical forward transformation of input tensor X.

        Args:
            X: Input Tensor of shape (batch_size, in_features).

        Returns:
            Tensor: Output predictions or activations.
        """
        raise NotImplementedError("Every model must implement forward()")

    @abstractmethod
    def fit(self, X: Any, y: Any, **kwargs: Any) -> "BaseModel":
        """
        Trains the model on dataset (X, y).

        Args:
            X: Features as Tensor, ndarray, or sequence.
            y: Targets as Tensor, ndarray, or sequence.
            **kwargs: Hyperparameters or training options.

        Returns:
            self: The fitted model instance.
        """
        raise NotImplementedError("Every model must implement fit()")

    @abstractmethod
    def predict(self, X: Any) -> Tensor:
        """
        Generates predictions for inputs X.

        Args:
            X: Features as Tensor, ndarray, or sequence.

        Returns:
            Tensor: Predicted targets.
        """
        raise NotImplementedError("Every model must implement predict()")

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Returns a dictionary of model configuration and hyperparameters.
        """
        raise NotImplementedError("Every model must implement get_config()")

    def evaluate(
        self,
        X: Any,
        y: Any,
        metric: Union[str, Callable[[Any, Any], float]] = "mse",
    ) -> float:
        """
        Evaluates model predictions against ground truth targets.

        Args:
            X: Features.
            y: Ground truth targets.
            metric: Metric name ('mse', 'mae', 'r2') or custom callable(y_true, y_pred).

        Returns:
            float: Evaluation score.
        """
        preds = self.predict(X)

        if callable(metric):
            return float(metric(y, preds))

        metric_normalized = metric.lower().strip()
        if metric_normalized in ("mse", "mean_squared_error"):
            return mean_squared_error(y, preds)
        elif metric_normalized in ("mae", "mean_absolute_error"):
            return mean_absolute_error(y, preds)
        elif metric_normalized in ("r2", "r2_score"):
            return r2_score(y, preds)
        else:
            raise ValueError(f"Unknown metric '{metric}'. Supported metrics: 'mse', 'mae', 'r2', or callable.")

    def _to_tensor(self, data: Any, requires_grad: bool = False) -> Tensor:
        """
        Internal utility ensuring input data is converted into a Nevula Tensor.

        Args:
            data: Tensor, numpy array, list, or scalar.
            requires_grad: Whether the resulting tensor requires gradient.

        Returns:
            Tensor: Formatted Nevula Tensor.
        """
        if isinstance(data, Tensor):
            return data

        if isinstance(data, np.ndarray):
            return Tensor(data, requires_grad=requires_grad)

        if isinstance(data, (list, tuple, int, float)):
            arr = np.array(data, dtype=np.float64)
            return Tensor(arr, requires_grad=requires_grad)

        raise TypeError(f"Cannot convert type {type(data)} to Nevula Tensor.")

    def save(self, filepath: str) -> None:
        """
        Serializes model parameters and configuration to disk.

        Args:
            filepath: Destination file path.
        """
        from nevula.utils.serialization import save_checkpoint

        checkpoint = {
            "model_class": self.__class__.__name__,
            "config": self.get_config(),
            "state_dict": self.state_dict(),
            "is_fitted": self._is_fitted,
        }
        save_checkpoint(filepath, self, extra_checkpoint=checkpoint)

    def load(self, filepath: str) -> "BaseModel":
        """
        Loads model parameters and configuration from disk.

        Args:
            filepath: Source checkpoint path.

        Returns:
            self: Restored model instance.
        """
        from nevula.utils.serialization import load_checkpoint

        checkpoint = load_checkpoint(filepath, model=self)
        if "extra_checkpoint" in checkpoint and isinstance(checkpoint["extra_checkpoint"], dict):
            extra = checkpoint["extra_checkpoint"]
            if "state_dict" in extra:
                self.load_state_dict(extra["state_dict"])
            if "is_fitted" in extra:
                self._is_fitted = extra["is_fitted"]
        elif "state_dict" in checkpoint:
            self.load_state_dict(checkpoint["state_dict"])
        if "is_fitted" in checkpoint:
            self._is_fitted = checkpoint["is_fitted"]
        return self


__all__ = ["BaseModel"]
