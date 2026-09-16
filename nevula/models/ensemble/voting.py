from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from nevula.core.tensor import Tensor
from nevula.autograd.engine import no_grad
from nevula.models.base import BaseModel
from nevula.models.ensemble.base import BaseEnsemble
from nevula.models.registry import register_model


@register_model(name="voting_regressor", category="ensemble")
class VotingRegressor(BaseEnsemble):
    """
    Prediction voting regressor for combining heterogeneous Nevula models.

    Computes a weighted or uniform average of predictions across multiple base estimators:
        y_hat(x) = sum(w_k * y_hat_k(x))

    Parameters:
        estimators: List of (name, estimator) tuples or list of estimator instances.
        weights: Sequence of weights (floats) corresponding to each estimator.
                 Normalized to sum to 1.0. If None, uniform weights are applied.
    """

    def __init__(
        self,
        estimators: Union[List[Tuple[str, BaseModel]], List[BaseModel]],
        weights: Optional[List[float]] = None,
    ):
        super().__init__()
        if not estimators:
            raise ValueError("VotingRegressor requires at least one estimator.")

        # Normalize estimators structure to list of (name, instance)
        self.named_estimators: List[Tuple[str, BaseModel]] = []
        for i, item in enumerate(estimators):
            if isinstance(item, tuple) and len(item) == 2:
                name, est = item
                self.named_estimators.append((str(name), est))
            elif isinstance(item, BaseModel):
                self.named_estimators.append((f"estimator_{i}", item))
            else:
                raise TypeError(f"Invalid estimator specification: {item}")

        self.estimators_ = [est for _, est in self.named_estimators]

        # Weights handling
        n_est = len(self.estimators_)
        if weights is not None:
            if len(weights) != n_est:
                raise ValueError(f"Number of weights ({len(weights)}) must match number of estimators ({n_est}).")
            if any(w < 0 for w in weights):
                raise ValueError("Weights must be non-negative.")
            total = sum(weights)
            if total <= 0:
                raise ValueError("Sum of weights must be positive.")
            self.weights_ = [float(w) / total for w in weights]
        else:
            self.weights_ = [1.0 / n_est] * n_est

    def fit(self, X: Any, y: Any, **kwargs: Any) -> "VotingRegressor":
        """
        Fits all underlying base estimators independently.

        Args:
            X: Training features.
            y: Training targets.

        Returns:
            self: The fitted ensemble.
        """
        self.train()
        for name, estimator in self.named_estimators:
            estimator.fit(X, y, **kwargs)

        self._is_fitted = True
        return self

    def forward(self, X: Any) -> Tensor:
        """
        Computes weighted average prediction across all base estimators.

        Args:
            X: Input features.

        Returns:
            Tensor: Ensembled predictions of shape (N, 1).
        """
        if not self._is_fitted:
            raise RuntimeError("VotingRegressor must be fitted before calling forward() or predict().")

        preds_list = []
        for est in self.estimators_:
            p = est.predict(X)
            # Flatten to 1D array
            p_arr = np.array(p.to_list()).reshape((-1, 1))
            preds_list.append(p_arr)

        # Weighted combination: sum(w_k * y_k)
        ensembled_pred = np.zeros_like(preds_list[0])
        for w, p in zip(self.weights_, preds_list):
            ensembled_pred += w * p

        return Tensor(ensembled_pred, shape=ensembled_pred.shape)

    def predict(self, X: Any) -> Tensor:
        """
        Predicts target values for inputs X under inference mode.

        Args:
            X: Input features.

        Returns:
            Tensor: Predicted targets.
        """
        self.eval()
        with no_grad():
            return self.forward(X)

    def get_config(self) -> Dict[str, Any]:
        """Returns model configuration for serialization."""
        return {
            "estimators": [(name, est.__class__.__name__) for name, est in self.named_estimators],
            "weights": self.weights_,
        }

    def __repr__(self) -> str:
        est_names = [name for name, _ in self.named_estimators]
        return f"VotingRegressor(estimators={est_names}, weights={[round(w, 3) for w in self.weights_]})"


# Register alias
register_model(VotingRegressor, name="voting", category="ensemble")

__all__ = ["VotingRegressor"]
