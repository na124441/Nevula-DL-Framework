from typing import Any, Dict, List, Optional, Union
import numpy as np

from nevula.core.tensor import Tensor
from nevula.autograd.engine import no_grad
from nevula.models.trees.decision_tree import DecisionTreeRegressor
from nevula.models.ensemble.base import BaseEnsemble
from nevula.models.registry import register_model


@register_model(name="gradient_boosting_regressor", category="ensemble")
class GradientBoostingRegressor(BaseEnsemble):
    """
    Gradient Boosting Regressor for squared error loss (GBDT).

    Builds an additive model in a forward stage-wise fashion by sequentially fitting
    weak base estimators (shallow decision trees) to the negative gradient (pseudo-residuals)
    of the loss function:
        F_m(x) = F_{m-1}(x) + eta * h_m(x)

    Parameters:
        n_estimators: Number of boosting stages to perform (default: 50).
        learning_rate: Shrinkage parameter eta in (0, 1] scaling the contribution of each tree.
        max_depth: Maximum depth of individual regression trees (default: 3).
        subsample: Fraction of samples to be used for fitting the individual base learners.
                   If < 1.0, results in Stochastic Gradient Boosting.
        min_samples_split: Minimum number of samples required to split an internal node.
        min_samples_leaf: Minimum number of samples required to be at a leaf node.
        random_state: Controls the random seed for subsampling.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        subsample: float = 1.0,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: Optional[int] = None,
    ):
        super().__init__()
        if n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1, got {n_estimators}")
        if learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be > 0.0, got {learning_rate}")
        if not (0.0 < subsample <= 1.0):
            raise ValueError(f"subsample must be in (0.0, 1.0], got {subsample}")

        self.n_estimators = n_estimators
        self.learning_rate = float(learning_rate)
        self.max_depth = max_depth
        self.subsample = float(subsample)
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        self.estimators_: List[DecisionTreeRegressor] = []
        self.init_value_: float = 0.0
        self.train_score_: List[float] = []
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(self, X: Any, y: Any, **kwargs: Any) -> "GradientBoostingRegressor":
        """
        Fits the gradient boosting model on (X, y).

        Args:
            X: Training features.
            y: Target values.

        Returns:
            self: The fitted gradient boosting ensemble.
        """
        if isinstance(X, Tensor):
            X_np = np.array(X.to_list(), dtype=np.float64)
        elif isinstance(X, np.ndarray):
            X_np = X.astype(np.float64)
        else:
            X_np = np.array(X, dtype=np.float64)

        if isinstance(y, Tensor):
            y_np = np.array(y.to_list(), dtype=np.float64)
        elif isinstance(y, np.ndarray):
            y_np = y.astype(np.float64)
        else:
            y_np = np.array(y, dtype=np.float64)

        if X_np.ndim == 1:
            X_np = X_np.reshape((-1, 1))
        y_np = y_np.reshape((-1,))

        n_samples, n_features = X_np.shape
        self.n_features_in_ = n_features

        rng = np.random.RandomState(self.random_state)
        self.estimators_ = []
        self.train_score_ = []
        all_importances = []

        # Stage 0: Initial constant prediction (empirical mean)
        self.init_value_ = float(np.mean(y_np))
        current_pred = np.full(n_samples, self.init_value_, dtype=np.float64)

        for m in range(self.n_estimators):
            # Compute pseudo-residuals for squared error: r_i = y_i - F_{m-1}(x_i)
            residuals = y_np - current_pred

            # Subsampling (Stochastic Gradient Boosting)
            if self.subsample < 1.0:
                sub_size = max(1, int(self.subsample * n_samples))
                sub_idx = rng.choice(n_samples, size=sub_size, replace=False)
                X_sub, r_sub = X_np[sub_idx], residuals[sub_idx]
            else:
                X_sub, r_sub = X_np, residuals

            # Fit shallow regression tree to pseudo-residuals
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(X_sub, r_sub)
            self.estimators_.append(tree)

            if tree.feature_importances_ is not None:
                all_importances.append(tree.feature_importances_)

            # Update predictions with shrinkage: F_m(x) = F_{m-1}(x) + eta * h_m(x)
            tree_step = np.array(tree.predict(X_np).to_list()).ravel()
            current_pred += self.learning_rate * tree_step

            # Track training MSE
            mse = float(np.mean((y_np - current_pred) ** 2))
            self.train_score_.append(mse)

        # Compute aggregate feature importances
        if all_importances:
            mean_imp = np.mean(all_importances, axis=0)
            self.feature_importances_ = mean_imp / (np.sum(mean_imp) + 1e-12)
        else:
            self.feature_importances_ = np.zeros(n_features)

        self._is_fitted = True
        return self

    def forward(self, X: Any) -> Tensor:
        """
        Computes stage-wise additive prediction across all boosting trees.

        Args:
            X: Input features.

        Returns:
            Tensor: Predicted targets of shape (N, 1).
        """
        if not self._is_fitted:
            raise RuntimeError("GradientBoostingRegressor must be fitted before calling forward() or predict().")

        if isinstance(X, Tensor):
            X_np = np.array(X.to_list(), dtype=np.float64)
        elif isinstance(X, np.ndarray):
            X_np = X.astype(np.float64)
        else:
            X_np = np.array(X, dtype=np.float64)

        if X_np.ndim == 1:
            X_np = X_np.reshape((-1, 1))

        # Start with initial base prediction F_0(x)
        pred = np.full(X_np.shape[0], self.init_value_, dtype=np.float64)

        # Add scaled contributions from each tree: sum(eta * h_m(x))
        for tree in self.estimators_:
            tree_out = np.array(tree.predict(X_np).to_list()).ravel()
            pred += self.learning_rate * tree_out

        pred_2d = pred.reshape((-1, 1))
        return Tensor(pred_2d, shape=pred_2d.shape)

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

    def feature_importances(self) -> np.ndarray:
        """Returns ensemble feature importances."""
        if self.feature_importances_ is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.feature_importances_

    def get_config(self) -> Dict[str, Any]:
        """Returns model configuration for serialization."""
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "init_value": self.init_value_,
        }

    def __repr__(self) -> str:
        return (
            f"GradientBoostingRegressor(n_estimators={self.n_estimators}, "
            f"learning_rate={self.learning_rate}, "
            f"max_depth={self.max_depth})"
        )


# Register alias
register_model(GradientBoostingRegressor, name="gradient_boosting", category="ensemble")

__all__ = ["GradientBoostingRegressor"]
