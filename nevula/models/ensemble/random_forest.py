from typing import Any, Dict, List, Optional, Union
import numpy as np

from nevula.core.tensor import Tensor
from nevula.autograd.engine import no_grad
from nevula.models.trees.decision_tree import DecisionTreeRegressor
from nevula.models.ensemble.base import BaseEnsemble
from nevula.models.registry import register_model
from nevula.metrics.regression import r2_score


@register_model(name="random_forest_regressor", category="ensemble")
class RandomForestRegressor(BaseEnsemble):
    """
    Random Forest Regressor (Bagging + Random Feature Subspaces).

    Fits an ensemble of randomized DecisionTreeRegressor estimators on bootstrap
    subsamples of the dataset. Aggregation drastically reduces model variance
    without increasing bias, overcoming the high variance of individual decision trees.

    Parameters:
        n_estimators: The number of trees in the forest (default: 50).
        max_depth: Maximum depth of each tree (default: 8).
        min_samples_split: Minimum number of samples required to split an internal node.
        min_samples_leaf: Minimum number of samples required to be at a leaf node.
        max_features: The number of features to consider when looking for the best split:
                      - 'sqrt': int(sqrt(n_features))
                      - 'log2': int(log2(n_features))
                      - float: fraction of features
                      - int: exact feature count
                      - None: all features
        bootstrap: Whether bootstrap samples are used when building trees (default: True).
        oob_score: Whether to use out-of-bag samples to estimate generalization R^2 score.
        random_state: Controls both the randomness of the bootstrapping and feature sampling.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: Optional[int] = 8,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[Union[int, float, str]] = "sqrt",
        bootstrap: bool = True,
        oob_score: bool = False,
        random_state: Optional[int] = None,
    ):
        super().__init__()
        if n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1, got {n_estimators}")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state

        self.estimators_: List[DecisionTreeRegressor] = []
        self.feature_importances_: Optional[np.ndarray] = None
        self.oob_score_: Optional[float] = None

    def _resolve_max_features(self, n_features: int) -> Optional[int]:
        if self.max_features is None:
            return None
        if isinstance(self.max_features, int):
            return max(1, min(self.max_features, n_features))
        if isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if self.max_features == "log2":
            return max(1, int(np.log2(n_features)))
        return None

    def fit(self, X: Any, y: Any, **kwargs: Any) -> "RandomForestRegressor":
        """
        Builds a forest of trees from the training set (X, y).

        Args:
            X: Training features.
            y: Target values.

        Returns:
            self: The fitted random forest.
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
        tree_max_features = self._resolve_max_features(n_features)

        rng = np.random.RandomState(self.random_state)
        self.estimators_ = []
        all_importances = []

        # Out-of-Bag tracking: dictionary mapping sample_idx -> list of predictions
        oob_preds: Dict[int, List[float]] = {i: [] for i in range(n_samples)}

        for i in range(self.n_estimators):
            if self.bootstrap:
                boot_indices = rng.choice(n_samples, size=n_samples, replace=True)
                X_boot = X_np[boot_indices]
                y_boot = y_np[boot_indices]

                if self.oob_score:
                    oob_indices = set(range(n_samples)) - set(boot_indices)
            else:
                X_boot, y_boot = X_np, y_np
                oob_indices = set()

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=tree_max_features,
            )
            tree.fit(X_boot, y_boot)
            self.estimators_.append(tree)

            if tree.feature_importances_ is not None:
                all_importances.append(tree.feature_importances_)

            # Record OOB predictions
            if self.bootstrap and self.oob_score and oob_indices:
                oob_idx_list = sorted(list(oob_indices))
                tree_oob_preds = tree.predict(X_np[oob_idx_list]).to_list()
                for idx, p in zip(oob_idx_list, tree_oob_preds):
                    val = float(p[0]) if isinstance(p, (list, tuple)) else float(p)
                    oob_preds[idx].append(val)

        # Compute aggregate feature importances
        if all_importances:
            mean_imp = np.mean(all_importances, axis=0)
            self.feature_importances_ = mean_imp / (np.sum(mean_imp) + 1e-12)
        else:
            self.feature_importances_ = np.zeros(n_features)

        # Calculate OOB R^2 score if enabled
        if self.bootstrap and self.oob_score:
            oob_y_true = []
            oob_y_pred = []
            for idx in range(n_samples):
                if oob_preds[idx]:
                    oob_y_true.append(y_np[idx])
                    oob_y_pred.append(np.mean(oob_preds[idx]))

            if len(oob_y_true) > 0:
                self.oob_score_ = r2_score(np.array(oob_y_true), np.array(oob_y_pred))

        self._is_fitted = True
        return self

    def forward(self, X: Any) -> Tensor:
        """
        Computes aggregate prediction across all trees in the forest.

        Args:
            X: Input features.

        Returns:
            Tensor: Predicted targets of shape (N, 1).
        """
        if not self._is_fitted or not self.estimators_:
            raise RuntimeError("RandomForestRegressor must be fitted before calling forward() or predict().")

        # Collect predictions from each tree
        preds_all = []
        for tree in self.estimators_:
            p = tree.predict(X)
            preds_all.append(np.array(p.to_list()).reshape((-1, 1)))

        # Ensemble mean: (1 / B) * sum(h_b(x))
        forest_pred = np.mean(preds_all, axis=0)
        return Tensor(forest_pred, shape=forest_pred.shape)

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
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "bootstrap": self.bootstrap,
            "oob_score": self.oob_score,
            "random_state": self.random_state,
        }

    def __repr__(self) -> str:
        return (
            f"RandomForestRegressor(n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth}, "
            f"max_features={self.max_features})"
        )


# Register alias
register_model(RandomForestRegressor, name="random_forest", category="ensemble")

__all__ = ["RandomForestRegressor"]
