from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np

from nevula.core.tensor import Tensor
from nevula.autograd.engine import no_grad
from nevula.models.base import BaseModel
from nevula.models.registry import register_model


@dataclass
class TreeNode:
    """Represents a single node in a decision tree."""
    feature_idx: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    value: Optional[float] = None
    mse: float = 0.0
    n_samples: int = 0
    is_leaf: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serializes node and child branches into a dictionary."""
        return {
            "feature_idx": self.feature_idx,
            "threshold": self.threshold,
            "value": self.value,
            "mse": self.mse,
            "n_samples": self.n_samples,
            "is_leaf": self.is_leaf,
            "left": self.left.to_dict() if self.left is not None else None,
            "right": self.right.to_dict() if self.right is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["TreeNode"]:
        """Reconstructs node and branches from a dictionary."""
        if data is None:
            return None
        node = cls(
            feature_idx=data.get("feature_idx"),
            threshold=data.get("threshold"),
            value=data.get("value"),
            mse=data.get("mse", 0.0),
            n_samples=data.get("n_samples", 0),
            is_leaf=data.get("is_leaf", False),
        )
        if data.get("left") is not None:
            node.left = cls.from_dict(data["left"])
        if data.get("right") is not None:
            node.right = cls.from_dict(data["right"])
        return node


@register_model(name="decision_tree_regressor", category="trees")
class DecisionTreeRegressor(BaseModel):
    """
    Decision Tree Regressor using CART (Classification and Regression Trees).

    Partitions feature space into orthogonal hyper-rectangles by recursively maximizing
    variance reduction (mean squared error reduction). Leaf nodes predict the empirical
    mean of target values belonging to that region.

    Parameters:
        max_depth: Maximum depth of the tree (default: 5).
        min_samples_split: Minimum number of samples required to split an internal node (default: 2).
        min_samples_leaf: Minimum number of samples required to be at a leaf node (default: 1).
        min_impurity_decrease: Threshold for variance reduction to proceed with a split (default: 0.0).
        max_features: Optional maximum number of features to consider when looking for best split.
    """

    def __init__(
        self,
        max_depth: Optional[int] = 5,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        max_features: Optional[Union[int, float, str]] = None,
    ):
        super().__init__()
        if max_depth is not None and max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth}")
        if min_samples_split < 2:
            raise ValueError(f"min_samples_split must be >= 2, got {min_samples_split}")
        if min_samples_leaf < 1:
            raise ValueError(f"min_samples_leaf must be >= 1, got {min_samples_leaf}")
        if min_impurity_decrease < 0:
            raise ValueError(f"min_impurity_decrease must be >= 0, got {min_impurity_decrease}")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = float(min_impurity_decrease)
        self.max_features = max_features

        self.root: Optional[TreeNode] = None
        self.n_features_in_: Optional[int] = None
        self.feature_importances_: Optional[np.ndarray] = None

    def _to_numpy_2d(self, X: Any, y: Optional[Any] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Converts inputs into numpy float64 arrays."""
        if isinstance(X, Tensor):
            X_np = np.array(X.to_list(), dtype=np.float64)
        elif isinstance(X, np.ndarray):
            X_np = X.astype(np.float64)
        else:
            X_np = np.array(X, dtype=np.float64)

        if X_np.ndim == 1:
            X_np = X_np.reshape((-1, 1))
        elif X_np.ndim != 2:
            raise ValueError(f"Expected 2D features (N, D), got shape {X_np.shape}")

        if y is None:
            return X_np, None

        if isinstance(y, Tensor):
            y_np = np.array(y.to_list(), dtype=np.float64)
        elif isinstance(y, np.ndarray):
            y_np = y.astype(np.float64)
        else:
            y_np = np.array(y, dtype=np.float64)

        y_np = y_np.reshape((-1,))
        if X_np.shape[0] != y_np.shape[0]:
            raise ValueError(f"Sample count mismatch: X has {X_np.shape[0]}, y has {y_np.shape[0]}")

        return X_np, y_np

    def fit(self, X: Any, y: Any, **kwargs: Any) -> "DecisionTreeRegressor":
        """
        Builds a Decision Tree Regressor from training set (X, y).

        Args:
            X: Training features (Tensor, ndarray, or list).
            y: Target values.

        Returns:
            self: The fitted decision tree.
        """
        X_np, y_np = self._to_numpy_2d(X, y)
        n_samples, n_features = X_np.shape
        self.n_features_in_ = n_features
        self._raw_importances = np.zeros(n_features, dtype=np.float64)

        self.root = self._build_tree(X_np, y_np, depth=0)

        # Normalize feature importances
        total_imp = np.sum(self._raw_importances)
        if total_imp > 0:
            self.feature_importances_ = self._raw_importances / total_imp
        else:
            self.feature_importances_ = np.zeros(n_features, dtype=np.float64)

        self._is_fitted = True
        return self

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        n_samples = len(y)
        node_mean = float(np.mean(y)) if n_samples > 0 else 0.0
        node_var = float(np.var(y)) if n_samples > 0 else 0.0

        # Stopping criteria
        if (
            n_samples < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
            or node_var <= 1e-12
        ):
            return TreeNode(value=node_mean, mse=node_var, n_samples=n_samples, is_leaf=True)

        best_feat, best_thresh, best_gain = self._find_best_split(X, y, node_var)

        if best_feat is None or best_gain < self.min_impurity_decrease:
            return TreeNode(value=node_mean, mse=node_var, n_samples=n_samples, is_leaf=True)

        # Accumulate feature importance: n_samples * gain
        self._raw_importances[best_feat] += n_samples * best_gain

        # Split data
        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return TreeNode(
            feature_idx=best_feat,
            threshold=best_thresh,
            left=left_child,
            right=right_child,
            value=node_mean,
            mse=node_var,
            n_samples=n_samples,
            is_leaf=False,
        )

    def _find_best_split(
        self, X: np.ndarray, y: np.ndarray, current_var: float
    ) -> Tuple[Optional[int], Optional[float], float]:
        n_samples, n_features = X.shape
        best_gain = -1.0
        best_feat: Optional[int] = None
        best_thresh: Optional[float] = None

        features_to_check = list(range(n_features))
        if self.max_features is not None:
            if isinstance(self.max_features, int):
                k = min(self.max_features, n_features)
            elif isinstance(self.max_features, float):
                k = max(1, int(self.max_features * n_features))
            else:
                k = n_features
            features_to_check = list(np.random.choice(n_features, size=k, replace=False))

        for feat in features_to_check:
            feat_vals = X[:, feat]
            unique_vals = np.unique(feat_vals)
            if len(unique_vals) <= 1:
                continue

            # Candidate thresholds as midpoints between adjacent unique sorted values
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            # Cap threshold evaluations for high-cardinality continuous features
            if len(thresholds) > 50:
                thresholds = np.quantile(thresholds, np.linspace(0.05, 0.95, 30))

            for thresh in thresholds:
                left_mask = feat_vals <= thresh
                n_left = np.sum(left_mask)
                n_right = n_samples - n_left

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                y_left = y[left_mask]
                y_right = y[~left_mask]

                var_left = float(np.var(y_left))
                var_right = float(np.var(y_right))

                child_var = (n_left / n_samples) * var_left + (n_right / n_samples) * var_right
                gain = current_var - child_var

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = float(thresh)

        return best_feat, best_thresh, max(0.0, best_gain)

    def forward(self, X: Any) -> Tensor:
        """
        Traverses the decision tree for input tensor X and returns predicted leaf means.

        Args:
            X: Feature Tensor or array of shape (N, in_features).

        Returns:
            Tensor: Prediction tensor of shape (N, 1).
        """
        X_np, _ = self._to_numpy_2d(X)
        if self.root is None:
            raise RuntimeError("DecisionTreeRegressor is not fitted yet. Call fit() first.")

        if self.n_features_in_ is not None and X_np.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Feature dimension mismatch: model was fitted with {self.n_features_in_} features, "
                f"got {X_np.shape[1]}"
            )

        preds = [self._predict_row(self.root, row) for row in X_np]
        return Tensor(preds, shape=(len(preds), 1))

    def _predict_row(self, node: TreeNode, row: np.ndarray) -> float:
        curr = node
        while not curr.is_leaf:
            if row[curr.feature_idx] <= curr.threshold:
                curr = curr.left  # type: ignore
            else:
                curr = curr.right  # type: ignore
        return curr.value if curr.value is not None else 0.0

    def predict(self, X: Any) -> Tensor:
        """
        Predicts target values for inputs X.

        Args:
            X: Input features (Tensor, ndarray, or list).

        Returns:
            Tensor: Predicted targets of shape (N, 1).
        """
        self.eval()
        with no_grad():
            return self.forward(X)

    def tree_depth(self) -> int:
        """Returns the actual depth of the fitted tree."""
        if self.root is None:
            return 0

        def _depth(node: Optional[TreeNode]) -> int:
            if node is None or node.is_leaf:
                return 0
            return 1 + max(_depth(node.left), _depth(node.right))

        return _depth(self.root)

    def n_leaves(self) -> int:
        """Returns the number of leaf nodes in the tree."""
        if self.root is None:
            return 0

        def _count(node: Optional[TreeNode]) -> int:
            if node is None:
                return 0
            if node.is_leaf:
                return 1
            return _count(node.left) + _count(node.right)

        return _count(self.root)

    def feature_importances(self) -> np.ndarray:
        """Returns normalized feature importances based on variance reduction."""
        if self.feature_importances_ is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.feature_importances_

    def get_config(self) -> Dict[str, Any]:
        """Returns model hyperparameters for serialization."""
        return {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "min_impurity_decrease": self.min_impurity_decrease,
            "max_features": self.max_features,
            "n_features_in_": self.n_features_in_,
        }

    def state_dict(self, destination: Optional[dict] = None, prefix: str = "", keep_vars: bool = False) -> Dict[str, Any]:
        """Serializes model hyperparameters and tree node structure."""
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        state["tree_root"] = self.root.to_dict() if self.root is not None else None
        state["n_features_in_"] = self.n_features_in_
        state["feature_importances_"] = self.feature_importances_.tolist() if self.feature_importances_ is not None else None
        return state

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> Dict[str, List[str]]:
        """Restores tree structure from saved state dict."""
        if "tree_root" in state_dict:
            self.root = TreeNode.from_dict(state_dict["tree_root"])
        if "n_features_in_" in state_dict:
            self.n_features_in_ = state_dict["n_features_in_"]
        if "feature_importances_" in state_dict and state_dict["feature_importances_"] is not None:
            self.feature_importances_ = np.array(state_dict["feature_importances_"])
        self._is_fitted = self.root is not None
        return {"missing_keys": [], "unexpected_keys": []}

    def __repr__(self) -> str:
        return (
            f"DecisionTreeRegressor(max_depth={self.max_depth}, "
            f"min_samples_split={self.min_samples_split}, "
            f"min_samples_leaf={self.min_samples_leaf}, "
            f"min_impurity_decrease={self.min_impurity_decrease})"
        )


# Register aliases
register_model(DecisionTreeRegressor, name="decision_tree", category="trees")
register_model(DecisionTreeRegressor, name="decision_tree_regression", category="trees")

__all__ = ["DecisionTreeRegressor", "TreeNode"]
