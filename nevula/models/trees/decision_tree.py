from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np

from nevula.core.tensor import Tensor
from nevula.autograd.engine import no_grad
from nevula.models.base import BaseModel
from nevula.models.registry import register_model
from nevula.metrics.classification import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    log_loss_score,
)


@dataclass
class TreeNode:
    """Represents a single node in a decision tree (regression or classification)."""
    feature_idx: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    value: Optional[Union[float, int]] = None
    probabilities: Optional[np.ndarray] = None
    impurity: float = 0.0
    mse: float = 0.0
    n_samples: int = 0
    is_leaf: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serializes node and child branches into a dictionary."""
        return {
            "feature_idx": self.feature_idx,
            "threshold": self.threshold,
            "value": self.value,
            "probabilities": self.probabilities.tolist() if self.probabilities is not None else None,
            "impurity": self.impurity,
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
        probs = data.get("probabilities")
        node = cls(
            feature_idx=data.get("feature_idx"),
            threshold=data.get("threshold"),
            value=data.get("value"),
            probabilities=np.array(probs, dtype=np.float64) if probs is not None else None,
            impurity=data.get("impurity", data.get("mse", 0.0)),
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


# Register aliases for regression
register_model(DecisionTreeRegressor, name="decision_tree", category="trees")
register_model(DecisionTreeRegressor, name="decision_tree_regression", category="trees")


@register_model(name="decision_tree_classifier", category="trees")
class DecisionTreeClassifier(BaseModel):
    """
    Decision Tree Classifier using CART (Classification and Regression Trees).

    Partitions feature space into orthogonal hyper-rectangles by recursively maximizing
    impurity reduction (Gini impurity or Information Gain / Entropy). Leaf nodes predict
    the majority class and empirical class probability distributions.

    Mathematical Formulation:
        Impurity Criteria:
            - Gini Impurity:
                I_Gini(S) = 1 - sum_{k=1}^K p_k^2
            - Entropy (Log Loss):
                I_Entropy(S) = - sum_{k=1}^K p_k * log2(p_k + eps)

        Split Gain:
            Delta I = I(S) - ( (N_L / N) * I(S_L) + (N_R / N) * I(S_R) )

        Class Probabilities:
            p_k = count(y in leaf == k) / N_leaf

        Predicted Class:
            y_hat = argmax_k p_k

    Parameters:
        criterion: Impurity metric, 'gini' or 'entropy' (default: 'gini').
        max_depth: Maximum depth of the tree (default: None, unbounded until leaves are pure).
        min_samples_split: Minimum number of samples required to split an internal node (default: 2).
        min_samples_leaf: Minimum number of samples required to be at a leaf node (default: 1).
        min_impurity_decrease: Threshold for impurity reduction to split a node (default: 0.0).
        max_features: Number of features to consider when looking for the best split (default: None).
        n_classes: Optional number of classes. Inferred automatically from targets during fit if None.
    """

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        max_features: Optional[Union[int, float, str]] = None,
        n_classes: Optional[int] = None,
    ):
        super().__init__()
        if criterion not in ("gini", "entropy"):
            raise ValueError(f"criterion must be 'gini' or 'entropy', got '{criterion}'")
        if max_depth is not None and max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth}")
        if min_samples_split < 2:
            raise ValueError(f"min_samples_split must be >= 2, got {min_samples_split}")
        if min_samples_leaf < 1:
            raise ValueError(f"min_samples_leaf must be >= 1, got {min_samples_leaf}")
        if min_impurity_decrease < 0:
            raise ValueError(f"min_impurity_decrease must be >= 0, got {min_impurity_decrease}")

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = float(min_impurity_decrease)
        self.max_features = max_features
        self.n_classes_param = n_classes

        self.root: Optional[TreeNode] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: int = 0
        self.n_features_in_: Optional[int] = None
        self.feature_importances_: Optional[np.ndarray] = None

    def _to_numpy_data(self, X: Any, y: Optional[Any] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Converts inputs into numpy arrays."""
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
            y_np = np.array(y.to_list())
        elif isinstance(y, np.ndarray):
            y_np = y
        else:
            y_np = np.array(y)

        y_np = y_np.reshape((-1,))
        if X_np.shape[0] != y_np.shape[0]:
            raise ValueError(f"Sample count mismatch: X has {X_np.shape[0]}, y has {y_np.shape[0]}")

        return X_np, y_np

    def _compute_impurity(self, y: np.ndarray) -> float:
        """Computes impurity (Gini or Entropy) for class labels vector y."""
        n_samples = len(y)
        if n_samples == 0:
            return 0.0

        # Frequency of each class index (0 .. n_classes_ - 1)
        counts = np.bincount(y, minlength=self.n_classes_)
        probs = counts / n_samples

        if self.criterion == "gini":
            # Gini = 1 - sum(p_k^2)
            return float(1.0 - np.sum(probs ** 2))
        else:
            # Entropy = - sum(p_k * log2(p_k))
            # Filter non-zero probs to prevent log2(0)
            nonzero_p = probs[probs > 0]
            if len(nonzero_p) == 0:
                return 0.0
            return float(-np.sum(nonzero_p * np.log2(nonzero_p)))

    def _leaf_node(self, y: np.ndarray, impurity: float) -> TreeNode:
        """Constructs a leaf node given class labels."""
        n_samples = len(y)
        if n_samples == 0:
            probs = np.zeros(self.n_classes_, dtype=np.float64)
            val = 0
        else:
            counts = np.bincount(y, minlength=self.n_classes_)
            probs = counts / n_samples
            val = int(np.argmax(counts))

        return TreeNode(
            value=val,
            probabilities=probs,
            impurity=impurity,
            mse=impurity,
            n_samples=n_samples,
            is_leaf=True,
        )

    def fit(self, X: Any, y: Any, **kwargs: Any) -> "DecisionTreeClassifier":
        """
        Builds a CART Decision Tree Classifier from training set (X, y).

        Args:
            X: Training features (Tensor, ndarray, or list of shape (N, D)).
            y: Target class labels (Tensor, ndarray, or list of shape (N,)).

        Returns:
            self: The fitted classifier instance.
        """
        X_np, y_np = self._to_numpy_data(X, y)
        n_samples, n_features = X_np.shape
        self.n_features_in_ = n_features

        # Identify unique classes and map labels to contiguous indices 0 .. K-1
        unique_classes = np.unique(y_np)
        if self.n_classes_param is not None:
            self.n_classes_ = self.n_classes_param
            self.classes_ = np.arange(self.n_classes_)
        else:
            self.classes_ = unique_classes
            self.n_classes_ = len(unique_classes)

        # Map y to integer indices [0 .. K-1]
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_indices = np.array([class_to_idx[val] for val in y_np], dtype=np.int64)

        self._raw_importances = np.zeros(n_features, dtype=np.float64)
        self.root = self._build_tree(X_np, y_indices, depth=0)

        total_imp = np.sum(self._raw_importances)
        if total_imp > 0:
            self.feature_importances_ = self._raw_importances / total_imp
        else:
            self.feature_importances_ = np.zeros(n_features, dtype=np.float64)

        self._is_fitted = True
        return self

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        n_samples = len(y)
        node_impurity = self._compute_impurity(y)

        # Stopping criteria: pure node, min_samples_split, or max_depth
        if (
            n_samples < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
            or node_impurity <= 1e-12
        ):
            return self._leaf_node(y, node_impurity)

        best_feat, best_thresh, best_gain = self._find_best_split(X, y, node_impurity)

        if best_feat is None or best_gain < self.min_impurity_decrease:
            return self._leaf_node(y, node_impurity)

        # Accumulate feature importance: n_samples * gain
        self._raw_importances[best_feat] += n_samples * best_gain

        # Partition data
        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        counts = np.bincount(y, minlength=self.n_classes_)
        probs = counts / n_samples
        val = int(np.argmax(counts))

        return TreeNode(
            feature_idx=best_feat,
            threshold=best_thresh,
            left=left_child,
            right=right_child,
            value=val,
            probabilities=probs,
            impurity=node_impurity,
            mse=node_impurity,
            n_samples=n_samples,
            is_leaf=False,
        )

    def _find_best_split(
        self, X: np.ndarray, y: np.ndarray, current_impurity: float
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

            # Candidate thresholds as midpoints
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
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

                imp_left = self._compute_impurity(y_left)
                imp_right = self._compute_impurity(y_right)

                child_imp = (n_left / n_samples) * imp_left + (n_right / n_samples) * imp_right
                gain = current_impurity - child_imp

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = float(thresh)

        return best_feat, best_thresh, max(0.0, best_gain)

    def _traverse_to_leaf(self, node: TreeNode, row: np.ndarray) -> TreeNode:
        """Traverses tree to locate corresponding leaf node for sample row."""
        curr = node
        while not curr.is_leaf:
            if row[curr.feature_idx] <= curr.threshold:
                curr = curr.left  # type: ignore
            else:
                curr = curr.right  # type: ignore
        return curr

    def predict_proba(self, X: Any) -> Tensor:
        """
        Predicts class probabilities for input features X.

        Args:
            X: Input features (Tensor, ndarray, or sequence) of shape (N, D).

        Returns:
            Tensor: Predicted class probabilities of shape (N, n_classes).
        """
        X_np, _ = self._to_numpy_data(X)
        if self.root is None or self.classes_ is None:
            raise RuntimeError("DecisionTreeClassifier is not fitted yet. Call fit() first.")

        if self.n_features_in_ is not None and X_np.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Feature dimension mismatch: model was fitted with {self.n_features_in_} features, "
                f"got {X_np.shape[1]}"
            )

        all_probs = []
        for row in X_np:
            leaf = self._traverse_to_leaf(self.root, row)
            if leaf.probabilities is not None:
                all_probs.append(leaf.probabilities)
            else:
                # Fallback one-hot
                p = np.zeros(self.n_classes_, dtype=np.float64)
                if leaf.value is not None:
                    p[int(leaf.value)] = 1.0
                all_probs.append(p)

        arr = np.array(all_probs, dtype=np.float64)
        return Tensor(arr, shape=(len(arr), self.n_classes_))

    def forward(self, X: Any) -> Tensor:
        """
        Forward pass returning predicted discrete class labels as a Tensor of shape (N, 1).

        Args:
            X: Feature Tensor or array of shape (N, in_features).

        Returns:
            Tensor: Prediction tensor of shape (N, 1).
        """
        X_np, _ = self._to_numpy_data(X)
        if self.root is None or self.classes_ is None:
            raise RuntimeError("DecisionTreeClassifier is not fitted yet. Call fit() first.")

        if self.n_features_in_ is not None and X_np.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Feature dimension mismatch: model was fitted with {self.n_features_in_} features, "
                f"got {X_np.shape[1]}"
            )

        preds = []
        for row in X_np:
            leaf = self._traverse_to_leaf(self.root, row)
            class_idx = int(leaf.value) if leaf.value is not None else 0
            original_class = self.classes_[class_idx]
            preds.append(original_class)

        arr = np.array(preds, dtype=np.float64).reshape((-1, 1))
        return Tensor(arr, shape=(len(preds), 1))

    def predict(self, X: Any) -> Tensor:
        """
        Generates class predictions for inputs X.

        Args:
            X: Input features (Tensor, ndarray, or list).

        Returns:
            Tensor: Predicted class labels of shape (N, 1).
        """
        self.eval()
        with no_grad():
            return self.forward(X)

    def tree_depth(self) -> int:
        """Returns the actual depth of the fitted decision tree."""
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
        """Returns normalized feature importances based on total impurity reduction."""
        if self.feature_importances_ is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.feature_importances_

    def evaluate(self, X: Any, y: Any, metric: str = "accuracy") -> float:
        """
        Evaluates the fitted classifier on test data according to specified metric.

        Args:
            X: Test features.
            y: Ground truth labels.
            metric: 'accuracy', 'precision', 'recall', 'f1', or 'log_loss'.

        Returns:
            float: Metric score.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before evaluation.")

        preds = self.predict(X)
        y_true_np = np.array(y.to_list() if isinstance(y, Tensor) else y, dtype=np.float64).ravel()
        y_pred_np = np.array(preds.to_list(), dtype=np.float64).ravel()

        metric_lower = metric.lower().strip()
        if metric_lower in ("accuracy", "acc"):
            return accuracy_score(y_true_np, y_pred_np)
        elif metric_lower == "precision":
            avg = "binary" if self.n_classes_ == 2 else "macro"
            return precision_score(y_true_np, y_pred_np, average=avg)
        elif metric_lower == "recall":
            avg = "binary" if self.n_classes_ == 2 else "macro"
            return recall_score(y_true_np, y_pred_np, average=avg)
        elif metric_lower in ("f1", "f1_score"):
            avg = "binary" if self.n_classes_ == 2 else "macro"
            return f1_score(y_true_np, y_pred_np, average=avg)
        elif metric_lower in ("log_loss", "cross_entropy"):
            probs = self.predict_proba(X)
            probs_np = np.array(probs.to_list(), dtype=np.float64)
            return log_loss_score(y_true_np, probs_np)
        else:
            return super().evaluate(X, y, metric=metric)

    def get_config(self) -> Dict[str, Any]:
        """Returns model hyperparameters for serialization."""
        return {
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "min_impurity_decrease": self.min_impurity_decrease,
            "max_features": self.max_features,
            "n_classes": self.n_classes_,
            "n_features_in_": self.n_features_in_,
        }

    def state_dict(self, destination: Optional[dict] = None, prefix: str = "", keep_vars: bool = False) -> Dict[str, Any]:
        """Serializes model hyperparameters and tree node structure."""
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        state["tree_root"] = self.root.to_dict() if self.root is not None else None
        state["classes_"] = self.classes_.tolist() if self.classes_ is not None else None
        state["n_classes_"] = self.n_classes_
        state["n_features_in_"] = self.n_features_in_
        state["feature_importances_"] = self.feature_importances_.tolist() if self.feature_importances_ is not None else None
        return state

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> Dict[str, List[str]]:
        """Restores tree structure from saved state dict."""
        if "tree_root" in state_dict:
            self.root = TreeNode.from_dict(state_dict["tree_root"])
        if "classes_" in state_dict and state_dict["classes_"] is not None:
            self.classes_ = np.array(state_dict["classes_"])
        if "n_classes_" in state_dict:
            self.n_classes_ = state_dict["n_classes_"]
        if "n_features_in_" in state_dict:
            self.n_features_in_ = state_dict["n_features_in_"]
        if "feature_importances_" in state_dict and state_dict["feature_importances_"] is not None:
            self.feature_importances_ = np.array(state_dict["feature_importances_"])
        self._is_fitted = self.root is not None
        return {"missing_keys": [], "unexpected_keys": []}

    def __repr__(self) -> str:
        return (
            f"DecisionTreeClassifier(criterion='{self.criterion}', "
            f"max_depth={self.max_depth}, "
            f"min_samples_split={self.min_samples_split}, "
            f"min_samples_leaf={self.min_samples_leaf}, "
            f"min_impurity_decrease={self.min_impurity_decrease})"
        )


# Register aliases for classification
register_model(DecisionTreeClassifier, name="decision_tree_classifier", category="classification")
register_model(DecisionTreeClassifier, name="decision_tree_classification", category="classification")
register_model(DecisionTreeClassifier, name="decision_tree_classification", category="trees")
register_model(DecisionTreeClassifier, name="dtc", category="trees")
register_model(DecisionTreeClassifier, name="dtc", category="classification")

__all__ = ["DecisionTreeRegressor", "DecisionTreeClassifier", "TreeNode"]

