"""
Experimental Decision Tree Regressor Prototype.
Developed inside model_lab before official graduation to nevula.models.
"""

import os
import sys

# Ensure repository root is on sys.path when script is executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

from nevula.core.tensor import Tensor
from nevula.models.trees.decision_tree import TreeNode


class ExperimentalDecisionTreeRegressor:
    """
    Experimental prototype of CART decision tree regressor.
    """

    def __init__(self, max_depth: int = 5, min_samples_split: int = 2, min_samples_leaf: int = 1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root: Optional[TreeNode] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ExperimentalDecisionTreeRegressor":
        self.root = self._build(X, y, depth=0)
        return self

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        n_samples = len(y)
        val = float(np.mean(y)) if n_samples > 0 else 0.0
        var = float(np.var(y)) if n_samples > 0 else 0.0

        if depth >= self.max_depth or n_samples < self.min_samples_split or var <= 1e-9:
            return TreeNode(value=val, mse=var, n_samples=n_samples, is_leaf=True)

        best_feat, best_thresh, best_gain = None, None, -1.0
        for feat in range(X.shape[1]):
            vals = np.unique(X[:, feat])
            threshs = (vals[:-1] + vals[1:]) / 2.0
            for t in threshs:
                mask = X[:, feat] <= t
                nl, nr = np.sum(mask), n_samples - np.sum(mask)
                if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                    continue
                gain = var - ((nl / n_samples) * np.var(y[mask]) + (nr / n_samples) * np.var(y[~mask]))
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, t

        if best_feat is None or best_gain <= 0:
            return TreeNode(value=val, mse=var, n_samples=n_samples, is_leaf=True)

        left = self._build(X[X[:, best_feat] <= best_thresh], y[X[:, best_feat] <= best_thresh], depth + 1)
        right = self._build(X[X[:, best_feat] > best_thresh], y[X[:, best_feat] > best_thresh], depth + 1)
        return TreeNode(feature_idx=best_feat, threshold=best_thresh, left=left, right=right, value=val, is_leaf=False)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._eval(self.root, row) for row in X])

    def _eval(self, node: TreeNode, row: np.ndarray) -> float:
        curr = node
        while not curr.is_leaf:
            curr = curr.left if row[curr.feature_idx] <= curr.threshold else curr.right
        return curr.value
