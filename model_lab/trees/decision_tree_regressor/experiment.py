"""
Reproducible Experiment Runner for Decision Tree Regression.
Compares non-linear function approximation capability of DecisionTreeRegressor against Linear Regression.
"""

import os
import sys

# Ensure repository root is on sys.path when script is executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
from nevula.models.trees.decision_tree import DecisionTreeRegressor
from nevula.models.regression.linear import LinearRegression


def run_non_linear_benchmark():
    print("=" * 72)
    print(" DECISION TREE vs LINEAR REGRESSION — NON-LINEAR BENCHMARK")
    print("=" * 72)

    np.random.seed(42)
    n_samples = 200

    # Non-linear function: y = sin(x) + 0.5 * step(x > 0) + noise
    X = np.sort(np.random.uniform(-np.pi, np.pi, size=(n_samples, 1)), axis=0)
    y = np.sin(X[:, 0]) + 0.5 * (X[:, 0] > 0) + np.random.normal(0.0, 0.05, size=(n_samples,))

    # 1. Fit Ordinary Linear Regression
    ols = LinearRegression(in_features=1, out_features=1)
    ols.fit(X, y.reshape((-1, 1)), epochs=150, lr=0.03, verbose=False)
    ols_r2 = ols.evaluate(X, y.reshape((-1, 1)), metric="r2")
    ols_mse = ols.evaluate(X, y.reshape((-1, 1)), metric="mse")

    # 2. Fit Decision Tree with various depths
    depths = [2, 4, 6]
    print(f"Dataset: {n_samples} samples approximating y = sin(x) + step(x > 0)")
    print(f"\n1. Ordinary Linear Regression (Cannot fit non-linear cycles):")
    print(f"   OLS R^2 Score: {ols_r2:.4f} | OLS MSE: {ols_mse:.4f}")

    print(f"\n2. Decision Tree Regressors across depths:")
    print(f"{'Max Depth':<12} | {'Tree Depth':<12} | {'Leaves':<10} | {'R^2 Score':<12} | {'MSE'}")
    print("-" * 62)

    for depth in depths:
        tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=2)
        tree.fit(X, y)
        r2 = tree.evaluate(X, y, metric="r2")
        mse = tree.evaluate(X, y, metric="mse")
        print(f"{depth:<12} | {tree.tree_depth():<12} | {tree.n_leaves():<10} | {r2:<12.4f} | {mse:.4f}")

    print("-" * 62)
    print("Notice: Decision Tree smoothly fits non-linear curves where linear models fail!")
    print("=" * 72)


if __name__ == "__main__":
    run_non_linear_benchmark()
