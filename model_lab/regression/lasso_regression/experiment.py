"""
Reproducible Experiment Runner for Lasso Regression.
Demonstrates feature selection and parameter sparsity by zeroing out irrelevant features.
"""

import os
import sys

# Ensure repository root is on sys.path when script is executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
from nevula.models.regression.lasso import LassoRegression
from nevula.models.regression.linear import LinearRegression


def run_feature_selection_experiment():
    print("=" * 72)
    print(" LASSO REGRESSION — SPARSE FEATURE SELECTION EXPERIMENT")
    print("=" * 72)

    np.random.seed(42)
    n_samples = 200
    n_features = 8

    # 8 total features:
    # Features 0 and 1 are informative: W = [3.0, -2.0, 0, 0, 0, 0, 0, 0]
    # Features 2 to 7 are pure noise
    X = np.random.uniform(-2.0, 2.0, size=(n_samples, n_features))
    true_w = np.array([[3.0], [-2.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
    true_b = 1.5
    noise = np.random.normal(0.0, 0.1, size=(n_samples, 1))
    y = X @ true_w + true_b + noise

    print(f"Generated {n_samples} samples with {n_features} features.")
    print("True Informative Features: Features 0 and 1 (weights: [3.0, -2.0])")
    print("True Noise Features:       Features 2 through 7 (weights: [0, 0, 0, 0, 0, 0])\n")

    # 1. Fit Ordinary Linear Regression (no regularization)
    ols = LinearRegression(in_features=n_features, out_features=1)
    ols.fit(X, y, epochs=150, lr=0.03, verbose=False)
    ols_weights = [float(w) for w in ols.weight.data]

    print("1. Ordinary Linear Regression (OLS):")
    for i, w in enumerate(ols_weights):
        status = "Informative" if i < 2 else "NOISE (failed to zero out)"
        print(f"   Feature {i}: {w:7.4f}  <- {status}")

    # 2. Fit Lasso Regression (alpha = 0.5)
    lasso = LassoRegression(in_features=n_features, out_features=1, alpha=0.5)
    lasso.fit(X, y, epochs=200, lr=0.03, tol=1e-2, verbose=False)
    lasso_weights = [float(w) for w in lasso.weight.data]

    print("\n2. Lasso Regression (alpha = 0.5):")
    for i, w in enumerate(lasso_weights):
        status = "Retained (Informative)" if abs(w) > 0.01 else "ZEROED OUT (Feature Selected!)"
        print(f"   Feature {i}: {w:7.4f}  <- {status}")

    sparsity_score = lasso.sparsity(tol=1e-2)
    print(f"\nLasso Sparsity Ratio: {sparsity_score * 100:.1f}% of features zeroed out!")
    print(f"True Noise Ratio:     {6 / 8 * 100:.1f}% (6 out of 8 were pure noise)")
    print("=" * 72)


if __name__ == "__main__":
    run_feature_selection_experiment()
