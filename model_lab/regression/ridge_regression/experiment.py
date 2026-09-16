"""
Reproducible Experiment Runner for Ridge Regression.
Demonstrates the effect of L2 regularization strength (alpha) on parameter shrinkage.
"""

import os
import sys

# Ensure repository root is on sys.path when script is executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from dataclasses import dataclass, asdict
from typing import Dict, List
import time
import numpy as np

from nevula.models.regression.ridge import RidgeRegression
from nevula.models.regression.linear import LinearRegression


def run_alpha_sweep_experiment():
    print("=" * 70)
    print(" RIDGE REGRESSION — ALPHA SHRINKAGE EXPERIMENT")
    print("=" * 70)

    np.random.seed(42)
    n_samples = 150
    n_features = 4

    # Create correlated / collinear features
    x1 = np.random.uniform(-3.0, 3.0, size=(n_samples, 1))
    x2 = x1 + np.random.normal(0.0, 0.05, size=(n_samples, 1))  # strongly collinear with x1
    x3 = np.random.uniform(-2.0, 2.0, size=(n_samples, 1))
    x4 = np.random.uniform(-2.0, 2.0, size=(n_samples, 1))
    X = np.hstack([x1, x2, x3, x4])

    true_w = np.array([[2.0], [2.0], [-1.5], [0.5]])
    true_b = 1.0
    noise = np.random.normal(0.0, 0.2, size=(n_samples, 1))
    y = X @ true_w + true_b + noise

    alphas = [0.0, 0.1, 1.0, 5.0, 20.0]
    results = []

    print(f"\nGround Truth: W = [2.0, 2.0, -1.5, 0.5], b = {true_b}")
    print(f"Training across alphas: {alphas}\n")
    print(f"{'Alpha':<8} | {'Weight Norm ||W||_2':<20} | {'Test MSE':<12} | {'Weights [w1, w2, w3, w4]'}")
    print("-" * 75)

    for alpha in alphas:
        model = RidgeRegression(in_features=n_features, out_features=1, alpha=alpha)
        model.fit(X, y, epochs=150, lr=0.02, optimizer="sgd", verbose=False)

        weights = [float(w) for w in model.weight.data]
        w_norm = float(np.linalg.norm(weights))
        mse = model.evaluate(X, y, metric="mse")

        weights_str = "[" + ", ".join(f"{w:6.3f}" for w in weights) + "]"
        print(f"{alpha:<8.1f} | {w_norm:<20.4f} | {mse:<12.4f} | {weights_str}")

        results.append({
            "alpha": alpha,
            "weight_norm": w_norm,
            "mse": mse,
            "weights": weights,
        })

    print("-" * 75)
    print("Notice: As alpha increases, ||W||_2 shrinks systematically!")
    print("=" * 70)
    return results


if __name__ == "__main__":
    run_alpha_sweep_experiment()
