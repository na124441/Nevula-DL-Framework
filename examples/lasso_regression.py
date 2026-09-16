"""
Nevula Example: Lasso Regression & Sparse Feature Selection Demonstration.

Demonstrates:
  1. Synthetic dataset with 8 features (2 true informative features, 6 noise features).
  2. Training Ordinary Linear Regression (OLS) vs Lasso Regression (L1).
  3. Automatic feature selection: Lasso zeros out the 6 noise coefficients.
  4. Evaluating generalization on test set and calculating sparsity ratio.
"""

import os
import sys

# Ensure repository root is on sys.path when script is executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from nevula.models import LinearRegression, LassoRegression, list_models, summary
from nevula.metrics import mean_squared_error, r2_score


def main():
    print("=" * 72)
    print("      NEVULA MODEL LIBRARY — LASSO REGRESSION & SPARSITY DEMO")
    print("=" * 72)

    # 1. Dataset Generation: 8 features, 2 informative, 6 pure noise
    print("\n--- 1. Generating High-Dimensional Sparse Feature Dataset ---")
    np.random.seed(42)
    n_samples = 220
    n_features = 8

    X_raw = np.random.uniform(-2.5, 2.5, size=(n_samples, n_features))
    # True relationship: y = 3.5 * x0 - 2.5 * x1 + 0 * (x2..x7) + 2.0
    true_w = np.array([[3.5], [-2.5], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
    true_b = 2.0
    noise = np.random.normal(0.0, 0.15, size=(n_samples, 1))
    y_raw = X_raw @ true_w + true_b + noise

    split = int(n_samples * 0.8)
    X_train, X_test = X_raw[:split], X_raw[split:]
    y_train, y_test = y_raw[:split], y_raw[split:]

    print(f"Generated {n_samples} samples across {n_features} features.")
    print("Ground Truth Informative Features: Feature 0 (w=3.5), Feature 1 (w=-2.5)")
    print("Ground Truth Irrelevant Noise:    Features 2 through 7 (w=0.0)")

    # 2. Train Unregularized Linear Regression
    print("\n--- 2. Training Ordinary Linear Regression (alpha = 0) ---")
    ols = LinearRegression(in_features=n_features, out_features=1)
    ols.fit(X_train, y_train, epochs=150, lr=0.03, optimizer="sgd", verbose=False)
    ols_weights = [float(w) for w in ols.weight.data]

    print(f"{'Feature':<10} | {'True Weight':<12} | {'OLS Weight':<12} | {'OLS Status'}")
    print("-" * 55)
    for i, w in enumerate(ols_weights):
        tw = float(true_w[i, 0])
        status = "Kept (Predictive)" if tw != 0.0 else "NOISE RETAINED"
        print(f"Feature {i:<2} | {tw:<12.2f} | {w:<12.4f} | {status}")

    # 3. Train Lasso Regression with alpha = 0.6
    print("\n--- 3. Training Lasso Regression (L1 Penalty, alpha = 0.6) ---")
    lasso = LassoRegression(in_features=n_features, out_features=1, alpha=0.6)
    lasso.fit(X_train, y_train, epochs=200, lr=0.03, tol=1e-2, verbose=False)
    lasso_weights = [float(w) for w in lasso.weight.data]

    print(f"{'Feature':<10} | {'True Weight':<12} | {'Lasso Weight':<12} | {'Lasso Selection'}")
    print("-" * 58)
    for i, w in enumerate(lasso_weights):
        tw = float(true_w[i, 0])
        status = "RETAINED" if abs(w) > 0.01 else "ZEROED OUT (Pruned)"
        print(f"Feature {i:<2} | {tw:<12.2f} | {w:<12.4f} | {status}")

    # 4. Inspect Sparsity Ratio
    sparsity_ratio = lasso.sparsity(tol=1e-2)
    print(f"\n--- 4. Sparsity Analysis ---")
    print(f"Lasso Sparsity:      {sparsity_ratio * 100:.1f}% of features successfully pruned to 0.0")
    print(f"Ground Truth Noise:  {6 / 8 * 100:.1f}% of total features were noise")

    # 5. Evaluate Test Set Performance
    print("\n--- 5. Generalization Performance on Test Set ---")
    ols_r2 = ols.evaluate(X_test, y_test, metric="r2")
    lasso_r2 = lasso.evaluate(X_test, y_test, metric="r2")
    print(f"OLS Test R^2:   {ols_r2:.6f}")
    print(f"Lasso Test R^2: {lasso_r2:.6f}")

    print("\n" + "=" * 72)
    print(" Lasso Regression demonstration completed successfully!")
    print("=" * 72)


if __name__ == "__main__":
    main()
