"""
Nevula Example: Ridge Regression & Multicollinearity Demonstration.

Demonstrates:
  1. Creating a synthetic dataset with highly collinear features (x2 ≈ x1).
  2. Training Ordinary Linear Regression (OLS) vs Ridge Regression.
  3. Visualizing parameter shrinkage (|W_ridge| < |W_ols|).
  4. Evaluating test set generalization performance (MSE and R^2 score).
"""

import os
import sys

# Ensure repository root is on sys.path when script is executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from nevula.models import LinearRegression, RidgeRegression, list_models, summary
from nevula.metrics import mean_squared_error, r2_score


def main():
    print("=" * 70)
    print("      NEVULA MODEL LIBRARY — RIDGE REGRESSION DEMONSTRATION")
    print("=" * 70)

    # 1. Dataset Generation with Collinear Features
    print("\n--- 1. Generating Synthetic Dataset with Multicollinearity ---")
    np.random.seed(42)
    n_samples = 250

    # Feature 1 is independent
    x1 = np.random.uniform(-3.0, 3.0, size=(n_samples, 1))
    # Feature 2 is strongly collinear with Feature 1 (x2 = x1 + tiny noise)
    x2 = x1 + np.random.normal(0.0, 0.05, size=(n_samples, 1))
    # Feature 3 is independent
    x3 = np.random.uniform(-2.0, 2.0, size=(n_samples, 1))

    X_raw = np.hstack([x1, x2, x3])
    true_w = np.array([[2.5], [2.5], [-1.8]])
    true_b = 3.0
    noise = np.random.normal(0.0, 0.25, size=(n_samples, 1))
    y_raw = X_raw @ true_w + true_b + noise

    # Split into Train (80%) and Test (20%)
    split = int(n_samples * 0.8)
    X_train, X_test = X_raw[:split], X_raw[split:]
    y_train, y_test = y_raw[:split], y_raw[split:]

    print(f"Generated {n_samples} samples with 3 features (features 1 & 2 are collinear).")
    print(f"Ground Truth: W = [2.5, 2.5, -1.8], b = {true_b}")

    # 2. Train Standard Linear Regression (Unregularized)
    print("\n--- 2. Training Ordinary Linear Regression (alpha = 0) ---")
    ols = LinearRegression(in_features=3, out_features=1)
    ols.fit(X_train, y_train, epochs=150, lr=0.03, optimizer="sgd", verbose=False)
    ols_weights = [float(w) for w in ols.weight.data]
    ols_b = float(ols.bias_param.data[0])
    ols_norm = float(np.linalg.norm(ols_weights))
    print(f"OLS Weights:       [{', '.join(f'{w:7.4f}' for w in ols_weights)}]")
    print(f"OLS Intercept:     {ols_b:.4f}")
    print(f"OLS Weight Norm:   {ols_norm:.4f}")

    # 3. Train Ridge Regression (L2 Regularized with alpha = 2.0)
    print("\n--- 3. Training Ridge Regression (alpha = 2.0) ---")
    ridge = RidgeRegression(in_features=3, out_features=1, alpha=2.0)
    ridge.fit(X_train, y_train, epochs=150, lr=0.03, optimizer="sgd", verbose=False)
    ridge_weights = [float(w) for w in ridge.weight.data]
    ridge_b = float(ridge.bias_param.data[0])
    ridge_norm = float(np.linalg.norm(ridge_weights))
    print(f"Ridge Weights:     [{', '.join(f'{w:7.4f}' for w in ridge_weights)}]")
    print(f"Ridge Intercept:   {ridge_b:.4f}")
    print(f"Ridge Weight Norm: {ridge_norm:.4f}")

    # 4. Compare Parameter Shrinkage
    print("\n--- 4. Weight Shrinkage Comparison ---")
    shrinkage_pct = (1.0 - (ridge_norm / ols_norm)) * 100.0
    print(f"||W_ols||_2:   {ols_norm:.4f}")
    print(f"||W_ridge||_2: {ridge_norm:.4f}")
    print(f"L2 Shrinkage:  {shrinkage_pct:.2f}% reduction in parameter norm!")

    # 5. Evaluate Generalization on Test Set
    print("\n--- 5. Generalization Evaluation on Test Set ---")
    ols_mse = ols.evaluate(X_test, y_test, metric="mse")
    ols_r2 = ols.evaluate(X_test, y_test, metric="r2")

    ridge_mse = ridge.evaluate(X_test, y_test, metric="mse")
    ridge_r2 = ridge.evaluate(X_test, y_test, metric="r2")

    print(f"OLS Test MSE:   {ols_mse:.6f} | OLS Test R^2:   {ols_r2:.6f}")
    print(f"Ridge Test MSE: {ridge_mse:.6f} | Ridge Test R^2: {ridge_r2:.6f}")

    print("\n" + "=" * 70)
    print(" Ridge Regression demonstration completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
