"""
Nevula Example: Decision Tree Regression & Non-Linear Modeling Demonstration.

Demonstrates:
  1. Creating a complex non-linear dataset (sine wave with step discontinuity).
  2. Training Linear Regression (fails on non-linear data) vs DecisionTreeRegressor.
  3. Visualizing piecewise constant approximation and structural tree depth.
  4. Evaluating non-linear generalization performance (R^2 and MSE).
"""

import os
import sys

# Ensure repository root is on sys.path when script is executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from nevula.models import LinearRegression, DecisionTreeRegressor, list_models, summary
from nevula.metrics import mean_squared_error, r2_score


def main():
    print("=" * 72)
    print("      NEVULA MODEL LIBRARY — DECISION TREE REGRESSION DEMO")
    print("=" * 72)

    # 1. Dataset Generation: Non-linear function with a step jump
    print("\n--- 1. Generating Non-Linear Sine Wave with Step Discontinuity ---")
    np.random.seed(42)
    n_samples = 240

    X_raw = np.sort(np.random.uniform(-np.pi, np.pi, size=(n_samples, 1)), axis=0)
    # y = sin(x) + 0.8 if x > 0 else 0.0 + tiny noise
    y_raw = np.sin(X_raw[:, 0]) + 0.8 * (X_raw[:, 0] > 0.0) + np.random.normal(0.0, 0.06, size=(n_samples,))

    # Train / Test split (80% / 20%)
    indices = np.random.permutation(n_samples)
    train_idx, test_idx = indices[: int(n_samples * 0.8)], indices[int(n_samples * 0.8) :]
    X_train, y_train = X_raw[train_idx], y_raw[train_idx]
    X_test, y_test = X_raw[test_idx], y_raw[test_idx]

    print(f"Generated {n_samples} samples approximating y = sin(x) + 0.8 * step(x > 0).")
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 2. Train Ordinary Linear Regression (Parametric linear model)
    print("\n--- 2. Training Ordinary Linear Regression (Parametric) ---")
    ols = LinearRegression(in_features=1, out_features=1)
    ols.fit(X_train, y_train.reshape((-1, 1)), epochs=150, lr=0.03, verbose=False)
    ols_r2 = ols.evaluate(X_test, y_test.reshape((-1, 1)), metric="r2")
    ols_mse = ols.evaluate(X_test, y_test.reshape((-1, 1)), metric="mse")

    print(f"OLS Test R^2 Score: {ols_r2:7.4f} (Linear model cannot fit cyclic wave)")
    print(f"OLS Test MSE:       {ols_mse:7.4f}")

    # 3. Train Decision Tree Regressor (Non-parametric tree partition)
    print("\n--- 3. Training Decision Tree Regressor (Non-Parametric CART) ---")
    tree = DecisionTreeRegressor(max_depth=5, min_samples_leaf=2)
    tree.fit(X_train, y_train)

    tree_r2 = tree.evaluate(X_test, y_test, metric="r2")
    tree_mse = tree.evaluate(X_test, y_test, metric="mse")

    print(f"Tree Test R^2 Score: {tree_r2:7.4f} (High-accuracy non-linear fit!)")
    print(f"Tree Test MSE:       {tree_mse:7.4f}")

    # 4. Tree Architecture Diagnostics
    print("\n--- 4. Tree Structural Properties ---")
    print(f"Configured Max Depth: {tree.max_depth}")
    print(f"Actual Tree Depth:    {tree.tree_depth()}")
    print(f"Number of Leaves:     {tree.n_leaves()} local regions")
    print(f"Feature Importances:  {tree.feature_importances()}")

    # 5. First 5 Test Set Predictions Comparison
    print("\n--- 5. Sample Predictions Comparison ---")
    tree_preds = np.array(tree.predict(X_test[:5]).to_list()).ravel()
    ols_preds = np.array(ols.predict(X_test[:5]).to_list()).ravel()

    for i in range(5):
        x_val = float(X_test[i, 0])
        y_true = float(y_test[i])
        print(
            f"x={x_val:6.2f} | y_true={y_true:6.2f} | "
            f"Tree={tree_preds[i]:6.2f} (err={abs(tree_preds[i]-y_true):.3f}) | "
            f"OLS={ols_preds[i]:6.2f} (err={abs(ols_preds[i]-y_true):.3f})"
        )

    print("\n" + "=" * 72)
    print(" Decision Tree Regression demonstration completed successfully!")
    print("=" * 72)


if __name__ == "__main__":
    main()
