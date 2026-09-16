"""
Nevula Example: Support Vector Regression & Outlier Resilience Demonstration.

Demonstrates:
  1. Creating a synthetic dataset contaminated with extreme vertical outliers.
  2. Training Ordinary Least Squares (OLS) vs Support Vector Regression (SVR).
  3. Visualizing parameter distortion in OLS vs robustness in SVR.
  4. Inspecting the epsilon-insensitive tube and support vectors.
"""

import os
import sys

# Ensure repository root is on sys.path when script is executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from nevula.models import LinearRegression, SVR, list_models, summary
from nevula.metrics import mean_squared_error, r2_score


def main():
    print("=" * 72)
    print("       NEVULA MODEL LIBRARY — SUPPORT VECTOR REGRESSION DEMO")
    print("=" * 72)

    # 1. Dataset Generation: True line y = 2.5 * x + 1.0 with severe outliers
    print("\n--- 1. Generating Contaminated Dataset ---")
    np.random.seed(42)
    n_clean = 120
    n_outliers = 12

    true_w = 2.5
    true_b = 1.0

    X_clean = np.random.uniform(-3.0, 3.0, size=(n_clean, 1))
    noise = np.random.normal(0.0, 0.08, size=(n_clean, 1))
    y_clean = true_w * X_clean + true_b + noise

    # Invert and spike outliers
    X_out = np.random.uniform(1.0, 3.0, size=(n_outliers, 1))
    y_out = np.ones((n_outliers, 1)) * -30.0  # extreme downward spikes

    X_train = np.vstack([X_clean, X_out])
    y_train = np.vstack([y_clean, y_out])

    # Uncontaminated Test Set for fair evaluation
    X_test = np.random.uniform(-3.0, 3.0, size=(40, 1))
    y_test = true_w * X_test + true_b + np.random.normal(0.0, 0.08, size=(40, 1))

    print(f"Dataset: {n_clean} clean samples (y = {true_w}x + {true_b})")
    print(f"Anomalies: {n_outliers} severe outlier points (y = -30.0)")

    # 2. Train Ordinary Least Squares (OLS)
    print("\n--- 2. Training Ordinary Linear Regression (Quadratic Loss) ---")
    ols = LinearRegression(in_features=1, out_features=1)
    ols.fit(X_train, y_train, epochs=200, lr=0.02, optimizer="sgd", verbose=False)
    ols_w = float(ols.weight.data[0])
    ols_b = float(ols.bias_param.data[0])

    print(f"OLS Learned Slope:     {ols_w:7.4f} (True: {true_w:.4f})")
    print(f"OLS Learned Intercept: {ols_b:7.4f} (True: {true_b:.4f})")
    print(f"OLS Slope Deviation:   {abs(ols_w - true_w):7.4f}")

    # 3. Train Support Vector Regression (SVR)
    print("\n--- 3. Training Support Vector Regression (Epsilon-Tube Loss) ---")
    svr = SVR(in_features=1, out_features=1, epsilon=0.2, C=2.0)
    svr.fit(X_train, y_train, epochs=250, lr=0.03, optimizer="sgd", verbose=False)
    svr_w = float(svr.weight.data[0])
    svr_b = float(svr.bias_param.data[0])

    print(f"SVR Learned Slope:     {svr_w:7.4f} (True: {true_w:.4f})")
    print(f"SVR Learned Intercept: {svr_b:7.4f} (True: {true_b:.4f})")
    print(f"SVR Slope Deviation:   {abs(svr_w - true_w):7.4f}")

    # 4. Support Vectors Analysis
    sv_ratio = svr.support_vectors_ratio(X_train, y_train)
    print(f"\n--- 4. Support Vectors Analysis ---")
    print(f"Epsilon Tube Radius:    epsilon = {svr.epsilon}")
    print(f"Support Vectors Ratio:  {sv_ratio * 100:.1f}% of data lie on/outside margin")

    # 5. Generalization on Clean Test Set
    print("\n--- 5. Test Set Generalization (Uncontaminated) ---")
    ols_r2 = ols.evaluate(X_test, y_test, metric="r2")
    svr_r2 = svr.evaluate(X_test, y_test, metric="r2")
    print(f"OLS Test R^2:  {ols_r2:.6f}  <- Heavily degraded by outliers")
    print(f"SVR Test R^2:  {svr_r2:.6f}  <- Robust to outliers!")

    print("\n" + "=" * 72)
    print(" Support Vector Regression demonstration completed successfully!")
    print("=" * 72)


if __name__ == "__main__":
    main()
