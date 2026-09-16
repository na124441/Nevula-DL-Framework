"""
Reproducible Experiment Runner for Support Vector Regression (SVR).
Demonstrates outlier resistance and robustness comparing SVR against Ordinary Least Squares (OLS).
"""

import os
import sys

# Ensure repository root is on sys.path when script is executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
from nevula.models.regression.svr import SVR
from nevula.models.regression.linear import LinearRegression


def run_outlier_resilience_experiment():
    print("=" * 72)
    print(" SVR vs OLS — OUTLIER RESILIENCE BENCHMARK EXPERIMENT")
    print("=" * 72)

    np.random.seed(42)
    n_clean = 100
    n_outliers = 10

    # True data: y = 2.0 * x + 1.0
    true_slope = 2.0
    true_intercept = 1.0

    X_clean = np.random.uniform(-3.0, 3.0, size=(n_clean, 1))
    noise_clean = np.random.normal(0.0, 0.05, size=(n_clean, 1))
    y_clean = true_slope * X_clean + true_intercept + noise_clean

    # Contaminate with extreme vertical outliers
    X_outliers = np.random.uniform(1.0, 3.0, size=(n_outliers, 1))
    y_outliers = np.ones((n_outliers, 1)) * -25.0  # massive negative spikes

    X_train = np.vstack([X_clean, X_outliers])
    y_train = np.vstack([y_clean, y_outliers])

    print(f"Dataset: {n_clean} clean samples (y = {true_slope}x + {true_intercept})")
    print(f"Contamination: {n_outliers} severe outlier spikes (y = -25.0)")

    # 1. Train Ordinary Least Squares (OLS)
    ols = LinearRegression(in_features=1, out_features=1)
    ols.fit(X_train, y_train, epochs=200, lr=0.02, verbose=False)
    ols_w = float(ols.weight.data[0])
    ols_b = float(ols.bias_param.data[0])

    # 2. Train Support Vector Regression (SVR)
    svr = SVR(in_features=1, out_features=1, epsilon=0.15, C=1.0)
    svr.fit(X_train, y_train, epochs=250, lr=0.03, verbose=False)
    svr_w = float(svr.weight.data[0])
    svr_b = float(svr.bias_param.data[0])

    sv_ratio = svr.support_vectors_ratio(X_train, y_train)

    print("\n" + "-" * 72)
    print(f"{'Metric':<25} | {'Ground Truth':<14} | {'OLS (MSE)':<14} | {'SVR (Epsilon-Tube)'}")
    print("-" * 72)
    print(f"{'Slope (W)':<25} | {true_slope:<14.4f} | {ols_w:<14.4f} | {svr_w:<14.4f}")
    print(f"{'Intercept (b)':<25} | {true_intercept:<14.4f} | {ols_b:<14.4f} | {svr_b:<14.4f}")
    print(f"{'Slope Error':<25} | {0.0:<14.4f} | {abs(ols_w - true_slope):<14.4f} | {abs(svr_w - true_slope):<14.4f}")
    print("-" * 72)
    print(f"Support Vectors Ratio: {sv_ratio * 100:.1f}% of observations lie on/outside the margin")
    print("=" * 72)


if __name__ == "__main__":
    run_outlier_resilience_experiment()
