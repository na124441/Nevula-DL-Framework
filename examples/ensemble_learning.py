"""
Nevula Example: Ensemble Learning Demonstration.

Demonstrates the three core ensemble learning paradigms in Nevula:
  1. VotingRegressor: Heterogeneous model averaging (Ridge + SVR + Decision Tree).
  2. RandomForestRegressor: Bootstrap Aggregation (Bagging) + Random Subspaces.
  3. GradientBoostingRegressor: Stage-wise pseudo-residual boosting with shrinkage.
"""

import os
import sys

# Ensure repository root is on sys.path when script is executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from nevula.models import (
    VotingRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
    DecisionTreeRegressor,
    RidgeRegression,
    SVR,
    summary,
)


def main():
    print("=" * 74)
    print("        NEVULA MODEL LIBRARY — ENSEMBLE LEARNING DEMONSTRATION")
    print("=" * 74)

    # Print model catalog summary
    print("\n--- Model Library Summary ---")
    print(summary())

    # 1. Dataset Generation: Multi-feature non-linear target with interaction
    print("--- 1. Generating Synthetic Non-Linear Dataset ---")
    np.random.seed(42)
    n_samples = 250
    n_features = 5

    X_raw = np.random.uniform(-2.5, 2.5, size=(n_samples, n_features))
    # y = 1.5 * sin(x0) + x1^2 - 0.8 * x2 + noise
    y_raw = (
        1.5 * np.sin(X_raw[:, 0])
        + 0.5 * (X_raw[:, 1] ** 2)
        - 0.8 * X_raw[:, 2]
        + np.random.normal(0.0, 0.15, size=(n_samples,))
    )

    split = int(0.8 * n_samples)
    X_train, X_test = X_raw[:split], X_raw[split:]
    y_train, y_test = y_raw[:split], y_raw[split:]

    print(f"Generated {n_samples} samples across {n_features} features.")
    print(f"Train split: {len(X_train)} samples | Test split: {len(X_test)} samples")
    print(f"Target function: y = 1.5*sin(x0) + 0.5*(x1^2) - 0.8*x2 + noise")

    # ---------------------------------------------------------
    # 2. Random Forest Regressor
    # ---------------------------------------------------------
    print("\n--- 2. Random Forest Regressor (Bagging + Feature Subspaces) ---")
    rf = RandomForestRegressor(
        n_estimators=35,
        max_depth=7,
        max_features="sqrt",
        bootstrap=True,
        oob_score=True,
        random_state=42,
    )
    rf.fit(X_train, y_train)

    rf_train_r2 = rf.evaluate(X_train, y_train, metric="r2")
    rf_test_r2 = rf.evaluate(X_test, y_test, metric="r2")
    rf_test_mse = rf.evaluate(X_test, y_test, metric="mse")

    print(f"Trees in Forest:     {len(rf)}")
    print(f"Train R^2 Score:     {rf_train_r2:.4f}")
    print(f"Test R^2 Score:      {rf_test_r2:.4f}")
    print(f"Test MSE:            {rf_test_mse:.4f}")
    print(f"Out-of-Bag (OOB) R^2:{rf.oob_score_:.4f}")
    print(f"Feature Importances: {[round(float(v), 4) for v in rf.feature_importances()]}")

    # ---------------------------------------------------------
    # 3. Gradient Boosting Regressor
    # ---------------------------------------------------------
    print("\n--- 3. Gradient Boosting Regressor (Stage-wise Boosting) ---")
    gbdt = GradientBoostingRegressor(
        n_estimators=45,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.85,
        random_state=42,
    )
    gbdt.fit(X_train, y_train)

    gbdt_train_r2 = gbdt.evaluate(X_train, y_train, metric="r2")
    gbdt_test_r2 = gbdt.evaluate(X_test, y_test, metric="r2")
    gbdt_test_mse = gbdt.evaluate(X_test, y_test, metric="mse")

    print(f"Boosting Stages:     {len(gbdt)}")
    print(f"Base Prediction F_0: {gbdt.init_value_:.4f}")
    print(f"Initial Stage Loss:  {gbdt.train_score_[0]:.4f}")
    print(f"Final Stage Loss:    {gbdt.train_score_[-1]:.4f}")
    print(f"Train R^2 Score:     {gbdt_train_r2:.4f}")
    print(f"Test R^2 Score:      {gbdt_test_r2:.4f}")
    print(f"Test MSE:            {gbdt_test_mse:.4f}")
    print(f"Feature Importances: {[round(float(v), 4) for v in gbdt.feature_importances()]}")

    # ---------------------------------------------------------
    # 4. Voting Regressor
    # ---------------------------------------------------------
    print("\n--- 4. Voting Regressor (Heterogeneous Model Averaging) ---")
    ridge = RidgeRegression(in_features=n_features, alpha=1.0)
    svr = SVR(in_features=n_features, epsilon=0.1, C=1.0)
    tree = DecisionTreeRegressor(max_depth=5, min_samples_leaf=2)

    voting = VotingRegressor(
        estimators=[("ridge", ridge), ("svr", svr), ("tree", tree)],
        weights=[0.15, 0.25, 0.60],
    )
    voting.fit(X_train, y_train.reshape((-1, 1)), epochs=80, lr=0.01, verbose=False)

    voting_train_r2 = voting.evaluate(X_train, y_train, metric="r2")
    voting_test_r2 = voting.evaluate(X_test, y_test, metric="r2")
    voting_test_mse = voting.evaluate(X_test, y_test, metric="mse")

    print(f"Constituent Models:  {[name for name, _ in voting.named_estimators]}")
    print(f"Voting Weights:      {voting.weights_}")
    print(f"Train R^2 Score:     {voting_train_r2:.4f}")
    print(f"Test R^2 Score:      {voting_test_r2:.4f}")
    print(f"Test MSE:            {voting_test_mse:.4f}")

    print("\n" + "=" * 74)
    print("All ensemble models trained, evaluated, and verified successfully.")
    print("=" * 74)


if __name__ == "__main__":
    main()
