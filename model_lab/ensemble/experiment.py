"""
Reproducible Experiment Runner for Ensemble Learning in Nevula.
Benchmarks DecisionTreeRegressor vs. RandomForestRegressor vs. GradientBoostingRegressor vs. VotingRegressor.
"""

import os
import sys

# Ensure repository root is on sys.path when script is executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
from nevula.models.trees.decision_tree import DecisionTreeRegressor
from nevula.models.ensemble.random_forest import RandomForestRegressor
from nevula.models.ensemble.gradient_boosting import GradientBoostingRegressor
from nevula.models.ensemble.voting import VotingRegressor
from nevula.models.regression.ridge import RidgeRegression
from nevula.models.regression.svr import SVR


def run_ensemble_benchmark():
    print("=" * 78)
    print(" ENSEMBLE BENCHMARK: DECISION TREE vs RANDOM FOREST vs GRADIENT BOOSTING vs VOTING")
    print("=" * 78)

    np.random.seed(42)
    n_samples = 300
    n_features = 6

    # Generate synthetic non-linear feature data
    X = np.random.uniform(-3.0, 3.0, size=(n_samples, n_features))
    # Target: non-linear interaction + noise
    # y = sin(x0) * cos(x1) + 0.25 * x2^2 - 0.5 * x3 + noise
    y = (
        np.sin(X[:, 0]) * np.cos(X[:, 1])
        + 0.25 * (X[:, 2] ** 2)
        - 0.5 * X[:, 3]
        + np.random.normal(0.0, 0.2, size=(n_samples,))
    )

    # Train / Test split (80/20)
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Dataset: {n_samples} samples, {n_features} features (Train: {len(X_train)}, Test: {len(X_test)})")
    print(f"Target function: y = sin(x0)*cos(x1) + 0.25*(x2^2) - 0.5*x3 + N(0, 0.04)")
    print("-" * 78)

    # 1. Single Decision Tree (unpruned / high depth -> prone to variance)
    single_tree = DecisionTreeRegressor(max_depth=10, min_samples_leaf=1)
    single_tree.fit(X_train, y_train)
    dt_train_r2 = single_tree.evaluate(X_train, y_train, metric="r2")
    dt_test_r2 = single_tree.evaluate(X_test, y_test, metric="r2")
    dt_test_mse = single_tree.evaluate(X_test, y_test, metric="mse")

    # 2. Random Forest Regressor (Bagging + Feature Subspaces)
    rf = RandomForestRegressor(
        n_estimators=40,
        max_depth=8,
        max_features="sqrt",
        bootstrap=True,
        oob_score=True,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    rf_train_r2 = rf.evaluate(X_train, y_train, metric="r2")
    rf_test_r2 = rf.evaluate(X_test, y_test, metric="r2")
    rf_test_mse = rf.evaluate(X_test, y_test, metric="mse")

    # 3. Gradient Boosting Regressor (Stage-wise Pseudo-Residual Boosting)
    gbdt = GradientBoostingRegressor(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.9,
        random_state=42,
    )
    gbdt.fit(X_train, y_train)
    gbdt_train_r2 = gbdt.evaluate(X_train, y_train, metric="r2")
    gbdt_test_r2 = gbdt.evaluate(X_test, y_test, metric="r2")
    gbdt_test_mse = gbdt.evaluate(X_test, y_test, metric="mse")

    # 4. Voting Regressor (Heterogeneous Combination: Ridge + SVR + Decision Tree)
    ridge = RidgeRegression(in_features=n_features, out_features=1, alpha=1.0)
    svr = SVR(in_features=n_features, epsilon=0.1, C=1.0)
    tree_shallow = DecisionTreeRegressor(max_depth=5, min_samples_leaf=2)
    voting = VotingRegressor(
        estimators=[("ridge", ridge), ("svr", svr), ("tree", tree_shallow)],
        weights=[0.15, 0.25, 0.60],
    )
    voting.fit(X_train, y_train.reshape((-1, 1)), epochs=80, lr=0.01, verbose=False)
    voting_train_r2 = voting.evaluate(X_train, y_train, metric="r2")
    voting_test_r2 = voting.evaluate(X_test, y_test, metric="r2")
    voting_test_mse = voting.evaluate(X_test, y_test, metric="mse")

    print(f"{'Model Architecture':<28} | {'Train R^2':<11} | {'Test R^2':<11} | {'Test MSE':<10} | {'Notes'}")
    print("-" * 78)
    print(
        f"{'Single DecisionTree (d=10)':<28} | {dt_train_r2:<11.4f} | {dt_test_r2:<11.4f} | {dt_test_mse:<10.4f} | High Variance / Overfitting"
    )
    print(
        f"{'Random Forest (40 trees)':<28} | {rf_train_r2:<11.4f} | {rf_test_r2:<11.4f} | {rf_test_mse:<10.4f} | Variance reduced (OOB R^2: {rf.oob_score_:.3f})"
    )
    print(
        f"{'Gradient Boosting (50 st.)':<28} | {gbdt_train_r2:<11.4f} | {gbdt_test_r2:<11.4f} | {gbdt_test_mse:<10.4f} | Stage-wise residual reduction"
    )
    print(
        f"{'Voting (Ridge+SVR+Tree)':<28} | {voting_train_r2:<11.4f} | {voting_test_r2:<11.4f} | {voting_test_mse:<10.4f} | Heterogeneous ensemble blend"
    )
    print("-" * 78)

    print("\nFeature Importances (Top 4 Features):")
    rf_imp = rf.feature_importances()
    gbdt_imp = gbdt.feature_importances()
    for f in range(4):
        print(f"  Feature x{f}: RF Importance = {rf_imp[f]:.4f} | GBDT Importance = {gbdt_imp[f]:.4f}")

    print("=" * 78)


if __name__ == "__main__":
    run_ensemble_benchmark()
