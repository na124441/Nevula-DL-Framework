"""
Nevula Cross-Validation & Model Selection End-to-End Walkthrough.

Demonstrates:
    1. Stratified train/test splitting preserving class ratios.
    2. K-Fold and Stratified K-Fold cross-validation evaluation.
    3. Multi-metric evaluation with cross_validate.
    4. Out-of-fold prediction generation with cross_val_predict.
    5. Hyperparameter optimization via GridSearchCV and RandomizedSearchCV.
    6. Model diagnostic analysis with learning curves.
"""

import os
import sys

# Ensure repository root is on sys.path when script is executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import nevula as nv
from nevula.core.tensor import Tensor
from nevula.models.regression.ridge import RidgeRegression
from nevula.models.classification.logistic import LogisticRegression
from nevula.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
    cross_val_predict,
    GridSearchCV,
    RandomizedSearchCV,
    learning_curve,
)


def run_cross_validation_demo():
    print("=" * 70)
    print("        NEVULA CROSS-VALIDATION & MODEL SELECTION SUITE")
    print("=" * 70)

    np.random.seed(42)

    # ---------------------------------------------------------
    # 1. Dataset Generation: Classification & Regression
    # ---------------------------------------------------------
    print("\n[Step 1] Preparing Datasets...")
    n_samples = 120

    # Classification data: 2 clusters with slight overlap
    c0 = np.random.randn(60, 2) * 0.7 + np.array([-1.2, -1.2])
    c1 = np.random.randn(60, 2) * 0.7 + np.array([1.2, 1.2])
    X_cls = Tensor(np.vstack([c0, c1]))
    y_cls = Tensor(np.array([0.0] * 60 + [1.0] * 60))

    # Regression data: y = 2.5 * x0 - 1.8 * x1 + 0.5 + noise
    X_reg = Tensor(np.random.uniform(-2.0, 2.0, size=(n_samples, 2)))
    y_reg = 2.5 * X_reg[:, 0] - 1.8 * X_reg[:, 1] + 0.5 + Tensor(np.random.normal(0, 0.1, n_samples))

    print(f"  Classification: X={X_cls.shape}, y={y_cls.shape}")
    print(f"  Regression:     X={X_reg.shape}, y={y_reg.shape}")

    # ---------------------------------------------------------
    # 2. Train-Test Splitting
    # ---------------------------------------------------------
    print("\n[Step 2] Stratified Train-Test Split (75% train / 25% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_cls, y_cls, test_size=0.25, random_state=42, stratify=y_cls
    )
    y_tr_np = y_train.numpy().ravel()
    y_te_np = y_test.numpy().ravel()
    print(f"  Train samples: {len(X_train)} (class 0: {np.sum(y_tr_np == 0)}, class 1: {np.sum(y_tr_np == 1)})")
    print(f"  Test samples:  {len(X_test)}  (class 0: {np.sum(y_te_np == 0)}, class 1: {np.sum(y_te_np == 1)})")

    # ---------------------------------------------------------
    # 3. K-Fold Cross-Validation Scores
    # ---------------------------------------------------------
    print("\n[Step 3] 5-Fold Cross-Validation with cross_val_score...")
    ridge = RidgeRegression(in_features=2, out_features=1, alpha=0.1)
    r2_scores = cross_val_score(ridge, X_reg, y_reg, cv=5, scoring="r2")
    mse_scores = cross_val_score(ridge, X_reg, y_reg, cv=5, scoring="neg_mean_squared_error")

    print(f"  Ridge Regression R2 Scores:  {np.round(r2_scores, 4)}")
    print(f"  Mean R2: {np.mean(r2_scores):.4f} (+/- {np.std(r2_scores):.4f})")
    print(f"  Mean MSE: {-np.mean(mse_scores):.4f}")

    # ---------------------------------------------------------
    # 4. Multi-Metric Evaluation with cross_validate
    # ---------------------------------------------------------
    print("\n[Step 4] Multi-Metric Cross-Validation with cross_validate...")
    clf = LogisticRegression(in_features=2, n_classes=2, C=1.0)
    cv_metrics = cross_validate(
        clf,
        X_cls,
        y_cls,
        cv=4,
        scoring={"accuracy": "accuracy", "f1": "f1", "precision": "precision", "recall": "recall"},
        return_train_score=True,
        fit_params={"epochs": 50, "lr": 0.1, "verbose": False},
    )

    print(f"  Test Accuracy:  {np.mean(cv_metrics['test_accuracy']):.4f} (+/- {np.std(cv_metrics['test_accuracy']):.4f})")
    print(f"  Test F1 Score:  {np.mean(cv_metrics['test_f1']):.4f} (+/- {np.std(cv_metrics['test_f1']):.4f})")
    print(f"  Test Precision: {np.mean(cv_metrics['test_precision']):.4f}")
    print(f"  Test Recall:    {np.mean(cv_metrics['test_recall']):.4f}")
    print(f"  Mean Fit Time:  {np.mean(cv_metrics['fit_time']) * 1000:.2f} ms")

    # ---------------------------------------------------------
    # 5. Out-of-Fold Predictions
    # ---------------------------------------------------------
    print("\n[Step 5] Out-of-Fold Predictions with cross_val_predict...")
    oof_preds = cross_val_predict(
        clf,
        X_cls,
        y_cls,
        cv=4,
        method="predict",
        fit_params={"epochs": 50, "lr": 0.1, "verbose": False},
    )
    oof_accuracy = np.mean(oof_preds.numpy().ravel() == y_cls.numpy().ravel())
    print(f"  Out-of-Fold Global Accuracy: {oof_accuracy:.4f}")

    # ---------------------------------------------------------
    # 6. Hyperparameter Optimization: GridSearchCV
    # ---------------------------------------------------------
    print("\n[Step 6] Hyperparameter Optimization via GridSearchCV...")
    param_grid = {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0]}
    grid_search = GridSearchCV(
        estimator=RidgeRegression(in_features=2, out_features=1),
        param_grid=param_grid,
        cv=4,
        scoring="r2",
        refit=True,
        verbose=1,
    )
    grid_search.fit(X_reg, y_reg)
    print(f"  Best Parameters: {grid_search.best_params_}")
    print(f"  Best CV R2:      {grid_search.best_score_:.4f}")

    # ---------------------------------------------------------
    # 7. Diagnostic Learning Curve
    # ---------------------------------------------------------
    print("\n[Step 7] Computing Learning Curve...")
    train_sizes, train_scores, test_scores = learning_curve(
        ridge,
        X_reg,
        y_reg,
        train_sizes=[0.25, 0.5, 0.75, 1.0],
        cv=3,
        scoring="r2",
    )
    for n_tr, tr_sc, te_sc in zip(train_sizes, np.mean(train_scores, axis=1), np.mean(test_scores, axis=1)):
        print(f"  Samples: {n_tr:3d} | Train R2: {tr_sc:.4f} | Validation R2: {te_sc:.4f}")

    print("\n" + "=" * 70)
    print("      CROSS-VALIDATION WALKTHROUGH COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_cross_validation_demo()
