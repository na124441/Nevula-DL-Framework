"""
Example: Comprehensive Evaluation Metrics in Nevula
====================================================

Demonstrates calculating:
1. Regression metrics: MAE, MSE, RMSE, R^2 score.
2. Classification metrics: Confusion Matrix, Accuracy, Precision, Recall, F1 Score, ROC Curve, and ROC-AUC.
"""

import os
import sys

# Ensure repository root is on sys.path when script is executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import nevula as nv
from nevula.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
)


def run_regression_demo():
    print("=" * 60)
    print("1. Regression Evaluation Metrics")
    print("=" * 60)

    np.random.seed(42)
    y_true = np.array([12.5, 14.0, 15.2, 18.0, 22.1, 25.0])
    y_pred = np.array([12.0, 14.5, 15.0, 17.5, 23.0, 24.2])

    print(f"Ground Truth: {y_true}")
    print(f"Predictions:  {y_pred}\n")

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"MAE  (Mean Absolute Error):      {mae:.4f}")
    print(f"MSE  (Mean Squared Error):       {mse:.4f}")
    print(f"RMSE (Root Mean Squared Error):  {rmse:.4f}")
    print(f"R^2  (Coefficient of Determ.):   {r2:.4f}")


def run_classification_demo():
    print("\n" + "=" * 60)
    print("2. Classification Evaluation Metrics")
    print("=" * 60)

    # Multi-class scenario (3 classes)
    y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 1, 0, 1, 2])
    y_probs = np.array([
        [0.85, 0.10, 0.05],
        [0.40, 0.50, 0.10],
        [0.05, 0.90, 0.05],
        [0.10, 0.80, 0.10],
        [0.05, 0.15, 0.80],
        [0.10, 0.60, 0.30],
        [0.90, 0.05, 0.05],
        [0.15, 0.80, 0.05],
        [0.05, 0.05, 0.90],
    ])

    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    p_macro = precision_score(y_true, y_pred, average="macro")
    r_macro = recall_score(y_true, y_pred, average="macro")
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    auc_score = roc_auc_score(y_true, y_probs, average="macro")

    print(f"Confusion Matrix (3x3):\n{cm}\n")
    print(f"Accuracy:         {acc * 100:.2f}%")
    print(f"Macro Precision:  {p_macro:.4f}")
    print(f"Macro Recall:     {r_macro:.4f}")
    print(f"Macro F1 Score:   {f1_macro:.4f}")
    print(f"Weighted F1:      {f1_weighted:.4f}")
    print(f"Macro ROC-AUC:    {auc_score:.4f}")

    # Binary ROC Curve inspection
    print("\n--- Binary ROC Curve Inspection ---")
    y_bin_true = np.array([0, 0, 1, 1])
    y_bin_scores = np.array([0.15, 0.35, 0.75, 0.90])
    fpr, tpr, thresh = roc_curve(y_bin_true, y_bin_scores)
    bin_auc = roc_auc_score(y_bin_true, y_bin_scores)

    for i in range(len(fpr)):
        print(f"Step {i}: Threshold={thresh[i]:.2f} | FPR={fpr[i]:.2f} | TPR={tpr[i]:.2f}")
    print(f"Binary ROC-AUC:   {bin_auc:.4f}")


def main():
    run_regression_demo()
    run_classification_demo()


if __name__ == "__main__":
    main()
