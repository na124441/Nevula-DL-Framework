"""
Nevula Example: Logistic Regression & Sigmoid Classification Demonstration.

Demonstrates:
  1. Binary classification using the Sigmoid link function and BCEWithLogitsLoss.
  2. Confusion matrix, precision, recall, and F1 score evaluation.
  3. Multiclass classification using Softmax regression.
  4. Inspecting predicted probabilities and learned linear decision boundaries.
"""

import os
import sys

# Ensure repository root is on sys.path when script is executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import nevula as nv
from nevula.models import LogisticRegression, summary
from nevula.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def main():
    print("=" * 74)
    print("      NEVULA MODEL LIBRARY — LOGISTIC REGRESSION DEMONSTRATION")
    print("=" * 74)

    # 0. Display Model Library catalog
    print("\n--- Model Library Catalog ---")
    print(summary())

    np.random.seed(42)

    # -------------------------------------------------------------
    # 1. Binary Classification: Decision Boundary Recovery
    # -------------------------------------------------------------
    print("--- 1. Binary Classification (Sigmoid + BCEWithLogits) ---")
    n_samples = 120
    # Class 0: centered at (-1.5, -1.0)
    c0 = np.random.normal(loc=[-1.5, -1.0], scale=0.7, size=(n_samples, 2))
    # Class 1: centered at (1.5, 1.0)
    c1 = np.random.normal(loc=[1.5, 1.0], scale=0.7, size=(n_samples, 2))

    X_raw = np.vstack([c0, c1])
    y_raw = np.array([0] * n_samples + [1] * n_samples)

    perm = np.random.permutation(len(X_raw))
    X_raw, y_raw = X_raw[perm], y_raw[perm]

    split = int(0.75 * len(X_raw))
    X_train, X_test = X_raw[:split], X_raw[split:]
    y_train, y_test = y_raw[:split], y_raw[split:]

    print(f"Generated {len(X_raw)} binary samples (Train: {len(X_train)}, Test: {len(X_test)})")

    # Instantiate and fit binary model
    clf_bin = LogisticRegression(in_features=2, n_classes=2, penalty="l2", C=1.0)
    clf_bin.fit(X_train, y_train, epochs=120, lr=0.1, verbose=False)

    train_acc = clf_bin.evaluate(X_train, y_train, metric="accuracy")
    test_acc = clf_bin.evaluate(X_test, y_test, metric="accuracy")
    test_prec = clf_bin.evaluate(X_test, y_test, metric="precision")
    test_rec = clf_bin.evaluate(X_test, y_test, metric="recall")
    test_f1 = clf_bin.evaluate(X_test, y_test, metric="f1")

    print(f"Train Accuracy:      {train_acc * 100:.2f}%")
    print(f"Test Accuracy:       {test_acc * 100:.2f}%")
    print(f"Test Precision:      {test_prec:.4f}")
    print(f"Test Recall:         {test_rec:.4f}")
    print(f"Test F1 Score:       {test_f1:.4f}")
    print(f"Learned Weights:     {clf_bin.coef_.ravel()}")
    print(f"Learned Bias:        {clf_bin.intercept_}")

    # Confusion matrix
    preds = clf_bin.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    print(f"Confusion Matrix:\n{cm}")

    # Probability calibration on first 4 test samples
    probs = clf_bin.predict_proba(X_test[:4])
    print(f"\nSample Predicted Probabilities [P(y=0), P(y=1)]:")
    for i, p in enumerate(probs.to_list()):
        true_label = int(y_test[i])
        print(f"  Sample {i}: P(y=0)={p[0]:.4f}, P(y=1)={p[1]:.4f} | True Class: {true_label}")

    # -------------------------------------------------------------
    # 2. Multiclass Classification: 3-Class Softmax Regression
    # -------------------------------------------------------------
    print("\n--- 2. Multiclass Classification (Softmax Regression) ---")
    n_per_class = 80
    c0 = np.random.normal(loc=[-2.0, 0.0], scale=0.6, size=(n_per_class, 2))
    c1 = np.random.normal(loc=[2.0, 0.0], scale=0.6, size=(n_per_class, 2))
    c2 = np.random.normal(loc=[0.0, 2.5], scale=0.6, size=(n_per_class, 2))

    X_m = np.vstack([c0, c1, c2])
    y_m = np.array([0] * n_per_class + [1] * n_per_class + [2] * n_per_class)

    m_perm = np.random.permutation(len(X_m))
    X_m, y_m = X_m[m_perm], y_m[m_perm]

    clf_multi = LogisticRegression(in_features=2, n_classes=3, penalty="l2", C=1.0)
    clf_multi.fit(X_m, y_m, epochs=150, lr=0.1, verbose=False)

    m_acc = clf_multi.evaluate(X_m, y_m, metric="accuracy")
    m_f1 = clf_multi.evaluate(X_m, y_m, metric="f1")

    print(f"3-Class Dataset:     {len(X_m)} samples across 3 classes")
    print(f"Multiclass Accuracy: {m_acc * 100:.2f}%")
    print(f"Macro F1 Score:      {m_f1:.4f}")

    print("\n" + "=" * 74)
    print(" Logistic Regression demonstration completed successfully.")
    print("=" * 74)


if __name__ == "__main__":
    main()
