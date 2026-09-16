"""
Reproducible Experiment Runner for Logistic Regression in Nevula.
Benchmarks binary and multiclass decision boundary learning, loss convergence,
and L1/L2 regularization effects.
"""

import os
import sys

# Ensure repository root is on sys.path when script is executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
from nevula.models.classification.logistic import LogisticRegression
from nevula.metrics.classification import confusion_matrix


def run_logistic_benchmark():
    print("=" * 76)
    print("      NEVULA MODEL LAB — LOGISTIC REGRESSION EXPERIMENTAL BENCHMARK")
    print("=" * 76)

    np.random.seed(42)

    # -------------------------------------------------------------
    # 1. Binary Classification Benchmark
    # -------------------------------------------------------------
    print("\n--- Experiment 1: Binary Decision Boundary Recovery ---")
    n_per_cluster = 100
    # Cluster 0: Centered at (-1.5, -1.5)
    c0 = np.random.normal(loc=[-1.5, -1.5], scale=0.8, size=(n_per_cluster, 2))
    # Cluster 1: Centered at (1.5, 1.5)
    c1 = np.random.normal(loc=[1.5, 1.5], scale=0.8, size=(n_per_cluster, 2))

    X_bin = np.vstack([c0, c1])
    y_bin = np.array([0] * n_per_cluster + [1] * n_per_cluster)

    # Shuffle
    indices = np.random.permutation(len(X_bin))
    X_bin, y_bin = X_bin[indices], y_bin[indices]

    split = int(0.75 * len(X_bin))
    X_tr, X_te = X_bin[:split], X_bin[split:]
    y_tr, y_te = y_bin[:split], y_bin[split:]

    clf_bin = LogisticRegression(in_features=2, n_classes=2, penalty="l2", C=1.0)
    clf_bin.fit(X_tr, y_tr, epochs=120, lr=0.1, verbose=False)

    train_acc = clf_bin.evaluate(X_tr, y_tr, metric="accuracy")
    test_acc = clf_bin.evaluate(X_te, y_te, metric="accuracy")
    test_f1 = clf_bin.evaluate(X_te, y_te, metric="f1")
    test_bce = clf_bin.evaluate(X_te, y_te, metric="bce")

    print(f"Dataset: 2 Gaussian clusters (Train: {len(X_tr)}, Test: {len(X_te)})")
    print(f"Learned Weights:   {clf_bin.coef_.ravel()}")
    print(f"Learned Intercept: {clf_bin.intercept_}")
    print(f"Train Accuracy:    {train_acc * 100:.2f}%")
    print(f"Test Accuracy:     {test_acc * 100:.2f}%")
    print(f"Test F1 Score:     {test_f1:.4f}")
    print(f"Test BCE Loss:     {test_bce:.4f}")

    preds_te = clf_bin.predict(X_te)
    cm = confusion_matrix(y_te, preds_te)
    print("Confusion Matrix:")
    print(f"  [[TN={cm[0, 0]}, FP={cm[0, 1]}], [FN={cm[1, 0]}, TP={cm[1, 1]}]]")

    # -------------------------------------------------------------
    # 2. Multiclass (Softmax) Benchmark
    # -------------------------------------------------------------
    print("\n--- Experiment 2: Multiclass (3-Class Softmax) Benchmark ---")
    n_multi = 80
    c0 = np.random.normal(loc=[-2.0, 0.0], scale=0.6, size=(n_multi, 2))
    c1 = np.random.normal(loc=[2.0, 0.0], scale=0.6, size=(n_multi, 2))
    c2 = np.random.normal(loc=[0.0, 3.0], scale=0.6, size=(n_multi, 2))

    X_multi = np.vstack([c0, c1, c2])
    y_multi = np.array([0] * n_multi + [1] * n_multi + [2] * n_multi)

    m_indices = np.random.permutation(len(X_multi))
    X_multi, y_multi = X_multi[m_indices], y_multi[m_indices]

    clf_multi = LogisticRegression(in_features=2, n_classes=3, penalty="l2", C=1.0)
    clf_multi.fit(X_multi, y_multi, epochs=150, lr=0.1, verbose=False)

    multi_acc = clf_multi.evaluate(X_multi, y_multi, metric="accuracy")
    multi_f1 = clf_multi.evaluate(X_multi, y_multi, metric="f1")
    print(f"Multiclass Accuracy: {multi_acc * 100:.2f}%")
    print(f"Multiclass Macro F1: {multi_f1:.4f}")

    # Inspect predicted probabilities sum to 1.0
    probs_sample = clf_multi.predict_proba(X_multi[:3])
    print(f"Sample Softmax Probabilities:\n{np.round(probs_sample.to_list(), 3)}")

    # -------------------------------------------------------------
    # 3. L1 Sparsity Benchmark
    # -------------------------------------------------------------
    print("\n--- Experiment 3: L1 vs L2 Regularization on Sparse Features ---")
    n_sparse = 150
    n_features = 8
    X_sparse = np.random.normal(0, 1, size=(n_sparse, n_features))
    # True signal only depends on features 0 and 1
    y_sparse = (1.5 * X_sparse[:, 0] - 2.0 * X_sparse[:, 1] > 0).astype(int)

    clf_l1 = LogisticRegression(in_features=n_features, penalty="l1", C=0.5)
    clf_l1.fit(X_sparse, y_sparse, epochs=200, lr=0.08, verbose=False)

    clf_l2 = LogisticRegression(in_features=n_features, penalty="l2", C=0.5)
    clf_l2.fit(X_sparse, y_sparse, epochs=200, lr=0.08, verbose=False)

    print(f"True Signal: Depends only on features x0 and x1.")
    print(f"L1 Weights: {np.round(clf_l1.coef_.ravel(), 4)}")
    print(f"L2 Weights: {np.round(clf_l2.coef_.ravel(), 4)}")
    print("=" * 76)


if __name__ == "__main__":
    run_logistic_benchmark()
