"""
Example: Decision Tree Classification with Nevula
==================================================

Demonstrates training and evaluating CART DecisionTreeClassifier on multi-class
synthetic data, comparing Gini vs Entropy, inspecting probability distributions,
and computing feature importances.
"""

import os
import sys

# Ensure repository root is on sys.path when script is executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import nevula as nv
from nevula.models.trees import DecisionTreeClassifier


def main():
    print("=" * 60)
    print("Nevula: CART Decision Tree Classifier")
    print("=" * 60)

    # 1. Generate 3-class clustered dataset
    np.random.seed(42)
    n_samples_per_class = 60
    c0 = np.random.randn(n_samples_per_class, 2) * 0.6 + np.array([-2.5, -2.0])
    c1 = np.random.randn(n_samples_per_class, 2) * 0.6 + np.array([0.0, 2.5])
    c2 = np.random.randn(n_samples_per_class, 2) * 0.6 + np.array([2.5, -2.0])

    X = np.vstack([c0, c1, c2])
    y = np.array([0] * n_samples_per_class + [1] * n_samples_per_class + [2] * n_samples_per_class)

    # Shuffle dataset
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    # Split train/test (80/20)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Dataset: {len(X)} samples across 3 classes (Train: {len(X_train)}, Test: {len(X_test)})")

    # 2. Train with Gini criterion
    print("\n--- Training DecisionTreeClassifier (Gini Impurity) ---")
    clf_gini = DecisionTreeClassifier(criterion="gini", max_depth=4, min_samples_leaf=2)
    clf_gini.fit(X_train, y_train)

    train_acc = clf_gini.evaluate(X_train, y_train, metric="accuracy")
    test_acc = clf_gini.evaluate(X_test, y_test, metric="accuracy")
    test_f1 = clf_gini.evaluate(X_test, y_test, metric="f1")
    test_loss = clf_gini.evaluate(X_test, y_test, metric="log_loss")

    print(f"Tree Depth: {clf_gini.tree_depth()}")
    print(f"Number of Leaves: {clf_gini.n_leaves()}")
    print(f"Feature Importances: {clf_gini.feature_importances()}")
    print(f"Train Accuracy: {train_acc * 100:.2f}%")
    print(f"Test Accuracy:  {test_acc * 100:.2f}%")
    print(f"Test Macro F1:  {test_f1:.4f}")
    print(f"Test Log Loss:  {test_loss:.4f}")

    # 3. Train with Entropy criterion
    print("\n--- Training DecisionTreeClassifier (Entropy / Information Gain) ---")
    clf_entropy = DecisionTreeClassifier(criterion="entropy", max_depth=4, min_samples_leaf=2)
    clf_entropy.fit(X_train, y_train)

    test_acc_ent = clf_entropy.evaluate(X_test, y_test, metric="accuracy")
    print(f"Test Accuracy (Entropy): {test_acc_ent * 100:.2f}%")

    # 4. Show sample predictions and probabilities
    print("\n--- Sample Test Predictions & Class Probabilities ---")
    sample_X = X_test[:5]
    sample_y = y_test[:5]
    sample_preds = clf_gini.predict(sample_X)
    sample_probs = clf_gini.predict_proba(sample_X)

    preds_list = sample_preds.to_list()
    probs_list = sample_probs.to_list()

    for i in range(5):
        print(f"Sample {i+1}: True={sample_y[i]} | Pred={int(preds_list[i][0])} | Probs={np.round(probs_list[i], 3)}")

    print("\nDecision Tree Classification complete.")


if __name__ == "__main__":
    main()
