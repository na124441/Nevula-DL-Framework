"""
Nevula Example: Support Vector Classification (SVC / SVM) Demonstration.

Demonstrates:
  1. Maximum margin hyperplane optimization with Hinge Loss.
  2. Support vector identification, margin width (2 / ||w||), and support vector ratio.
  3. Binary classification with performance metrics (accuracy, precision, recall, F1).
  4. Multiclass classification using One-vs-Rest (OvR) hyperplanes.
  5. Checkpointing and model serialization.
"""

import os
import sys

# Ensure repository root is on sys.path when script is executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import nevula as nv
from nevula.models import SVC, SVM, summary
from nevula.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def main():
    print("=" * 76)
    print("      NEVULA MODEL LIBRARY — SUPPORT VECTOR CLASSIFIER (SVC / SVM)")
    print("=" * 76)

    # 0. Display Model Library catalog
    print("\n--- 0. Model Library Catalog ---")
    print(summary())

    np.random.seed(42)

    # -----------------------------------------------------------------
    # 1. Binary Classification: Maximum Margin Separating Hyperplane
    # -----------------------------------------------------------------
    print("\n--- 1. Binary Classification: Maximum Margin Separating Hyperplane ---")
    n_samples = 150
    # Class 0 centered at (-1.8, -1.5)
    c0 = np.random.normal(loc=[-1.8, -1.5], scale=0.75, size=(n_samples, 2))
    # Class 1 centered at (1.8, 1.5)
    c1 = np.random.normal(loc=[1.8, 1.5], scale=0.75, size=(n_samples, 2))

    X_raw = np.vstack([c0, c1])
    y_raw = np.array([0] * n_samples + [1] * n_samples)

    perm = np.random.permutation(len(X_raw))
    X_raw, y_raw = X_raw[perm], y_raw[perm]

    split = int(0.75 * len(X_raw))
    X_train, X_test = X_raw[:split], X_raw[split:]
    y_train, y_test = y_raw[:split], y_raw[split:]

    print(f"Dataset: {len(X_raw)} samples across 2 features (Train: {len(X_train)}, Test: {len(X_test)})")

    # Instantiate SVC with C=5.0 and Hinge Loss
    svc = SVC(in_features=2, n_classes=2, C=5.0, loss="hinge")
    print(f"Model initialized: {svc}")

    # Fit using Adam optimizer
    svc.fit(X_train, y_train, epochs=120, lr=0.06, optimizer="adam")

    # Compute predictions and metrics
    train_acc = svc.evaluate(X_train, y_train, metric="accuracy")
    test_acc = svc.evaluate(X_test, y_test, metric="accuracy")
    test_prec = svc.evaluate(X_test, y_test, metric="precision")
    test_rec = svc.evaluate(X_test, y_test, metric="recall")
    test_f1 = svc.evaluate(X_test, y_test, metric="f1")

    # Inspect learned margin geometry and support vectors
    w = svc.coef_.ravel()
    b = float(svc.intercept_[0])
    w_norm = float(np.linalg.norm(w))
    margin_width = 2.0 / w_norm if w_norm > 0 else 0.0

    n_sv = svc.n_support_(X_train, y_train)
    sv_ratio = svc.support_vectors_ratio(X_train, y_train)
    sv_coords = svc.support_vectors(X_train, y_train)

    print(f"\nLearned Separating Hyperplane : {w[0]:+.4f}*x0 + {w[1]:+.4f}*x1 {b:+.4f} = 0")
    print(f"Weight Vector Norm ||w||_2    : {w_norm:.4f}")
    print(f"Geometric Margin (2 / ||w||)  : {margin_width:.4f}")
    print(f"Support Vectors Count         : {n_sv} / {len(X_train)} ({sv_ratio * 100:.1f}%)")
    print(f"Sample Support Vector Points  :\n{sv_coords[:3]}")

    print(f"\nTrain Accuracy                : {train_acc * 100:.2f}%")
    print(f"Test Accuracy                 : {test_acc * 100:.2f}%")
    print(f"Test Precision                : {test_prec:.4f}")
    print(f"Test Recall                   : {test_rec:.4f}")
    print(f"Test F1 Score                 : {test_f1:.4f}")

    # -----------------------------------------------------------------
    # 2. Multiclass Classification: 3-Class One-vs-Rest (OvR)
    # -----------------------------------------------------------------
    print("\n--- 2. Multiclass Classification: 3-Class One-vs-Rest (OvR) ---")
    n_multi = 100
    # 3 distinct clusters in 2D
    cl0 = np.random.normal(loc=[-2.5, -1.0], scale=0.7, size=(n_multi, 2))
    cl1 = np.random.normal(loc=[0.0, 2.5], scale=0.7, size=(n_multi, 2))
    cl2 = np.random.normal(loc=[2.5, -1.0], scale=0.7, size=(n_multi, 2))

    X_m = np.vstack([cl0, cl1, cl2])
    y_m = np.array([0] * n_multi + [1] * n_multi + [2] * n_multi)

    perm_m = np.random.permutation(len(X_m))
    X_m, y_m = X_m[perm_m], y_m[perm_m]

    split_m = int(0.75 * len(X_m))
    X_m_train, X_m_test = X_m[:split_m], X_m[split_m:]
    y_m_train, y_m_test = y_m[:split_m], y_m[split_m:]

    multi_svm = SVM(in_features=2, n_classes=3, C=2.0, loss="hinge")
    multi_svm.fit(X_m_train, y_m_train, epochs=120, lr=0.06, optimizer="adam")

    m_acc = multi_svm.evaluate(X_m_test, y_m_test, metric="accuracy")
    m_f1 = multi_svm.evaluate(X_m_test, y_m_test, metric="f1")

    preds_m = multi_svm.predict(X_m_test)
    preds_m_np = np.array(preds_m.to_list()).ravel()
    cm = confusion_matrix(y_m_test, preds_m_np)

    print(f"Multiclass Model Configuration: {multi_svm.get_config()}")
    print(f"Learned OvR Coefficient Matrix Shape: {multi_svm.coef_.shape}")
    print(f"Multiclass Test Accuracy      : {m_acc * 100:.2f}%")
    print(f"Macro-averaged F1 Score       : {m_f1:.4f}")
    print("\nConfusion Matrix (Rows=True, Columns=Predicted):")
    print(cm)

    # -----------------------------------------------------------------
    # 3. Model Checkpointing & Serialization
    # -----------------------------------------------------------------
    print("\n--- 3. Checkpointing & Model Serialization ---")
    save_path = "scratch/svm_checkpoint.nv"
    os.makedirs("scratch", exist_ok=True)
    nv.save(svc.state_dict(), save_path)
    print(f"Saved binary SVC state dictionary to: {save_path}")

    # Instantiate fresh model and load weights
    loaded_svc = SVC(in_features=2, n_classes=2, C=5.0)
    loaded_svc.classes_ = svc.classes_
    loaded_svc.load_state_dict(nv.load(save_path))
    loaded_svc._is_fitted = True

    loaded_acc = loaded_svc.evaluate(X_test, y_test, metric="accuracy")
    print(f"Restored Model Test Accuracy  : {loaded_acc * 100:.2f}% (Matches original: {loaded_acc == test_acc})")

    # Clean up scratch checkpoint
    if os.path.exists(save_path):
        os.remove(save_path)

    print("\n" + "=" * 76)
    print("      ALL SVM CHECKS AND DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 76)


if __name__ == "__main__":
    main()
