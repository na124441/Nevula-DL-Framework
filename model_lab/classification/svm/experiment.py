import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import numpy as np

from nevula.models.classification.svm import SVC, SVM
from nevula.metrics.classification import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def generate_separable_data(n_samples: int = 120, seed: int = 42):
    """Generates a clean, linearly separable 2D dataset."""
    np.random.seed(seed)
    n_per_class = n_samples // 2
    # Class 0 centered at (-2, -2)
    X0 = np.random.randn(n_per_class, 2) * 0.7 + np.array([-2.0, -2.0])
    y0 = np.zeros(n_per_class)
    # Class 1 centered at (2, 2)
    X1 = np.random.randn(n_per_class, 2) * 0.7 + np.array([2.0, 2.0])
    y1 = np.ones(n_per_class)

    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])
    indices = np.random.permutation(n_samples)
    return X[indices], y[indices]


def generate_noisy_data(n_samples: int = 200, noise: float = 1.2, seed: int = 42):
    """Generates non-separable 2D data with overlapping distributions."""
    np.random.seed(seed)
    n_per_class = n_samples // 2
    X0 = np.random.randn(n_per_class, 2) * noise + np.array([-1.0, -1.0])
    y0 = np.zeros(n_per_class)
    X1 = np.random.randn(n_per_class, 2) * noise + np.array([1.0, 1.0])
    y1 = np.ones(n_per_class)

    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])
    indices = np.random.permutation(n_samples)
    return X[indices], y[indices]


def generate_multiclass_data(n_samples: int = 300, seed: int = 42):
    """Generates a 3-class 2D dataset."""
    np.random.seed(seed)
    n_per_class = n_samples // 3
    # Class 0: bottom-left
    X0 = np.random.randn(n_per_class, 2) * 0.8 + np.array([-3.0, -1.5])
    y0 = np.zeros(n_per_class)
    # Class 1: top-middle
    X1 = np.random.randn(n_per_class, 2) * 0.8 + np.array([0.0, 3.0])
    y1 = np.ones(n_per_class)
    # Class 2: bottom-right
    X2 = np.random.randn(n_per_class, 2) * 0.8 + np.array([3.0, -1.5])
    y2 = np.full(n_per_class, 2)

    X = np.vstack([X0, X1, X2])
    y = np.concatenate([y0, y1, y2])
    indices = np.random.permutation(n_samples)
    return X[indices], y[indices]


def run_experiment_1_separable():
    print("=" * 70)
    print("Experiment 1: Linearly Separable Binary Classification")
    print("=" * 70)
    X, y = generate_separable_data(n_samples=160, seed=123)
    split = 120
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    model = SVC(in_features=2, n_classes=2, C=10.0, loss="hinge")
    model.fit(X_train, y_train, epochs=120, lr=0.08, optimizer="sgd")

    test_acc = model.evaluate(X_test, y_test, metric="accuracy")
    test_f1 = model.evaluate(X_test, y_test, metric="f1")
    n_sv = model.n_support_(X_train, y_train)
    sv_ratio = model.support_vectors_ratio(X_train, y_train)

    w = model.coef_.ravel()
    b = float(model.intercept_[0])
    w_norm = float(np.linalg.norm(w))
    margin_width = 2.0 / w_norm if w_norm > 0 else 0.0

    print(f"Hyperplane weights w     : [{w[0]:.4f}, {w[1]:.4f}]")
    print(f"Hyperplane intercept b   : {b:.4f}")
    print(f"Weight norm ||w||_2      : {w_norm:.4f}")
    print(f"Margin width (2 / ||w||) : {margin_width:.4f}")
    print(f"Training Support Vectors : {n_sv} / {len(X_train)} ({sv_ratio * 100:.1f}%)")
    print(f"Test Accuracy            : {test_acc * 100:.2f}%")
    print(f"Test F1 Score            : {test_f1:.4f}")
    assert test_acc >= 0.95, f"Expected high accuracy on separable data, got {test_acc}"
    print("[PASS] Experiment 1 successful!\n")


def run_experiment_2_regularization_sweep():
    print("=" * 70)
    print("Experiment 2: Regularization Parameter C Sensitivity Sweep")
    print("=" * 70)
    X, y = generate_noisy_data(n_samples=250, noise=1.1, seed=42)
    split = 180
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    c_values = [0.05, 0.5, 2.0, 10.0, 50.0]
    print(f"{'C Value':<10} | {'||w||_2':<10} | {'Margin Width':<14} | {'SV Count':<10} | {'SV Ratio':<10} | {'Test Acc':<10}")
    print("-" * 75)

    for c in c_values:
        clf = SVC(in_features=2, n_classes=2, C=c, loss="hinge")
        clf.fit(X_train, y_train, epochs=100, lr=0.05, optimizer="adam")

        w = clf.coef_.ravel()
        w_norm = float(np.linalg.norm(w))
        margin = 2.0 / w_norm if w_norm > 0 else 0.0
        n_sv = clf.n_support_(X_train, y_train)
        sv_ratio = clf.support_vectors_ratio(X_train, y_train)
        acc = clf.evaluate(X_test, y_test, metric="accuracy")

        print(f"{c:<10.2f} | {w_norm:<10.4f} | {margin:<14.4f} | {n_sv:<10d} | {sv_ratio * 100:<9.1f}% | {acc * 100:<9.2f}%")

    print("\nObservation: Smaller C enforces greater regularization (smaller ||w||, wider margin, more SVs).")
    print("[PASS] Experiment 2 completed!\n")


def run_experiment_3_loss_comparison():
    print("=" * 70)
    print("Experiment 3: Standard Hinge vs Squared Hinge Loss")
    print("=" * 70)
    X, y = generate_noisy_data(n_samples=200, noise=0.9, seed=77)
    split = 150
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    for loss_name in ["hinge", "squared_hinge"]:
        model = SVC(in_features=2, n_classes=2, C=1.0, loss=loss_name)
        model.fit(X_train, y_train, epochs=100, lr=0.05, optimizer="adam")
        acc = model.evaluate(X_test, y_test, metric="accuracy")
        f1 = model.evaluate(X_test, y_test, metric="f1")
        final_loss = model.loss_history[-1] if model.loss_history else 0.0

        print(f"Loss: {loss_name:<14} | Final Loss: {final_loss:.4f} | Test Acc: {acc * 100:.2f}% | Test F1: {f1:.4f}")

    print("[PASS] Experiment 3 completed!\n")


def run_experiment_4_multiclass():
    print("=" * 70)
    print("Experiment 4: Multiclass (3-Class) One-vs-Rest Benchmark")
    print("=" * 70)
    X, y = generate_multiclass_data(n_samples=300, seed=99)
    split = 240
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    model = SVM(in_features=2, n_classes=3, C=2.0, loss="hinge")
    model.fit(X_train, y_train, epochs=120, lr=0.06, optimizer="adam")

    test_acc = model.evaluate(X_test, y_test, metric="accuracy")
    test_prec = model.evaluate(X_test, y_test, metric="precision")
    test_rec = model.evaluate(X_test, y_test, metric="recall")
    test_f1 = model.evaluate(X_test, y_test, metric="f1")

    preds = model.predict(X_test)
    preds_np = np.array(preds.to_list()).ravel()
    cm = confusion_matrix(y_test, preds_np)

    print(f"Model Configuration: {model.get_config()}")
    print(f"Learned Coefficients shape: {model.coef_.shape}")
    print(f"Test Accuracy    : {test_acc * 100:.2f}%")
    print(f"Macro Precision  : {test_prec:.4f}")
    print(f"Macro Recall     : {test_rec:.4f}")
    print(f"Macro F1 Score   : {test_f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    assert test_acc >= 0.90, f"Expected multiclass accuracy >= 90%, got {test_acc}"
    print("[PASS] Experiment 4 completed successfully!\n")


if __name__ == "__main__":
    print("Starting SVM Model Lab Diagnostic Suite...\n")
    run_experiment_1_separable()
    run_experiment_2_regularization_sweep()
    run_experiment_3_loss_comparison()
    run_experiment_4_multiclass()
    print("All SVM Model Lab Experiments Passed Successfully!")
