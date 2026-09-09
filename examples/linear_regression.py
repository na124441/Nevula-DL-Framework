"""
Nevula Example: End-to-End Linear Regression.

Demonstrates:
  1. Synthetic dataset creation with known slope and intercept.
  2. Model instantiation via Nevula Model Library.
  3. Model training using Autograd and gradient descent.
  4. Loss monitoring during training.
  5. Generating predictions on holdout test set.
  6. Metric evaluation (MSE and R^2 score).
"""

import os
import sys

# Ensure repository root is on sys.path when script is executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from nevula.models import LinearRegression, list_models, summary
from nevula.metrics import mean_squared_error, r2_score


def main():
    print("=" * 65)
    print("       NEVULA MODEL LIBRARY — LINEAR REGRESSION DEMO")
    print("=" * 65)

    # 0. Inspect available models in registry
    print("\n--- 1. Inspecting Model Registry ---")
    print(summary().strip())

    # 1. Dataset Generation: True relationship: y = 3.0 * x + 2.0
    print("\n--- 2. Creating Synthetic Dataset ---")
    np.random.seed(42)
    n_samples = 200
    true_slope = 3.0
    true_intercept = 2.0

    X_raw = np.random.uniform(-3.0, 3.0, size=(n_samples, 1))
    noise = np.random.normal(0.0, 0.2, size=(n_samples, 1))
    y_raw = true_slope * X_raw + true_intercept + noise

    # Train / Test split (80% train, 20% test)
    split_idx = int(n_samples * 0.8)
    X_train, X_test = X_raw[:split_idx], X_raw[split_idx:]
    y_train, y_test = y_raw[:split_idx], y_raw[split_idx:]

    print(f"Generated {n_samples} samples.")
    print(f"Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")
    print(f"Ground Truth Relation: y = {true_slope} * x + {true_intercept}")

    # 2. Model Creation
    print("\n--- 3. Instantiating LinearRegression ---")
    model = LinearRegression(
        in_features=1,
        out_features=1,
        fit_intercept=True,
        mode="framework",
    )
    print(f"Model: {model}")
    print(f"Initial parameters: W={model.weight.data[0]:.4f}, b={model.bias_param.data[0]:.4f}")

    # 3. Model Training
    print("\n--- 4. Training Model with Nevula Autograd & SGD ---")
    model.fit(
        X_train,
        y_train,
        epochs=120,
        lr=0.03,
        batch_size=16,
        optimizer="sgd",
        verbose=True,
    )

    # 4. Inspect Learned Parameters
    learned_w = float(model.weight.data[0])
    learned_b = float(model.bias_param.data[0])
    print("\n--- 5. Learned Parameters vs Ground Truth ---")
    print(f"True Slope:     {true_slope:8.4f} | Learned Slope:     {learned_w:8.4f}")
    print(f"True Intercept: {true_intercept:8.4f} | Learned Intercept: {learned_b:8.4f}")

    # 5. Prediction on Holdout Test Set
    print("\n--- 6. Inference on Test Set ---")
    predictions = model.predict(X_test)
    print(f"Prediction output shape: {predictions.shape}")

    # Display first 5 test predictions
    print("\nFirst 5 test sample comparisons:")
    preds_np = np.array(predictions.to_list())
    for i in range(5):
        print(
            f"  x={float(X_test[i, 0]):6.2f} | "
            f"y_true={float(y_test[i, 0]):6.2f} | "
            f"y_pred={float(preds_np[i, 0]):6.2f} | "
            f"error={abs(float(y_test[i, 0]) - float(preds_np[i, 0])):6.3f}"
        )

    # 6. Evaluation
    print("\n--- 7. Evaluation Metrics ---")
    test_mse = model.evaluate(X_test, y_test, metric="mse")
    test_mae = model.evaluate(X_test, y_test, metric="mae")
    test_r2 = model.evaluate(X_test, y_test, metric="r2")

    print(f"Test MSE: {test_mse:.6f}")
    print(f"Test MAE: {test_mae:.6f}")
    print(f"Test R^2: {test_r2:.6f}")

    print("\n" + "=" * 65)
    print(" Linear Regression demonstration completed successfully!")
    print("=" * 65)


if __name__ == "__main__":
    main()
