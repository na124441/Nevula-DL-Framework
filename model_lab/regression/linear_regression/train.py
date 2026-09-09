"""
Experimental training runner for Linear Regression.
"""

import os
import sys

# Ensure repository root is on sys.path when script is executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
from nevula.core.tensor import Tensor
from nevula.nn.losses import MSELoss
from nevula.optim.sgd import SGD
from model_lab.regression.linear_regression.model import ExperimentalLinearRegression


def train_linear_model(
    n_samples: int = 200,
    true_w: float = 3.0,
    true_b: float = 2.0,
    epochs: int = 150,
    lr: float = 0.05,
    seed: int = 42,
):
    np.random.seed(seed)
    X_np = np.random.uniform(-3.0, 3.0, size=(n_samples, 1))
    noise = np.random.normal(0.0, 0.1, size=(n_samples, 1))
    y_np = true_w * X_np + true_b + noise

    X = Tensor(X_np)
    y = Tensor(y_np)

    model = ExperimentalLinearRegression(in_features=1, out_features=1, fit_intercept=True)
    optimizer = SGD(model.parameters(), lr=lr)
    criterion = MSELoss()

    model.train()
    print(f"Training on synthetic line: y = {true_w}x + {true_b}...")
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        if epoch % 25 == 0 or epoch == epochs:
            loss_val = float(loss.data[0]) if hasattr(loss.data, "__getitem__") else float(loss.data)
            w_val = float(model.weight.data[0])
            b_val = float(model.bias.data[0])
            print(f"Epoch {epoch:3d} | Loss: {loss_val:.6f} | Learned: W={w_val:.4f}, b={b_val:.4f}")

    return model


if __name__ == "__main__":
    train_linear_model()
