from typing import Any, Dict, List, Optional, Union
import math
import random
import numpy as np

from nevula.core.tensor import Tensor
from nevula.autograd.engine import no_grad
from nevula.nn.parameter import Parameter
from nevula.nn.linear import Linear
from nevula.nn.losses import MSELoss
from nevula.optim.sgd import SGD
from nevula.optim.adam import Adam
from nevula.optim.adamw import AdamW
from nevula.optim.optimizer import Optimizer
from nevula.data.dataset import TensorDataset
from nevula.data.dataloader import DataLoader
from nevula.models.base import BaseModel
from nevula.models.registry import register_model


@register_model(name="linear_regression", category="regression")
class LinearRegression(BaseModel):
    """
    Ordinary Linear Regression model using Nevula primitives.

    Mathematical Model:
        y_hat = X @ W + b

    Objective (MSE Loss):
        L = (1 / n) * sum((y_hat - y) ** 2)

    Supports two educational modes:
        - "framework": Built by composing `nevula.nn.Linear`, `MSELoss`, and optimizers.
        - "tensor": Built directly from elementary `Parameter` and `Tensor` matrix multiplications.

    Parameters:
        in_features: Number of input features (D).
        out_features: Number of output targets (K, default 1).
        fit_intercept: Whether to learn an additive bias vector b.
        mode: Implementation mode ('framework' or 'tensor').
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        fit_intercept: bool = True,
        mode: str = "framework",
    ):
        super().__init__()
        if in_features <= 0:
            raise ValueError(f"in_features must be positive, got {in_features}")
        if out_features <= 0:
            raise ValueError(f"out_features must be positive, got {out_features}")
        if mode not in ("framework", "tensor"):
            raise ValueError(f"Invalid mode '{mode}'. Expected 'framework' or 'tensor'.")

        self.in_features = in_features
        self.out_features = out_features
        self.fit_intercept = fit_intercept
        self.mode = mode
        self.loss_history: List[float] = []

        if self.mode == "framework":
            self.linear = Linear(in_features, out_features, bias=fit_intercept)
        else:
            # Mode A: Direct tensor parameterization
            bound = 1.0 / math.sqrt(in_features)
            w_init = [random.uniform(-bound, bound) for _ in range(in_features * out_features)]
            self.weights = Parameter(Tensor(w_init, shape=(in_features, out_features)))

            if fit_intercept:
                b_init = [0.0 for _ in range(out_features)]
                self.bias = Parameter(Tensor(b_init, shape=(out_features,)))
            else:
                self.bias = None

    @property
    def weight(self) -> Parameter:
        """Returns the weight parameter tensor."""
        if self.mode == "framework":
            return self.linear.weight
        return self.weights

    @property
    def weights_param(self) -> Parameter:
        """Alias for weight parameter tensor."""
        return self.weight

    @property
    def bias_param(self) -> Optional[Parameter]:
        """Returns the bias parameter tensor if fitted, else None."""
        if self.mode == "framework":
            return self.linear.bias
        return self.bias

    def forward(self, X: Tensor) -> Tensor:
        """
        Computes the linear prediction: y_hat = X @ W + b.

        Args:
            X: Input Tensor of shape (N, in_features).

        Returns:
            Tensor: Prediction of shape (N, out_features).
        """
        if self.mode == "framework":
            return self.linear(X)

        out = X @ self.weights
        if self.bias is not None:
            out = out + self.bias
        return out

    def _prepare_inputs(self, X: Any, y: Optional[Any] = None) -> tuple[Tensor, Optional[Tensor]]:
        """Ensures X and y are properly shaped Nevula 2D Tensors."""
        X_t = self._to_tensor(X)
        if len(X_t.shape) == 1:
            X_t = X_t.reshape((X_t.shape[0], 1))
        elif len(X_t.shape) != 2:
            raise ValueError(f"Expected 2D input features (N, D), got shape {X_t.shape}")

        if X_t.shape[1] != self.in_features:
            raise ValueError(
                f"Feature dimension mismatch: expected in_features={self.in_features}, "
                f"got {X_t.shape[1]}"
            )

        if y is None:
            return X_t, None

        y_t = self._to_tensor(y)
        if len(y_t.shape) == 1:
            y_t = y_t.reshape((y_t.shape[0], 1))
        elif len(y_t.shape) != 2:
            raise ValueError(f"Expected 1D or 2D target values, got shape {y_t.shape}")

        if y_t.shape[1] != self.out_features:
            raise ValueError(
                f"Target dimension mismatch: expected out_features={self.out_features}, "
                f"got {y_t.shape[1]}"
            )

        if X_t.shape[0] != y_t.shape[0]:
            raise ValueError(
                f"Batch size mismatch: X has {X_t.shape[0]} samples, y has {y_t.shape[0]} samples"
            )

        return X_t, y_t

    def fit(
        self,
        X: Any,
        y: Any,
        epochs: int = 100,
        lr: float = 0.01,
        batch_size: Optional[int] = None,
        optimizer: Union[str, Optimizer] = "sgd",
        verbose: bool = False,
    ) -> "LinearRegression":
        """
        Trains the Linear Regression model using gradient descent and Nevula autograd.

        Args:
            X: Training features (Tensor, ndarray, or sequence).
            y: Training targets (Tensor, ndarray, or sequence).
            epochs: Number of optimization epochs.
            lr: Learning rate (if creating an internal optimizer).
            batch_size: Optional mini-batch size for mini-batch SGD.
            optimizer: Optimizer name ('sgd', 'adam', 'adamw') or an existing Optimizer instance.
            verbose: If True, logs training loss every 10 epochs.

        Returns:
            self: The fitted model.
        """
        self.train()
        X_t, y_t = self._prepare_inputs(X, y)
        n_samples = X_t.shape[0]

        # Configure optimizer
        if isinstance(optimizer, Optimizer):
            opt = optimizer
        elif isinstance(optimizer, str):
            opt_name = optimizer.lower().strip()
            if opt_name == "sgd":
                opt = SGD(self.parameters(), lr=lr)
            elif opt_name == "adam":
                opt = Adam(self.parameters(), lr=lr)
            elif opt_name == "adamw":
                opt = AdamW(self.parameters(), lr=lr)
            else:
                raise ValueError(f"Unknown optimizer '{optimizer}'. Expected 'sgd', 'adam', 'adamw', or Optimizer.")
        else:
            raise TypeError(f"Invalid optimizer type: {type(optimizer)}")

        criterion = MSELoss(reduction="mean")
        self.loss_history = []

        # Mini-batch or Full-batch training
        use_dataloader = batch_size is not None and 0 < batch_size < n_samples

        if use_dataloader:
            dataset = TensorDataset(X_t, y_t)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(1, epochs + 1):
            if use_dataloader:
                batch_losses = []
                for batch_x, batch_y in dataloader:
                    opt.zero_grad()
                    pred = self.forward(batch_x)
                    loss = criterion(pred, batch_y)
                    loss.backward()
                    opt.step()
                    # loss.data is scalar or 1D array
                    loss_val = float(loss.data[0]) if hasattr(loss.data, "__getitem__") else float(loss.data)
                    batch_losses.append(loss_val)
                current_loss = float(np.mean(batch_losses))
            else:
                opt.zero_grad()
                pred = self.forward(X_t)
                loss = criterion(pred, y_t)
                loss.backward()
                opt.step()
                current_loss = float(loss.data[0]) if hasattr(loss.data, "__getitem__") else float(loss.data)

            self.loss_history.append(current_loss)

            if verbose and (epoch == 1 or epoch % max(1, epochs // 10) == 0 or epoch == epochs):
                print(f"Epoch [{epoch:4d}/{epochs:4d}] - Loss (MSE): {current_loss:.6f}")

        self._is_fitted = True
        return self

    def predict(self, X: Any) -> Tensor:
        """
        Computes predictions for input features X.

        Args:
            X: Input features (Tensor, ndarray, or sequence).

        Returns:
            Tensor: Predicted targets of shape (N, out_features).
        """
        self.eval()
        X_t, _ = self._prepare_inputs(X, y=None)
        with no_grad():
            preds = self.forward(X_t)
        return preds

    def get_config(self) -> Dict[str, Any]:
        """Returns model hyperparameters and configuration."""
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "fit_intercept": self.fit_intercept,
            "mode": self.mode,
        }

    def __repr__(self) -> str:
        return (
            f"LinearRegression(in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"fit_intercept={self.fit_intercept}, "
            f"mode='{self.mode}')"
        )


__all__ = ["LinearRegression"]
