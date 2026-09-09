"""
Model Template for Nevula Model Library.

Follow this template when contributing a new machine-learning or deep-learning model.

Lifecycle Checklist:
    [ ] 1. Define mathematical formulation and objective function in docstrings.
    [ ] 2. Subclass `BaseModel` and register with `@register_model(name="...", category="...")`.
    [ ] 3. Initialize hyperparameters and register trainable `Parameter` objects.
    [ ] 4. Implement `forward(X)` computing forward predictions.
    [ ] 5. Implement `fit(X, y, **kwargs)` handling data conversion, optimizer, and loss backward.
    [ ] 6. Implement `predict(X)` generating predictions under `no_grad()`.
    [ ] 7. Implement `get_config()` returning dictionary of hyperparameters.
    [ ] 8. Add unit tests in `tests/models/<category>/test_<model>.py`.
    [ ] 9. Add runnable example in `examples/<model>.py`.
    [ ] 10. Add mathematical documentation in `docs/models/<model>.md`.
"""

from typing import Any, Dict, Optional, Union
import numpy as np

from nevula.core.tensor import Tensor
from nevula.autograd.engine import no_grad
from nevula.nn.parameter import Parameter
from nevula.models.base import BaseModel
from nevula.models.registry import register_model


# Example: change category to 'regression', 'classification', 'trees', or 'deep'
@register_model(name="example_model_template", category="general")
class ModelTemplate(BaseModel):
    """
    Brief 1-2 line description of the algorithm and problem solved.

    Mathematical Formulation:
        y_hat = f(X; theta)

    Objective Function:
        L(theta) = Loss(y_hat, y) + Regularization(theta)

    Parameters:
        in_features (int): Number of input features.
        out_features (int): Number of output targets or classes.
        learning_rate (float): Step size for optimizer updates.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        learning_rate: float = 0.01,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.learning_rate = learning_rate

        # 1. Define and register trainable parameters
        # Example: self.weights = Parameter(Tensor(...))
        # self.bias = Parameter(Tensor(...))

    def forward(self, X: Tensor) -> Tensor:
        """
        Computes forward pass: y_hat = f(X).

        Args:
            X: Input Tensor of shape (batch_size, in_features).

        Returns:
            Tensor: Prediction tensor of shape (batch_size, out_features).
        """
        # Implement mathematical forward transformation
        raise NotImplementedError("Implement forward()")

    def fit(
        self,
        X: Any,
        y: Any,
        epochs: int = 100,
        batch_size: Optional[int] = None,
        verbose: bool = False,
    ) -> "ModelTemplate":
        """
        Trains model parameters on dataset (X, y).

        Args:
            X: Input features (Tensor, ndarray, or list).
            y: Target values (Tensor, ndarray, or list).
            epochs: Number of training epochs.
            batch_size: Optional mini-batch size.
            verbose: If True, prints training loss per epoch.

        Returns:
            self: Fitted model.
        """
        X_t = self._to_tensor(X)
        y_t = self._to_tensor(y)

        # Training loop implementation (e.g. forward -> loss -> backward -> optimizer step)
        self._is_fitted = True
        return self

    def predict(self, X: Any) -> Tensor:
        """
        Generates predictions for inputs X under inference mode.

        Args:
            X: Input features (Tensor, ndarray, or list).

        Returns:
            Tensor: Predicted targets.
        """
        self.eval()
        X_t = self._to_tensor(X)
        with no_grad():
            preds = self.forward(X_t)
        return preds

    def get_config(self) -> Dict[str, Any]:
        """
        Returns model configuration and hyperparameters.
        """
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "learning_rate": self.learning_rate,
        }
