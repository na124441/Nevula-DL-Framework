from typing import Any, Dict, List, Optional, Union
import math
import numpy as np

from nevula.core.tensor import Tensor
from nevula.autograd.engine import no_grad
from nevula.nn.linear import Linear
from nevula.losses.bce import BCEWithLogitsLoss
from nevula.nn.losses import CrossEntropyLoss
from nevula.optim.sgd import SGD
from nevula.models.base import BaseModel
from nevula.models.registry import register_model
from nevula.metrics.classification import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    bce_loss,
)


@register_model(name="logistic_regression", category="classification")
class LogisticRegression(BaseModel):
    """
    Logistic Regression classifier for binary and multiclass problems.

    Mathematical Formulation:
        Binary:
            z = X @ W + b
            P(y = 1 | x) = sigma(z) = 1 / (1 + exp(-z))
            L = - (1 / N) * sum( y_i * log(p_i) + (1 - y_i) * log(1 - p_i) ) + Regularization

        Multiclass (Multinomial / Softmax):
            Z = X @ W + b
            P(y = c | x) = exp(z_c) / sum_k exp(z_k)
            L = - (1 / N) * sum( log(P(y_i = true_c | x_i)) ) + Regularization

    Regularization Penalties:
        - L2 (Ridge-style): (1 / (2 * C)) * ||W||_2^2
        - L1 (Lasso-style): (1 / C) * ||W||_1

    Parameters:
        in_features: Number of input features (D).
        n_classes: Number of distinct classes (default: 2).
        penalty: Regularization norm ('l2', 'l1', or None, default 'l2').
        C: Inverse regularization strength (default: 1.0). Must be positive.
        fit_intercept: Whether to calculate the intercept b (default: True).
    """

    def __init__(
        self,
        in_features: int,
        n_classes: int = 2,
        penalty: Optional[str] = "l2",
        C: float = 1.0,
        fit_intercept: bool = True,
    ):
        super().__init__()
        if in_features <= 0:
            raise ValueError(f"in_features must be > 0, got {in_features}")
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {n_classes}")
        if C <= 0.0:
            raise ValueError(f"C must be > 0.0, got {C}")
        if penalty not in ("l2", "l1", None):
            raise ValueError(f"Invalid penalty '{penalty}'. Expected 'l2', 'l1', or None.")

        self.in_features = in_features
        self.n_classes = n_classes
        self.penalty = penalty
        self.C = float(C)
        self.fit_intercept = fit_intercept

        # Output dimension: 1 logit for binary, n_classes logits for multiclass
        self.out_features = 1 if n_classes == 2 else n_classes
        self.linear = Linear(in_features, self.out_features, bias=fit_intercept)

        self.loss_history_: List[float] = []

    def _to_tensor(self, data: Any) -> Tensor:
        if isinstance(data, Tensor):
            return data
        if isinstance(data, np.ndarray):
            return Tensor(data, shape=data.shape)
        return Tensor(data)

    def decision_function(self, X: Any) -> Tensor:
        """
        Computes raw linear logits z = X @ W + b.

        Args:
            X: Input tensor or array-like of shape (N, in_features).

        Returns:
            Tensor: Raw logits of shape (N, 1) if binary else (N, n_classes).
        """
        X_tensor = self._to_tensor(X)
        return self.linear(X_tensor)

    def forward(self, X: Any) -> Tensor:
        """
        Computes forward predictions (raw logits during training, or probabilities during eval).
        """
        return self.decision_function(X)

    def predict_proba(self, X: Any) -> Tensor:
        """
        Computes predicted class probabilities under inference mode.

        Args:
            X: Input features.

        Returns:
            Tensor: Probabilities of shape (N, 2) for binary, or (N, n_classes) for multiclass.
        """
        self.eval()
        with no_grad():
            logits = self.decision_function(X)

            if self.n_classes == 2:
                # Binary: sigmoid(z)
                p1 = logits.sigmoid()
                p0 = 1.0 - p1
                p0_np = np.array(p0.to_list(), dtype=np.float64).reshape((-1, 1))
                p1_np = np.array(p1.to_list(), dtype=np.float64).reshape((-1, 1))
                probs_np = np.hstack([p0_np, p1_np])
                return Tensor(probs_np, shape=probs_np.shape, device=logits.device)
            else:
                # Multiclass: softmax along last dimension
                return logits.softmax(dim=-1)

    def predict(self, X: Any) -> Tensor:
        """
        Predicts discrete class labels for input samples under inference mode.

        Args:
            X: Input features.

        Returns:
            Tensor: Predicted integer labels of shape (N, 1).
        """
        self.eval()
        with no_grad():
            if self.n_classes == 2:
                logits = self.decision_function(X)
                logits_np = np.array(logits.to_list(), dtype=np.float64).reshape((-1, 1))
                preds = (logits_np >= 0.0).astype(np.float64)
            else:
                probs = self.predict_proba(X)
                probs_np = np.array(probs.to_list(), dtype=np.float64)
                preds = np.argmax(probs_np, axis=-1).reshape((-1, 1)).astype(np.float64)

            return Tensor(preds, shape=preds.shape)

    def fit(
        self,
        X: Any,
        y: Any,
        epochs: int = 100,
        lr: float = 0.05,
        optimizer_cls: Any = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> "LogisticRegression":
        """
        Fits the logistic regression model on training data (X, y).

        Args:
            X: Training features (N, D).
            y: Target binary labels {0, 1} or multiclass indices {0, ..., C-1}.
            epochs: Number of optimization epochs.
            lr: Learning rate for gradient descent.
            optimizer_cls: Optimizer class (defaults to SGD).
            verbose: If True, prints training loss every 10 epochs.

        Returns:
            self: The fitted LogisticRegression model.
        """
        self.train()
        X_tensor = self._to_tensor(X)
        if len(X_tensor.shape) == 1:
            X_tensor = X_tensor.reshape((X_tensor.shape[0], 1))

        if self.n_classes == 2:
            # Binary classification: y should be shape (N, 1)
            if isinstance(y, Tensor):
                y_np = np.array(y.to_list(), dtype=np.float64).reshape((-1, 1))
            else:
                y_np = np.array(y, dtype=np.float64).reshape((-1, 1))
            y_tensor = Tensor(y_np, shape=y_np.shape, device=X_tensor.device)
            criterion = BCEWithLogitsLoss(reduction="mean")
        else:
            # Multiclass: y should be 1D class indices (N,)
            if isinstance(y, Tensor):
                y_np = np.array(y.to_list(), dtype=np.float64).reshape((-1,))
            else:
                y_np = np.array(y, dtype=np.float64).reshape((-1,))
            y_tensor = Tensor(y_np, shape=(len(y_np),), device=X_tensor.device)
            criterion = CrossEntropyLoss(reduction="mean")

        opt_class = optimizer_cls if optimizer_cls is not None else SGD
        optimizer = opt_class(self.parameters(), lr=lr)
        self.loss_history_ = []

        for epoch in range(epochs):
            optimizer.zero_grad()

            logits = self.decision_function(X_tensor)
            loss = criterion(logits, y_tensor)

            # Add regularization penalty
            reg_loss = None
            if self.penalty == "l2":
                # L2 penalty: (1 / (2 * C)) * sum(W ** 2)
                reg_loss = (self.linear.weight ** 2).sum() * (1.0 / (2.0 * self.C))
            elif self.penalty == "l1":
                # L1 penalty: (1 / C) * sum(|W|)
                reg_loss = self.linear.weight.abs().sum() * (1.0 / self.C)

            if reg_loss is not None:
                total_loss = loss + reg_loss
            else:
                total_loss = loss

            total_loss.backward()
            optimizer.step()

            loss_val = float(total_loss.item()) if hasattr(total_loss, "item") else float(total_loss[()])
            self.loss_history_.append(loss_val)

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(f"Epoch [{epoch + 1:4d}/{epochs:4d}] - Loss: {loss_val:.4f}")

        self._is_fitted = True
        return self

    def evaluate(self, X: Any, y: Any, metric: str = "accuracy") -> float:
        """
        Evaluates the fitted model on test data according to specified metric.

        Args:
            X: Test features.
            y: True labels.
            metric: 'accuracy', 'precision', 'recall', 'f1', or 'bce'.

        Returns:
            float: Computed metric value.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before evaluation.")

        preds = self.predict(X)
        y_true_np = np.array(y.to_list() if isinstance(y, Tensor) else y, dtype=np.float64).ravel()
        y_pred_np = np.array(preds.to_list(), dtype=np.float64).ravel()

        if metric == "accuracy":
            return accuracy_score(y_true_np, y_pred_np)
        elif metric == "precision":
            avg = "binary" if self.n_classes == 2 else "macro"
            return precision_score(y_true_np, y_pred_np, average=avg)
        elif metric == "recall":
            avg = "binary" if self.n_classes == 2 else "macro"
            return recall_score(y_true_np, y_pred_np, average=avg)
        elif metric == "f1":
            avg = "binary" if self.n_classes == 2 else "macro"
            return f1_score(y_true_np, y_pred_np, average=avg)
        elif metric in ("bce", "log_loss"):
            probs = self.predict_proba(X)
            probs_np = np.array(probs.to_list(), dtype=np.float64)
            if self.n_classes == 2:
                return bce_loss(y_true_np, probs_np[:, 1])
            else:
                # Mean negative log likelihood for multiclass
                n = len(y_true_np)
                nll = 0.0
                for i in range(n):
                    c = int(y_true_np[i])
                    p = max(probs_np[i, c], 1e-12)
                    nll -= np.log(p)
                return float(nll / n)
        else:
            raise ValueError(f"Unknown metric '{metric}'. Expected 'accuracy', 'precision', 'recall', 'f1', 'bce'.")

    @property
    def coef_(self) -> np.ndarray:
        """Returns weight matrix as a numpy array."""
        return np.array(self.linear.weight.to_list(), dtype=np.float64).T

    @property
    def intercept_(self) -> Optional[np.ndarray]:
        """Returns intercept vector as a numpy array."""
        if self.linear.bias is None:
            return None
        return np.array(self.linear.bias.to_list(), dtype=np.float64)

    def get_config(self) -> Dict[str, Any]:
        """Returns model configuration for checkpoint serialization."""
        return {
            "in_features": self.in_features,
            "n_classes": self.n_classes,
            "penalty": self.penalty,
            "C": self.C,
            "fit_intercept": self.fit_intercept,
        }

    def __repr__(self) -> str:
        return (
            f"LogisticRegression(in_features={self.in_features}, "
            f"n_classes={self.n_classes}, penalty='{self.penalty}', C={self.C})"
        )


# Register alias
register_model(LogisticRegression, name="logistic", category="classification")

__all__ = ["LogisticRegression"]
