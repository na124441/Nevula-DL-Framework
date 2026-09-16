from typing import Any, Dict, List, Optional, Union
import math
import random
import numpy as np

from nevula.core.tensor import Tensor
from nevula.autograd.engine import no_grad
from nevula.nn.parameter import Parameter
from nevula.nn.linear import Linear
from nevula.optim.sgd import SGD
from nevula.optim.adam import Adam
from nevula.optim.adamw import AdamW
from nevula.optim.optimizer import Optimizer
from nevula.data.dataset import TensorDataset
from nevula.data.dataloader import DataLoader
from nevula.models.base import BaseModel
from nevula.models.registry import register_model
from nevula.metrics.classification import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    hinge_loss_score,
)


@register_model(name="svc", category="classification")
class SVC(BaseModel):
    """
    Support Vector Classifier (SVC / SVM) using Nevula autograd and tensor primitives.

    Mathematical Formulation:
        Linear Decision Boundary:
            z = X @ W + b

        Primal Soft-Margin Objective Function (Binary):
            min_{W, b}  (1 / N) * sum( max(0, 1 - y_i * z_i)^p ) + (1 / (2 * C)) * ||W||_2^2
            where p = 1 for standard Hinge Loss ('hinge'),
                  p = 2 for Squared Hinge Loss ('squared_hinge'),
                  and y_i in {-1, +1}.

        Multiclass Strategy (One-vs-Rest / OvR):
            For K classes, K hyperplanes (w_k, b_k) are jointly optimized against OvR binary targets
            Y_ik in {-1, +1}, where Y_ik = +1 if y_i == k else -1.
            The predicted class is:
                y_hat = argmax_k (x^T w_k + b_k)

        Support Vectors:
            Observations x_i that lie on the margin boundary or violate the margin:
                1 - y_i * z_i >= -tol   (i.e. y_i * z_i <= 1 + tol)

    Parameters:
        in_features: Number of input features (D).
        n_classes: Number of distinct classes (default: 2).
        C: Regularization parameter (default: 1.0). Controls the trade-off between
           maximizing the margin and minimizing margin violations. Must be strictly positive.
        loss: Loss function to optimize ('hinge' or 'squared_hinge', default 'hinge').
        fit_intercept: Whether to compute the intercept/bias b (default: True).
        mode: Implementation mode ('framework' with Linear layer, or 'tensor' with Parameters).
    """

    def __init__(
        self,
        in_features: int,
        n_classes: int = 2,
        C: float = 1.0,
        loss: str = "hinge",
        fit_intercept: bool = True,
        mode: str = "framework",
    ):
        super().__init__()
        if in_features <= 0:
            raise ValueError(f"in_features must be strictly positive, got {in_features}")
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {n_classes}")
        if C <= 0.0:
            raise ValueError(f"C must be strictly positive, got {C}")
        if loss not in ("hinge", "squared_hinge"):
            raise ValueError(f"Invalid loss '{loss}'. Expected 'hinge' or 'squared_hinge'.")
        if mode not in ("framework", "tensor"):
            raise ValueError(f"Invalid mode '{mode}'. Expected 'framework' or 'tensor'.")

        self.in_features = in_features
        self.n_classes = n_classes
        self.C = float(C)
        self.loss = loss
        self.fit_intercept = fit_intercept
        self.mode = mode

        self.loss_history: List[float] = []
        self.classes_: Optional[np.ndarray] = None
        self._is_fitted: bool = False

        # Output feature dimension: 1 for binary classification, n_classes for multiclass OvR
        self.out_features = 1 if n_classes == 2 else n_classes

        if self.mode == "framework":
            self.linear = Linear(in_features, self.out_features, bias=fit_intercept)
        else:
            bound = 1.0 / math.sqrt(in_features)
            w_init = [random.uniform(-bound, bound) for _ in range(in_features * self.out_features)]
            self.weights = Parameter(Tensor(w_init, shape=(in_features, self.out_features)))
            if fit_intercept:
                b_init = [0.0 for _ in range(self.out_features)]
                self.bias_param = Parameter(Tensor(b_init, shape=(self.out_features,)))
            else:
                self.bias_param = None

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
    def bias(self) -> Optional[Parameter]:
        """Returns the bias/intercept parameter tensor."""
        if self.mode == "framework":
            return self.linear.bias
        return self.bias_param

    @property
    def coef_(self) -> np.ndarray:
        """
        Returns model coefficients as a NumPy array.
        Shape is (1, in_features) for binary, or (n_classes, in_features) for multiclass.
        """
        w_np = np.array(self.weight.to_list(), dtype=np.float64)
        if len(w_np.shape) == 1:
            w_np = w_np.reshape((self.in_features, self.out_features))
        # Transpose so rows correspond to classes/hyperplanes: (out_features, in_features)
        return w_np.T

    @property
    def intercept_(self) -> np.ndarray:
        """Returns model intercept(s) as a NumPy array."""
        if self.bias is None:
            return np.zeros(self.out_features, dtype=np.float64)
        b_np = np.array(self.bias.to_list(), dtype=np.float64)
        return b_np.ravel()

    def _to_tensor(self, data: Any) -> Tensor:
        if isinstance(data, Tensor):
            return data
        if isinstance(data, np.ndarray):
            return Tensor(data, shape=data.shape)
        return Tensor(data)

    def decision_function(self, X: Any) -> Tensor:
        """
        Computes raw linear decision scores z = X @ W + b.

        Args:
            X: Input tensor or array-like of shape (N, in_features).

        Returns:
            Tensor: Raw scores of shape (N, 1) if binary else (N, n_classes).
        """
        X_tensor = self._to_tensor(X)
        if len(X_tensor.shape) == 1:
            X_tensor = X_tensor.reshape((1, len(X_tensor)))

        if self.mode == "framework":
            return self.linear(X_tensor)
        else:
            out = X_tensor @ self.weights
            if self.bias_param is not None:
                out = out + self.bias_param
            return out

    def forward(self, X: Any) -> Tensor:
        """Alias to decision_function."""
        return self.decision_function(X)

    def predict(self, X: Any) -> Tensor:
        """
        Predicts discrete class labels for input samples under inference mode.

        Args:
            X: Input features of shape (N, in_features).

        Returns:
            Tensor: Predicted labels of shape (N, 1).
        """
        self.eval()
        with no_grad():
            scores = self.decision_function(X)
            scores_np = np.array(scores.to_list(), dtype=np.float64)

            if len(scores_np.shape) == 1:
                scores_np = scores_np.reshape((-1, self.out_features))

            if self.n_classes == 2:
                # Binary: positive score predicts positive class (index 1), else negative class (index 0)
                pred_indices = (scores_np >= 0.0).astype(int).ravel()
            else:
                # Multiclass: argmax over class decision hyperplanes
                pred_indices = np.argmax(scores_np, axis=-1).ravel()

            if self.classes_ is not None:
                mapped_preds = self.classes_[pred_indices].reshape((-1, 1)).astype(np.float64)
            else:
                mapped_preds = pred_indices.reshape((-1, 1)).astype(np.float64)

            return Tensor(mapped_preds, shape=mapped_preds.shape)

    def fit(
        self,
        X: Any,
        y: Any,
        epochs: int = 100,
        lr: float = 0.05,
        batch_size: Optional[int] = None,
        optimizer: Union[str, Optimizer, None] = "sgd",
        verbose: bool = False,
        **kwargs: Any,
    ) -> "SVC":
        """
        Fits the Support Vector Classifier on training data (X, y).

        Args:
            X: Training features (N, in_features).
            y: Target class labels.
            epochs: Number of optimization epochs.
            lr: Learning rate.
            batch_size: Optional mini-batch size. If None, uses full-batch gradient descent.
            optimizer: Optimizer name ('sgd', 'adam', 'adamw') or Optimizer instance.
            verbose: If True, prints training loss progress.

        Returns:
            self: The fitted SVC model.
        """
        self.train()
        X_t = self._to_tensor(X)
        if len(X_t.shape) == 1:
            X_t = X_t.reshape((X_t.shape[0], 1))

        # Convert y to 1D numpy array to inspect unique classes
        if isinstance(y, Tensor):
            y_raw = np.array(y.to_list(), dtype=np.float64).ravel()
        elif isinstance(y, np.ndarray):
            y_raw = y.astype(np.float64).ravel()
        else:
            y_raw = np.array(y, dtype=np.float64).ravel()

        unique_classes = np.unique(y_raw)
        if len(unique_classes) < 2:
            raise ValueError(f"Training data must contain at least 2 classes, got {len(unique_classes)}")
        if len(unique_classes) > self.n_classes:
            raise ValueError(
                f"Data contains {len(unique_classes)} classes, but model initialized for n_classes={self.n_classes}"
            )

        self.classes_ = np.sort(unique_classes)
        n_samples = X_t.shape[0]

        # Construct target array in {-1.0, +1.0}
        if self.n_classes == 2:
            # Binary: assign -1.0 to class 0, +1.0 to class 1
            pos_label = self.classes_[1]
            y_pm = np.where(y_raw == pos_label, 1.0, -1.0).reshape((-1, 1))
            y_t = Tensor(y_pm, shape=y_pm.shape, device=X_t.device)
        else:
            # Multiclass OvR: target shape (N, n_classes) with +1 for true class, -1 otherwise
            Y_ovr = -np.ones((n_samples, self.n_classes), dtype=np.float64)
            for idx, c in enumerate(self.classes_):
                Y_ovr[y_raw == c, idx] = 1.0
            y_t = Tensor(Y_ovr, shape=Y_ovr.shape, device=X_t.device)

        # Setup optimizer
        if isinstance(optimizer, Optimizer):
            opt = optimizer
        elif isinstance(optimizer, str) or optimizer is None:
            opt_name = (optimizer or "sgd").lower().strip()
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

        self.loss_history = []
        use_dataloader = batch_size is not None and 0 < batch_size < n_samples

        if use_dataloader:
            dataset = TensorDataset(X_t, y_t)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        reg_factor = 0.5 / self.C

        for epoch in range(1, epochs + 1):
            if use_dataloader:
                batch_losses = []
                for batch_x, batch_y in dataloader:
                    opt.zero_grad()
                    scores = self.decision_function(batch_x)

                    # Margin violation: max(0, 1 - y * score)
                    margin_violation = (1.0 - batch_y * scores).relu()
                    if self.loss == "squared_hinge":
                        margin_loss = (margin_violation ** 2).mean()
                    else:
                        margin_loss = margin_violation.mean()

                    reg_loss = (self.weight ** 2).sum() * reg_factor
                    total_loss = margin_loss + reg_loss

                    total_loss.backward()
                    opt.step()

                    loss_val = float(total_loss.data[0]) if hasattr(total_loss.data, "__getitem__") else float(total_loss.data)
                    batch_losses.append(loss_val)
                current_loss = float(np.mean(batch_losses))
            else:
                opt.zero_grad()
                scores = self.decision_function(X_t)

                margin_violation = (1.0 - y_t * scores).relu()
                if self.loss == "squared_hinge":
                    margin_loss = (margin_violation ** 2).mean()
                else:
                    margin_loss = margin_violation.mean()

                reg_loss = (self.weight ** 2).sum() * reg_factor
                total_loss = margin_loss + reg_loss

                total_loss.backward()
                opt.step()
                current_loss = float(total_loss.data[0]) if hasattr(total_loss.data, "__getitem__") else float(total_loss.data)

            self.loss_history.append(current_loss)

            if verbose and (epoch == 1 or epoch % max(1, epochs // 10) == 0 or epoch == epochs):
                print(f"Epoch [{epoch:4d}/{epochs:4d}] - Loss: {current_loss:.6f}")

        self._is_fitted = True
        return self

    def support_vectors_mask(self, X: Any, y: Any, tol: float = 1e-3) -> np.ndarray:
        """
        Returns a boolean mask indicating which observations are Support Vectors
        (i.e. lie on or violate the margin boundary: 1 - y * score >= -tol).

        Args:
            X: Input features.
            y: Ground truth targets.
            tol: Numerical tolerance for margin boundary condition.

        Returns:
            np.ndarray: 1D boolean array of shape (N,).
        """
        scores = self.decision_function(X)
        scores_np = np.array(scores.to_list(), dtype=np.float64)

        if isinstance(y, Tensor):
            y_np = np.array(y.to_list(), dtype=np.float64).ravel()
        elif isinstance(y, np.ndarray):
            y_np = y.astype(np.float64).ravel()
        else:
            y_np = np.array(y, dtype=np.float64).ravel()

        if self.n_classes == 2:
            scores_vec = scores_np.ravel()
            pos_label = self.classes_[1] if self.classes_ is not None else 1.0
            y_pm = np.where(y_np == pos_label, 1.0, -1.0)
            margin = y_pm * scores_vec
            # Points on or within the margin boundary
            return margin <= (1.0 + tol)
        else:
            # Multiclass OvR: sample is a support vector if it is on/violates the margin for any class
            n_samples = len(y_np)
            mask = np.zeros(n_samples, dtype=bool)
            classes = self.classes_ if self.classes_ is not None else np.arange(self.n_classes)
            for idx, c in enumerate(classes):
                y_c = np.where(y_np == c, 1.0, -1.0)
                class_scores = scores_np[:, idx]
                class_margin = y_c * class_scores
                mask = mask | (class_margin <= (1.0 + tol))
            return mask

    def support_vectors(self, X: Any, y: Any, tol: float = 1e-3) -> np.ndarray:
        """
        Extracts the input features corresponding to the Support Vectors.

        Args:
            X: Input features.
            y: Ground truth targets.
            tol: Numerical tolerance for margin boundary condition.

        Returns:
            np.ndarray: 2D array of support vector feature vectors.
        """
        if isinstance(X, Tensor):
            X_np = np.array(X.to_list(), dtype=np.float64)
        elif isinstance(X, np.ndarray):
            X_np = X.astype(np.float64)
        else:
            X_np = np.array(X, dtype=np.float64)

        mask = self.support_vectors_mask(X, y, tol=tol)
        return X_np[mask]

    def support_vectors_ratio(self, X: Any, y: Any, tol: float = 1e-3) -> float:
        """
        Computes the proportion of observations that are Support Vectors.

        Args:
            X: Input features.
            y: Ground truth targets.
            tol: Numerical tolerance.

        Returns:
            float: Fraction between 0.0 and 1.0.
        """
        mask = self.support_vectors_mask(X, y, tol=tol)
        if len(mask) == 0:
            return 0.0
        return float(np.mean(mask))

    def n_support_(self, X: Any, y: Any, tol: float = 1e-3) -> int:
        """
        Computes total number of Support Vectors.

        Args:
            X: Input features.
            y: Ground truth targets.
            tol: Numerical tolerance.

        Returns:
            int: Number of support vectors.
        """
        mask = self.support_vectors_mask(X, y, tol=tol)
        return int(np.sum(mask))

    def evaluate(self, X: Any, y: Any, metric: str = "accuracy") -> float:
        """
        Evaluates the fitted model on test data according to specified metric.

        Args:
            X: Test features.
            y: True labels.
            metric: 'accuracy', 'precision', 'recall', 'f1', or 'hinge_loss'.

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
        elif metric == "hinge_loss":
            scores = self.decision_function(X)
            scores_np = np.array(scores.to_list(), dtype=np.float64)
            return hinge_loss_score(y_true_np, scores_np)
        else:
            raise ValueError(f"Unknown metric '{metric}'. Expected 'accuracy', 'precision', 'recall', 'f1', or 'hinge_loss'.")

    def get_config(self) -> Dict[str, Any]:
        """Returns model hyperparameters and configuration dictionary."""
        return {
            "in_features": self.in_features,
            "n_classes": self.n_classes,
            "C": self.C,
            "loss": self.loss,
            "fit_intercept": self.fit_intercept,
            "mode": self.mode,
            "classes": self.classes_.tolist() if self.classes_ is not None else None,
        }

    def __repr__(self) -> str:
        return (
            f"SVC(in_features={self.in_features}, "
            f"n_classes={self.n_classes}, "
            f"C={self.C}, "
            f"loss='{self.loss}', "
            f"fit_intercept={self.fit_intercept}, "
            f"mode='{self.mode}')"
        )


# Register aliases for maximum discoverability
register_model(SVC, name="svm", category="classification")
register_model(SVC, name="support_vector_classifier", category="classification")

# Export alias SVM
SVM = SVC

__all__ = ["SVC", "SVM"]
