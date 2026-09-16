"""
Experimental Lasso Regression Prototype.
Developed inside model_lab before official graduation to nevula.models.
"""

import os
import sys

# Ensure repository root is on sys.path when script is executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from typing import Any, Dict, Optional
import math
import random

from nevula.core.tensor import Tensor
from nevula.autograd.engine import no_grad
from nevula.nn.module import Module
from nevula.nn.parameter import Parameter


class ExperimentalLassoRegression(Module):
    """
    Experimental prototype of Lasso Regression using elementary tensor operations.
    """

    def __init__(self, in_features: int, out_features: int = 1, alpha: float = 1.0, fit_intercept: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = float(alpha)
        self.fit_intercept = fit_intercept

        bound = 1.0 / math.sqrt(in_features) if in_features > 0 else 1.0
        w_vals = [random.uniform(-bound, bound) for _ in range(in_features * out_features)]
        self.weight = Parameter(Tensor(w_vals, shape=(in_features, out_features)))

        if fit_intercept:
            b_vals = [0.0 for _ in range(out_features)]
            self.bias = Parameter(Tensor(b_vals, shape=(out_features,)))
        else:
            self.bias = None

    def forward(self, X: Tensor) -> Tensor:
        out = X @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out

    def predict(self, X: Tensor) -> Tensor:
        self.eval()
        with no_grad():
            return self.forward(X)

    def get_config(self) -> Dict[str, Any]:
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "alpha": self.alpha,
            "fit_intercept": self.fit_intercept,
        }
