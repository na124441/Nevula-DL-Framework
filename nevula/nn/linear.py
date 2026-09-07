import math
import random
from typing import Any
from nevula.core.tensor import Tensor
from nevula.nn.module import Module
from nevula.nn.parameter import Parameter


class Linear(Module):
    """
    Applies an affine linear transformation to incoming data:
        y = x @ W + b
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights uniformly in [-bound, bound] where bound = 1 / sqrt(in_features)
        bound = 1.0 / math.sqrt(in_features) if in_features > 0 else 1.0
        
        weight_data = [random.uniform(-bound, bound) for _ in range(in_features * out_features)]
        self.weight = Parameter(Tensor(weight_data, shape=(in_features, out_features)))

        if bias:
            bias_data = [random.uniform(-bound, bound) for _ in range(out_features)]
            self.bias = Parameter(Tensor(bias_data, shape=(out_features,)))
        else:
            self.bias = None

    def forward(self, x: Any) -> Any:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self) -> str:
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"
