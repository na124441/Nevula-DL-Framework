from typing import Any
from nevula.nn.module import Module


class ReLU(Module):
    """
    Applies the rectified linear unit activation function elementwise:
        ReLU(x) = max(0, x)
    """

    def forward(self, x: Any) -> Any:
        return x.relu()

    def __repr__(self) -> str:
        return "ReLU()"


class Sigmoid(Module):
    """
    Applies the sigmoid activation function elementwise:
        Sigmoid(x) = 1 / (1 + exp(-x))
    """

    def forward(self, x: Any) -> Any:
        return x.sigmoid()

    def __repr__(self) -> str:
        return "Sigmoid()"


class Tanh(Module):
    """
    Applies the hyperbolic tangent activation function elementwise:
        Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    """

    def forward(self, x: Any) -> Any:
        return x.tanh()

    def __repr__(self) -> str:
        return "Tanh()"
