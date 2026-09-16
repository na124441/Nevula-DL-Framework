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


class LogSigmoid(Module):
    """
    Applies the log-sigmoid activation function elementwise:
        LogSigmoid(x) = log(1 / (1 + exp(-x)))
    """

    def forward(self, x: Any) -> Any:
        return x.log_sigmoid()

    def __repr__(self) -> str:
        return "LogSigmoid()"


class Softmax(Module):
    """
    Applies the Softmax function to an n-dimensional input Tensor,
    rescaling elements along dim so that they lie in range [0, 1] and sum to 1:
        Softmax(x_i) = exp(x_i) / sum_j exp(x_j)
    """

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: Any) -> Any:
        return x.softmax(dim=self.dim)

    def __repr__(self) -> str:
        return f"Softmax(dim={self.dim})"


class LogSoftmax(Module):
    """
    Applies the LogSoftmax function to an n-dimensional input Tensor:
        LogSoftmax(x_i) = log(exp(x_i) / sum_j exp(x_j))
    """

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: Any) -> Any:
        return x.log_softmax(dim=self.dim)

    def __repr__(self) -> str:
        return f"LogSoftmax(dim={self.dim})"


