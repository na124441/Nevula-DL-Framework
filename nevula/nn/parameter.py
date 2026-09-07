from typing import Any
from nevula.core.tensor import Tensor


class Parameter(Tensor):
    """
    A Tensor that is considered a model parameter.
    Parameters are automatically registered in Module instances and have
    requires_grad=True by default.
    """

    def __init__(self, data: Any, requires_grad: bool = True):
        if isinstance(data, Tensor):
            super().__init__(
                data.data,
                shape=data.shape,
                strides=data.strides,
                offset=data.offset,
                _clone=False,
                requires_grad=requires_grad,
            )
        else:
            super().__init__(data, requires_grad=requires_grad)

    def __repr__(self) -> str:
        return f"Parameter containing:\n{super().__repr__()}"
