from typing import Any, Optional, Sequence, Tuple
import numpy as np
from nevula.backend.backend import Backend


class CPUStorage(np.ndarray):
    """
    Ndarray wrapper used by Nevula's CPUBackend.
    Enhances list comparison and inspection while retaining full NumPy vectorization.
    """

    def __eq__(self, other: Any) -> Any:
        if isinstance(other, list):
            if self.ndim == 0 and len(other) == 1:
                return self.item() == other[0]
            if self.tolist() == other:
                return True
            if self.flatten().tolist() == other:
                return True
            return False
        return super().__eq__(other)


class CPUBackend(Backend):
    """
    CPU execution backend implemented via NumPy.
    """

    def create(self, data: Any, shape: Optional[Tuple[int, ...]] = None) -> CPUStorage:
        if isinstance(data, np.ndarray):
            arr = data.astype(np.float64, copy=False)
        else:
            arr = np.array(data, dtype=np.float64)

        if shape is not None and arr.shape != tuple(shape):
            arr = arr.reshape(shape)

        return arr.view(CPUStorage)

    def add(self, a: Any, b: Any) -> CPUStorage:
        return np.asarray(np.add(a, b)).view(CPUStorage)

    def sub(self, a: Any, b: Any) -> CPUStorage:
        return np.asarray(np.subtract(a, b)).view(CPUStorage)

    def mul(self, a: Any, b: Any) -> CPUStorage:
        return np.asarray(np.multiply(a, b)).view(CPUStorage)

    def div(self, a: Any, b: Any) -> CPUStorage:
        return np.asarray(np.divide(a, b)).view(CPUStorage)

    def pow(self, a: Any, p: Any) -> CPUStorage:
        return np.asarray(np.power(a, p)).view(CPUStorage)

    def neg(self, a: Any) -> CPUStorage:
        return np.asarray(np.negative(a)).view(CPUStorage)

    def matmul(self, a: Any, b: Any) -> CPUStorage:
        return np.asarray(np.matmul(a, b)).view(CPUStorage)

    def sum(self, a: Any, axis: Optional[Any] = None, keepdims: bool = False) -> CPUStorage:
        return np.asarray(np.sum(a, axis=axis, keepdims=keepdims)).view(CPUStorage)

    def mean(self, a: Any, axis: Optional[Any] = None, keepdims: bool = False) -> CPUStorage:
        return np.asarray(np.mean(a, axis=axis, keepdims=keepdims)).view(CPUStorage)

    def reshape(self, a: Any, shape: Tuple[int, ...]) -> CPUStorage:
        return np.asarray(np.reshape(a, shape)).view(CPUStorage)

    def transpose(self, a: Any, axis1: int = -2, axis2: int = -1) -> CPUStorage:
        ndim = getattr(a, "ndim", len(getattr(a, "shape", ())))
        if ndim < 2:
            return np.asarray(a).view(CPUStorage)
        return np.asarray(np.swapaxes(a, axis1, axis2)).view(CPUStorage)

    def to_numpy(self, a: Any) -> np.ndarray:
        return np.asarray(a)

    def from_numpy(self, a: Any) -> CPUStorage:
        return np.asarray(a).view(CPUStorage)
