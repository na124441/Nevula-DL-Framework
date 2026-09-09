from typing import Any, Optional, Tuple
import numpy as np
from nevula.backend.backend import Backend
from nevula.backend.device import cuda_available

try:
    import cupy as cp
except ImportError:
    cp = None


class CUDABackend(Backend):
    """
    GPU execution backend implemented via CuPy.
    Provides zero-code-change GPU acceleration if CuPy and CUDA are installed,
    with a graceful fallback when CUDA hardware/runtime is not present.
    """

    def __init__(self):
        if not cuda_available() or cp is None:
            self._available = False
        else:
            self._available = True

    def _ensure_available(self) -> None:
        if not self._available:
            raise RuntimeError(
                "CUDA backend is not available. Please install 'cupy' and ensure an NVIDIA GPU with drivers is present."
            )

    def create(self, data: Any, shape: Optional[Tuple[int, ...]] = None) -> Any:
        if not self._available:
            from nevula.backend.cpu.backend import CPUStorage
            if isinstance(data, np.ndarray):
                arr = data.astype(np.float64, copy=False)
            else:
                arr = np.array(data, dtype=np.float64)
            if shape is not None and arr.shape != tuple(shape):
                arr = arr.reshape(shape)
            return arr.view(CPUStorage)

        arr = cp.asarray(data, dtype=cp.float64)
        if shape is not None and arr.shape != tuple(shape):
            arr = arr.reshape(shape)
        return arr

    def add(self, a: Any, b: Any) -> Any:
        self._ensure_available()
        return cp.add(a, b)

    def sub(self, a: Any, b: Any) -> Any:
        self._ensure_available()
        return cp.subtract(a, b)

    def mul(self, a: Any, b: Any) -> Any:
        self._ensure_available()
        return cp.multiply(a, b)

    def div(self, a: Any, b: Any) -> Any:
        self._ensure_available()
        return cp.divide(a, b)

    def pow(self, a: Any, p: Any) -> Any:
        self._ensure_available()
        return cp.power(a, p)

    def neg(self, a: Any) -> Any:
        self._ensure_available()
        return cp.negative(a)

    def matmul(self, a: Any, b: Any) -> Any:
        self._ensure_available()
        return cp.matmul(a, b)

    def sum(self, a: Any, axis: Optional[Any] = None, keepdims: bool = False) -> Any:
        self._ensure_available()
        return cp.sum(a, axis=axis, keepdims=keepdims)

    def mean(self, a: Any, axis: Optional[Any] = None, keepdims: bool = False) -> Any:
        self._ensure_available()
        return cp.mean(a, axis=axis, keepdims=keepdims)

    def reshape(self, a: Any, shape: Tuple[int, ...]) -> Any:
        self._ensure_available()
        return cp.reshape(a, shape)

    def transpose(self, a: Any, axis1: int = -2, axis2: int = -1) -> Any:
        self._ensure_available()
        if getattr(a, "ndim", len(getattr(a, "shape", ()))) < 2:
            return a
        return cp.swapaxes(a, axis1, axis2)

    def to_numpy(self, a: Any) -> Any:
        if not self._available:
            return np.asarray(a)
        return cp.asnumpy(a)

    def from_numpy(self, a: Any) -> Any:
        if not self._available:
            from nevula.backend.cpu.backend import CPUStorage
            return np.asarray(a).view(CPUStorage)
        return cp.asarray(a)

