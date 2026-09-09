from typing import Any, Optional, Sequence, Tuple


class Backend:
    """
    Abstract interface for compute backends (CPU, CUDA, etc.).
    Defines numerical and array operations performed on raw storage buffers.
    """

    def create(self, data: Any, shape: Optional[Tuple[int, ...]] = None) -> Any:
        raise NotImplementedError

    def add(self, a: Any, b: Any) -> Any:
        raise NotImplementedError

    def sub(self, a: Any, b: Any) -> Any:
        raise NotImplementedError

    def mul(self, a: Any, b: Any) -> Any:
        raise NotImplementedError

    def div(self, a: Any, b: Any) -> Any:
        raise NotImplementedError

    def pow(self, a: Any, p: Any) -> Any:
        raise NotImplementedError

    def neg(self, a: Any) -> Any:
        raise NotImplementedError

    def matmul(self, a: Any, b: Any) -> Any:
        raise NotImplementedError

    def sum(self, a: Any, axis: Optional[Any] = None, keepdims: bool = False) -> Any:
        raise NotImplementedError

    def mean(self, a: Any, axis: Optional[Any] = None, keepdims: bool = False) -> Any:
        raise NotImplementedError

    def reshape(self, a: Any, shape: Tuple[int, ...]) -> Any:
        raise NotImplementedError

    def transpose(self, a: Any, axis1: int = -2, axis2: int = -1) -> Any:
        raise NotImplementedError

    def to_numpy(self, a: Any) -> Any:
        raise NotImplementedError

    def from_numpy(self, a: Any) -> Any:
        raise NotImplementedError
