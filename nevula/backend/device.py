from typing import Any, Union


class Device:
    """
    Represents the compute device on which a Tensor is or will be allocated.
    Currently supports 'cpu' and 'cuda'.
    """

    def __init__(self, device_type: Union[str, 'Device']):
        if isinstance(device_type, Device):
            self.type = device_type.type
            return

        if not isinstance(device_type, str):
            raise TypeError(f"Expected str or Device, but got {type(device_type).__name__}")

        # Parse device string (handles e.g. "cuda", "cuda:0", "cpu")
        base_type = device_type.split(":")[0].strip().lower()
        if base_type not in ("cpu", "cuda"):
            raise ValueError(f"Unsupported device: {device_type}")

        self.type = base_type

    def __repr__(self) -> str:
        return f"Device('{self.type}')"

    def __str__(self) -> str:
        return self.type

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            base_other = other.split(":")[0].strip().lower()
            return self.type == base_other
        if not isinstance(other, Device):
            return False
        return self.type == other.type


def cuda_available() -> bool:
    """
    Returns True if CUDA support is available via CuPy or CUDA runtime.
    """
    try:
        import cupy as cp
        return bool(cp.cuda.is_available())
    except (ImportError, Exception):
        return False
