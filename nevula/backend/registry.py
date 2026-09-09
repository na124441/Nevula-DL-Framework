from typing import Any, Union
from nevula.backend.backend import Backend
from nevula.backend.cpu.backend import CPUBackend
from nevula.backend.device import Device

_BACKENDS: dict[str, Backend] = {
    "cpu": CPUBackend()
}


def get_backend(device: Union[str, Device]) -> Backend:
    """
    Retrieves the execution backend registered for the specified device.
    """
    device_type = device.type if isinstance(device, Device) else str(device).split(":")[0].strip().lower()

    if device_type not in _BACKENDS:
        if device_type == "cuda":
            from nevula.backend.cuda.backend import CUDABackend
            _BACKENDS["cuda"] = CUDABackend()
        else:
            raise RuntimeError(f"No backend available for device '{device}'")

    return _BACKENDS[device_type]


def register_backend(name: str, backend: Backend) -> None:
    """
    Registers a custom compute backend under a unique device name.
    """
    if not isinstance(backend, Backend):
        raise TypeError(f"Expected instance of Backend, got {type(backend).__name__}")
    _BACKENDS[name.lower()] = backend
