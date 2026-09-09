from typing import Any, Union
from nevula.backend.device import Device
from nevula.backend.registry import get_backend
from nevula.core.tensor import Tensor


class Parameter(Tensor):
    """
    A Tensor that is considered a model parameter.
    Parameters are automatically registered in Module instances and have
    requires_grad=True by default.
    """

    def __init__(self, data: Any, requires_grad: bool = True, device: Union[str, Device] = "cpu"):
        if isinstance(data, Tensor):
            dev = data.device if device == "cpu" and data.device != "cpu" else device
            super().__init__(
                data.data,
                shape=data.shape,
                strides=data.strides,
                offset=data.offset,
                _clone=False,
                requires_grad=requires_grad,
                device=dev,
            )
        else:
            super().__init__(data, requires_grad=requires_grad, device=device)

    def to(self, device: Union[str, Device]) -> 'Parameter':
        """
        Migrates parameter to target device in-place and returns self.
        """
        target_device = Device(device)
        if self.device == target_device:
            return self

        src_backend = get_backend(self.device.type)
        dst_backend = get_backend(target_device.type)

        src_arr = self.to_array()
        np_arr = src_backend.to_numpy(src_arr)
        dst_data = dst_backend.from_numpy(np_arr)

        self.data = dst_data.flatten()
        self.device = target_device
        if self.grad is not None:
            self.grad = self.grad.to(target_device)
        return self

    def __repr__(self) -> str:
        return f"Parameter containing:\n{super().__repr__()}"

