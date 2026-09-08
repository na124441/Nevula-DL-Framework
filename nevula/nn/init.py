import math
import random
from typing import Optional, Tuple
from nevula.core.tensor import Tensor, _prod


def _calculate_fan_in_and_fan_out(tensor: Tensor) -> Tuple[int, int]:
    """
    Calculates the number of input and output units for a tensor.
    For 2D matrices (in_features, out_features): fan_in = shape[0], fan_out = shape[1].
    For >2D convolutions (out_channels, in_channels, ...): fan_in = shape[1] * k_size, fan_out = shape[0] * k_size.
    """
    ndim = len(tensor.shape)
    if ndim < 1:
        return 1, 1
    if ndim == 1:
        return tensor.shape[0], tensor.shape[0]
    if ndim == 2:
        # Standard Linear layer weight matrix: (in_features, out_features)
        return tensor.shape[0], tensor.shape[1]

    # Conv layers: (out_channels, in_channels, *kernel_size)
    num_input_fmaps = tensor.shape[1]
    num_output_fmaps = tensor.shape[0]
    receptive_field_size = _prod(tensor.shape[2:])

    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def calculate_gain(nonlinearity: str, param: Optional[float] = None) -> float:
    """
    Returns the recommended gain value for the given nonlinearity function.
    """
    linear_fns = ["linear", "conv1d", "conv2d", "conv3d"]
    if nonlinearity in linear_fns or nonlinearity == "sigmoid":
        return 1.0
    elif nonlinearity == "tanh":
        return 5.0 / 3.0
    elif nonlinearity == "relu":
        return math.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        negative_slope = param if param is not None else 0.01
        return math.sqrt(2.0 / (1.0 + negative_slope ** 2))
    elif nonlinearity == "selu":
        return 3.0 / 4.0
    else:
        raise ValueError(f"Unsupported nonlinearity {nonlinearity}")


def zeros_(tensor: Tensor) -> Tensor:
    """Fills the input Tensor with the scalar value 0."""
    for i in range(len(tensor.data)):
        tensor.data[i] = 0.0
    return tensor


def ones_(tensor: Tensor) -> Tensor:
    """Fills the input Tensor with the scalar value 1."""
    for i in range(len(tensor.data)):
        tensor.data[i] = 1.0
    return tensor


def constant_(tensor: Tensor, val: float) -> Tensor:
    """Fills the input Tensor with the value `val`."""
    val_float = float(val)
    for i in range(len(tensor.data)):
        tensor.data[i] = val_float
    return tensor


def uniform_(tensor: Tensor, a: float = 0.0, b: float = 1.0) -> Tensor:
    """Fills the input Tensor with values drawn from uniform distribution U(a, b)."""
    for i in range(len(tensor.data)):
        tensor.data[i] = random.uniform(a, b)
    return tensor


def normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0) -> Tensor:
    """Fills the input Tensor with values drawn from normal distribution N(mean, std^2)."""
    for i in range(len(tensor.data)):
        tensor.data[i] = random.gauss(mean, std)
    return tensor


def xavier_uniform_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """
    Fills the input Tensor with values according to the Xavier uniform distribution.
    Also known as Glorot initialization.
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bound from standard deviation
    return uniform_(tensor, -a, a)


def xavier_normal_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """
    Fills the input Tensor with values according to the Xavier normal distribution.
    Also known as Glorot initialization.
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    return normal_(tensor, 0.0, std)


def kaiming_uniform_(
    tensor: Tensor,
    a: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "relu"
) -> Tensor:
    """
    Fills the input Tensor with values according to the Kaiming uniform distribution.
    Also known as He initialization.
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    fan = fan_in if mode == "fan_in" else fan_out
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    return uniform_(tensor, -bound, bound)


def kaiming_normal_(
    tensor: Tensor,
    a: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "relu"
) -> Tensor:
    """
    Fills the input Tensor with values according to the Kaiming normal distribution.
    Also known as He initialization.
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    fan = fan_in if mode == "fan_in" else fan_out
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return normal_(tensor, 0.0, std)
