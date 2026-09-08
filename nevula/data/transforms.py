from typing import Any, Callable, Sequence, Union
from nevula.core.tensor import Tensor


class Compose:
    """
    Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """

    def __init__(self, transforms: Sequence[Callable[[Any], Any]]):
        self.transforms = list(transforms)

    def __call__(self, img: Any) -> Any:
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += f"\n    {t}"
        format_string += "\n)"
        return format_string


class ToTensor:
    """
    Converts a Python list, tuple, or scalar to a Tensor.
    """

    def __call__(self, pic: Any) -> Tensor:
        if isinstance(pic, Tensor):
            return pic
        return Tensor(pic)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Normalize:
    """
    Normalize a tensor image with mean and standard deviation.
    Given mean: (mean[1],...,mean[n]) and std: (std[1],..,std[n]) for n channels,
    this transform will normalize each channel:
        output[channel] = (input[channel] - mean[channel]) / std[channel]
    """

    def __init__(self, mean: Union[float, Sequence[float]], std: Union[float, Sequence[float]]):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: Tensor) -> Tensor:
        if not isinstance(tensor, Tensor):
            tensor = Tensor(tensor)

        mean_t = Tensor(self.mean)
        std_t = Tensor(self.std)

        return (tensor - mean_t) / std_t

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class Lambda:
    """
    Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd: Callable[[Any], Any]):
        if not callable(lambd):
            raise TypeError("Argument lambd should be callable")
        self.lambd = lambd

    def __call__(self, img: Any) -> Any:
        return self.lambd(img)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
