from typing import Any, Sequence
from nevula.core.tensor import Tensor


class Dataset:
    """
    An abstract class representing a Dataset.
    All datasets that represent a map from keys to data samples should subclass it.
    All subclasses should overwrite `__getitem__`, supporting fetching a
    data sample for a given key, and `__len__`, expected to return the size of the dataset.
    """

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class TensorDataset(Dataset):
    """
    Dataset wrapping tensors.
    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        *tensors (Tensor): Tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors: Tensor):
        if len(tensors) == 0:
            raise ValueError("TensorDataset requires at least one tensor")

        first_len = tensors[0].shape[0] if len(tensors[0].shape) > 0 else 1
        for t in tensors:
            t_len = t.shape[0] if len(t.shape) > 0 else 1
            if t_len != first_len:
                raise ValueError(f"Size mismatch among tensors: expected first dimension {first_len}, got {t_len}")

        self.tensors = tuple(tensors)

    def __len__(self) -> int:
        return self.tensors[0].shape[0] if len(self.tensors[0].shape) > 0 else 1

    def __getitem__(self, index: int) -> Any:
        if len(self.tensors) == 1:
            return self.tensors[0][index]
        return tuple(tensor[index] for tensor in self.tensors)
