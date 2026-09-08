import random
from typing import Any, Iterator, Optional, Sequence


class Sampler:
    """
    Base class for all Samplers.
    Every Sampler subclass has to provide an `__iter__()` method, providing a
    way to iterate over indices of dataset elements, and a `__len__()` method
    that returns the length of the returned iterators.
    """

    def __init__(self, data_source: Optional[Any] = None):
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class SequentialSampler(Sampler):
    """
    Samples elements sequentially, always in the same order.

    Args:
        data_source: Dataset to sample from.
    """

    def __init__(self, data_source: Sequence[Any]):
        super().__init__(data_source)

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)


class RandomSampler(Sampler):
    """
    Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify num_samples to draw.

    Args:
        data_source: Dataset to sample from.
        replacement: Samples are drawn on-demand with replacement if True (default: False).
        num_samples: Number of samples to draw (default: len(dataset)).
    """

    def __init__(
        self,
        data_source: Sequence[Any],
        replacement: bool = False,
        num_samples: Optional[int] = None,
    ):
        super().__init__(data_source)
        self.replacement = replacement
        self._num_samples = num_samples

        if not isinstance(self.replacement, bool):
            raise TypeError(f"replacement should be a boolean, but got replacement={self.replacement}")

        if self._num_samples is not None and not isinstance(self._num_samples, int):
            raise TypeError(f"num_samples should be an integer, but got num_samples={self._num_samples}")

    @property
    def num_samples(self) -> int:
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.replacement:
            for _ in range(self.num_samples):
                yield random.randint(0, n - 1)
        else:
            indices = list(range(n))
            random.shuffle(indices)
            yield from indices[:self.num_samples]

    def __len__(self) -> int:
        return self.num_samples


class BatchSampler(Sampler):
    """
    Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler: Base sampler.
        batch_size: Size of mini-batch.
        drop_last: If True, the sampler will drop the last batch if
            its size would be less than `batch_size` (default: False).
    """

    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool = False):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError(f"batch_size should be a positive integer value, but got batch_size={batch_size}")
        if not isinstance(drop_last, bool):
            raise ValueError(f"drop_last should be a boolean value, but got drop_last={drop_last}")

        super().__init__(sampler)
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[list[int]]:
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
