from typing import Any, Callable, Iterator, Optional, Sequence
from nevula.core.tensor import Tensor
from nevula.data.dataset import Dataset
from nevula.data.sampler import Sampler, SequentialSampler, RandomSampler, BatchSampler


def default_collate(batch: Sequence[Any]) -> Any:
    """
    Puts each sub-element of samples into a batched Tensor with outer dimension size = len(batch).
    Handles Tensors, tuples, lists, numbers, and dictionaries.
    """
    if len(batch) == 0:
        return Tensor([])

    elem = batch[0]
    elem_type = type(elem)

    if isinstance(elem, Tensor):
        out_shape = (len(batch),) + elem.shape
        batch_data = [t.to_list() for t in batch]
        return Tensor(batch_data, shape=out_shape)

    elif isinstance(elem, (int, float)):
        return Tensor(list(batch))

    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))

    elif isinstance(elem, (tuple, list)):
        transposed = zip(*batch)
        return elem_type(default_collate(samples) for samples in transposed)

    elif isinstance(elem, dict):
        return {key: default_collate([d[key] for d in batch]) for key in elem}

    return batch


class DataLoader:
    """
    Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.

    Args:
        dataset: Dataset from which to load the data.
        batch_size: How many samples per batch to load (default: 1).
        shuffle: Set to True to have the data reshuffled at every epoch (default: False).
        sampler: Defines the strategy to draw samples from the dataset.
        batch_sampler: Like sampler, but returns a batch of indices at a time.
        collate_fn: Merges a list of samples to form a mini-batch of Tensor(s).
        drop_last: Set to True to drop the last incomplete batch if the dataset size
            is not divisible by the batch size (default: False).
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: Optional[Sampler] = None,
        batch_sampler: Optional[BatchSampler] = None,
        collate_fn: Optional[Callable[[list[Any]], Any]] = None,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.collate_fn = collate_fn if collate_fn is not None else default_collate

        if batch_sampler is not None:
            if batch_size > 1 or shuffle or sampler is not None or drop_last:
                raise ValueError("batch_sampler option is mutually exclusive with "
                                 "batch_size, shuffle, sampler, and drop_last")
            self.batch_sampler = batch_sampler
        else:
            if sampler is not None and shuffle:
                raise ValueError("sampler option is mutually exclusive with shuffle")

            if sampler is None:
                if shuffle:
                    sampler = RandomSampler(dataset)
                else:
                    sampler = SequentialSampler(dataset)

            self.batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=drop_last)

    def __iter__(self) -> Iterator[Any]:
        for batch_indices in self.batch_sampler:
            batch = [self.dataset[idx] for idx in batch_indices]
            yield self.collate_fn(batch)

    def __len__(self) -> int:
        return len(self.batch_sampler)
