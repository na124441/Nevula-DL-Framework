"""
Cross-validation generators and data splitting utilities.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union
import itertools
import math
import numpy as np

from nevula.model_selection.utils import to_numpy, safe_indexing, is_classifier


def _num_samples(x: Any) -> int:
    """Returns number of samples in an array, tensor, list, or dataset."""
    if x is None:
        return 0
    if hasattr(x, "shape") and len(x.shape) > 0:
        return int(x.shape[0])
    if hasattr(x, "__len__"):
        return len(x)
    raise TypeError(f"Cannot determine sample count for object of type {type(x)}")


def _validate_shuffle_split(
    n_samples: int,
    test_size: Optional[Union[float, int]],
    train_size: Optional[Union[float, int]],
    default_test_size: float = 0.1,
) -> Tuple[int, int]:
    """Validates and resolves train_size and test_size counts."""
    if test_size is None and train_size is None:
        test_size = default_test_size

    if test_size is not None:
        if isinstance(test_size, float):
            if not 0.0 < test_size < 1.0:
                raise ValueError(f"test_size={test_size} should be in (0.0, 1.0)")
            n_test = int(math.ceil(test_size * n_samples))
        elif isinstance(test_size, (int, np.integer)):
            if not 0 < test_size < n_samples:
                raise ValueError(f"test_size={test_size} should be between 1 and {n_samples - 1}")
            n_test = int(test_size)
        else:
            raise TypeError(f"Invalid test_size type: {type(test_size)}")
    else:
        n_test = None

    if train_size is not None:
        if isinstance(train_size, float):
            if not 0.0 < train_size < 1.0:
                raise ValueError(f"train_size={train_size} should be in (0.0, 1.0)")
            n_train = int(math.floor(train_size * n_samples))
        elif isinstance(train_size, (int, np.integer)):
            if not 0 < train_size < n_samples:
                raise ValueError(f"train_size={train_size} should be between 1 and {n_samples - 1}")
            n_train = int(train_size)
        else:
            raise TypeError(f"Invalid train_size type: {type(train_size)}")
    else:
        n_train = None

    if n_test is None:
        n_test = n_samples - n_train
    if n_train is None:
        n_train = n_samples - n_test

    if n_train + n_test > n_samples:
        raise ValueError(
            f"The sum of train_size ({n_train}) and test_size ({n_test}) "
            f"is larger than the total number of samples ({n_samples})."
        )

    return n_train, n_test


class BaseCrossValidator(ABC):
    """Abstract base class for all cross-validation splitters in Nevula."""

    @abstractmethod
    def split(
        self,
        X: Any,
        y: Optional[Any] = None,
        groups: Optional[Any] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generates indices to split data into training and test sets.

        Args:
            X: Training data.
            y: Target values.
            groups: Group labels for samples.

        Yields:
            train_indices (np.ndarray): The training set indices for that split.
            test_indices (np.ndarray): The testing set indices for that split.
        """
        raise NotImplementedError

    @abstractmethod
    def get_n_splits(
        self,
        X: Optional[Any] = None,
        y: Optional[Any] = None,
        groups: Optional[Any] = None,
    ) -> int:
        """Returns the number of splitting iterations in the cross-validator."""
        raise NotImplementedError


class KFold(BaseCrossValidator):
    """
    K-Fold cross-validator.

    Provides train/test indices to split data in train/test sets. Split dataset
    into k consecutive folds (without shuffling by default). Each fold is then
    used once as a validation while the k - 1 remaining folds form the training set.

    Args:
        n_splits: Number of folds. Must be at least 2.
        shuffle: Whether to shuffle the data before splitting into batches.
        random_state: Random state seed when shuffle is True.
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ):
        if not isinstance(n_splits, (int, np.integer)) or n_splits <= 1:
            raise ValueError(f"k-fold cross-validation requires at least 2 splits; got {n_splits}")
        self.n_splits = int(n_splits)
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(
        self,
        X: Optional[Any] = None,
        y: Optional[Any] = None,
        groups: Optional[Any] = None,
    ) -> int:
        return self.n_splits

    def split(
        self,
        X: Any,
        y: Optional[Any] = None,
        groups: Optional[Any] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n_samples = _num_samples(X)
        if self.n_splits > n_samples:
            raise ValueError(
                f"Cannot have number of splits n_splits={self.n_splits} greater "
                f"than the number of samples: {n_samples}."
            )

        indices = np.arange(n_samples)
        if self.shuffle:
            rng = np.random.RandomState(self.random_state) if not isinstance(self.random_state, np.random.RandomState) else self.random_state
            rng.shuffle(indices)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_idx = indices[start:stop]
            train_idx = np.concatenate([indices[:start], indices[stop:]])
            yield train_idx, test_idx
            current = stop

    def __repr__(self) -> str:
        return (
            f"KFold(n_splits={self.n_splits}, shuffle={self.shuffle}, "
            f"random_state={self.random_state})"
        )


class StratifiedKFold(BaseCrossValidator):
    """
    Stratified K-Fold cross-validator.

    Provides train/test indices to split data in train/test sets. Each fold
    contains approximately the same percentage of samples of each target class
    as the complete dataset.

    Args:
        n_splits: Number of folds. Must be at least 2.
        shuffle: Whether to shuffle each class's samples before splitting into batches.
        random_state: Random state seed when shuffle is True.
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ):
        if not isinstance(n_splits, (int, np.integer)) or n_splits <= 1:
            raise ValueError(f"k-fold cross-validation requires at least 2 splits; got {n_splits}")
        self.n_splits = int(n_splits)
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(
        self,
        X: Optional[Any] = None,
        y: Optional[Any] = None,
        groups: Optional[Any] = None,
    ) -> int:
        return self.n_splits

    def split(
        self,
        X: Any,
        y: Any,
        groups: Optional[Any] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if y is None:
            raise ValueError("StratifiedKFold requires target labels y for stratification.")

        n_samples = _num_samples(X)
        y_arr = to_numpy(y).ravel()
        if len(y_arr) != n_samples:
            raise ValueError(f"Length mismatch: X has {n_samples} samples, y has {len(y_arr)}.")

        if self.n_splits > n_samples:
            raise ValueError(
                f"Cannot have number of splits n_splits={self.n_splits} greater "
                f"than the number of samples: {n_samples}."
            )

        rng = np.random.RandomState(self.random_state) if not isinstance(self.random_state, np.random.RandomState) else self.random_state

        unique_classes, y_inversed = np.unique(y_arr, return_inverse=True)
        n_classes = len(unique_classes)

        # Allocate sample indices per fold per class
        test_folds = np.zeros(n_samples, dtype=int)

        for c_idx in range(n_classes):
            cls_indices = np.where(y_inversed == c_idx)[0]
            n_cls_samples = len(cls_indices)
            if self.shuffle:
                rng.shuffle(cls_indices)

            cls_fold_sizes = np.full(self.n_splits, n_cls_samples // self.n_splits, dtype=int)
            cls_fold_sizes[: n_cls_samples % self.n_splits] += 1

            current = 0
            for fold_idx, size in enumerate(cls_fold_sizes):
                start, stop = current, current + size
                test_folds[cls_indices[start:stop]] = fold_idx
                current = stop

        indices = np.arange(n_samples)
        for fold_idx in range(self.n_splits):
            test_mask = (test_folds == fold_idx)
            test_idx = indices[test_mask]
            train_idx = indices[~test_mask]
            yield train_idx, test_idx

    def __repr__(self) -> str:
        return (
            f"StratifiedKFold(n_splits={self.n_splits}, shuffle={self.shuffle}, "
            f"random_state={self.random_state})"
        )


class LeaveOneOut(BaseCrossValidator):
    """
    Leave-One-Out (LOO) cross-validator.

    Provides train/test indices to split data in train/test sets. Each sample
    is used once as an individual test set (singleton) while the remaining
    samples form the training set.
    """

    def get_n_splits(
        self,
        X: Optional[Any] = None,
        y: Optional[Any] = None,
        groups: Optional[Any] = None,
    ) -> int:
        if X is None:
            raise ValueError("LeaveOneOut requires X to compute number of splits.")
        return _num_samples(X)

    def split(
        self,
        X: Any,
        y: Optional[Any] = None,
        groups: Optional[Any] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n_samples = _num_samples(X)
        if n_samples <= 1:
            raise ValueError(f"LeaveOneOut requires at least 2 samples; got {n_samples}")
        indices = np.arange(n_samples)
        for i in range(n_samples):
            test_idx = np.array([i], dtype=int)
            train_idx = np.delete(indices, i)
            yield train_idx, test_idx

    def __repr__(self) -> str:
        return "LeaveOneOut()"


class LeavePOut(BaseCrossValidator):
    """
    Leave-P-Out (LPO) cross-validator.

    Provides train/test indices to split data into train/test sets by taking
    all permutations of p samples out of N samples as test sets.

    Args:
        p: Size of the test sets.
    """

    def __init__(self, p: int):
        if not isinstance(p, (int, np.integer)) or p <= 0:
            raise ValueError(f"p must be a positive integer; got {p}")
        self.p = int(p)

    def get_n_splits(
        self,
        X: Optional[Any] = None,
        y: Optional[Any] = None,
        groups: Optional[Any] = None,
    ) -> int:
        if X is None:
            raise ValueError("LeavePOut requires X to compute number of splits.")
        n_samples = _num_samples(X)
        if self.p >= n_samples:
            raise ValueError(f"p={self.p} must be strictly less than n_samples={n_samples}")
        return int(math.comb(n_samples, self.p))

    def split(
        self,
        X: Any,
        y: Optional[Any] = None,
        groups: Optional[Any] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n_samples = _num_samples(X)
        if self.p >= n_samples:
            raise ValueError(f"p={self.p} must be strictly less than n_samples={n_samples}")
        indices = np.arange(n_samples)
        for test_comb in itertools.combinations(range(n_samples), self.p):
            test_idx = np.array(test_comb, dtype=int)
            train_idx = np.setdiff1d(indices, test_idx)
            yield train_idx, test_idx

    def __repr__(self) -> str:
        return f"LeavePOut(p={self.p})"


class ShuffleSplit(BaseCrossValidator):
    """
    Random permutation cross-validator (Monte Carlo cross-validation).

    Yields indices to split data into train and test sets by generating
    random permutations with fixed test and train fractions/sizes.

    Args:
        n_splits: Number of re-shuffling & splitting iterations.
        test_size: Should be between 0.0 and 1.0 (fraction) or an int.
        train_size: Should be between 0.0 and 1.0 (fraction) or an int.
        random_state: Pseudo-random number generator seed.
    """

    def __init__(
        self,
        n_splits: int = 10,
        test_size: Optional[Union[float, int]] = 0.1,
        train_size: Optional[Union[float, int]] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ):
        if not isinstance(n_splits, (int, np.integer)) or n_splits <= 0:
            raise ValueError(f"n_splits must be positive; got {n_splits}")
        self.n_splits = int(n_splits)
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state

    def get_n_splits(
        self,
        X: Optional[Any] = None,
        y: Optional[Any] = None,
        groups: Optional[Any] = None,
    ) -> int:
        return self.n_splits

    def split(
        self,
        X: Any,
        y: Optional[Any] = None,
        groups: Optional[Any] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n_samples = _num_samples(X)
        n_train, n_test = _validate_shuffle_split(n_samples, self.test_size, self.train_size, 0.1)
        rng = np.random.RandomState(self.random_state) if not isinstance(self.random_state, np.random.RandomState) else self.random_state

        for _ in range(self.n_splits):
            permutation = rng.permutation(n_samples)
            test_idx = permutation[:n_test]
            train_idx = permutation[n_test : n_test + n_train]
            yield train_idx, test_idx

    def __repr__(self) -> str:
        return (
            f"ShuffleSplit(n_splits={self.n_splits}, test_size={self.test_size}, "
            f"train_size={self.train_size}, random_state={self.random_state})"
        )


class StratifiedShuffleSplit(BaseCrossValidator):
    """
    Stratified ShuffleSplit cross-validator.

    Provides train/test indices to split data in train/test sets with
    proportional class representation in each random iteration.

    Args:
        n_splits: Number of re-shuffling & splitting iterations.
        test_size: Should be between 0.0 and 1.0 (fraction) or an int.
        train_size: Should be between 0.0 and 1.0 (fraction) or an int.
        random_state: Pseudo-random number generator seed.
    """

    def __init__(
        self,
        n_splits: int = 10,
        test_size: Optional[Union[float, int]] = 0.1,
        train_size: Optional[Union[float, int]] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ):
        if not isinstance(n_splits, (int, np.integer)) or n_splits <= 0:
            raise ValueError(f"n_splits must be positive; got {n_splits}")
        self.n_splits = int(n_splits)
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state

    def get_n_splits(
        self,
        X: Optional[Any] = None,
        y: Optional[Any] = None,
        groups: Optional[Any] = None,
    ) -> int:
        return self.n_splits

    def split(
        self,
        X: Any,
        y: Any,
        groups: Optional[Any] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if y is None:
            raise ValueError("StratifiedShuffleSplit requires target labels y.")

        n_samples = _num_samples(X)
        y_arr = to_numpy(y).ravel()
        if len(y_arr) != n_samples:
            raise ValueError(f"Length mismatch: X has {n_samples} samples, y has {len(y_arr)}.")

        n_train, n_test = _validate_shuffle_split(n_samples, self.test_size, self.train_size, 0.1)
        rng = np.random.RandomState(self.random_state) if not isinstance(self.random_state, np.random.RandomState) else self.random_state

        classes, y_indices = np.unique(y_arr, return_inverse=True)
        n_classes = len(classes)
        class_counts = np.bincount(y_indices)

        for _ in range(self.n_splits):
            train_idx_list: List[np.ndarray] = []
            test_idx_list: List[np.ndarray] = []

            for c in range(n_classes):
                cls_samples = np.where(y_indices == c)[0]
                n_c = len(cls_samples)
                # Compute proportional allocation
                n_c_test = int(round(n_test * (n_c / n_samples)))
                n_c_train = int(round(n_train * (n_c / n_samples)))
                # Bounds check
                n_c_test = max(1, min(n_c_test, n_c - 1)) if n_c > 1 else 0
                n_c_train = max(1, min(n_c_train, n_c - n_c_test)) if n_c > 1 else n_c

                perm = rng.permutation(cls_samples)
                test_idx_list.append(perm[:n_c_test])
                train_idx_list.append(perm[n_c_test : n_c_test + n_c_train])

            test_idx = np.concatenate(test_idx_list)
            train_idx = np.concatenate(train_idx_list)
            rng.shuffle(test_idx)
            rng.shuffle(train_idx)
            yield train_idx, test_idx

    def __repr__(self) -> str:
        return (
            f"StratifiedShuffleSplit(n_splits={self.n_splits}, test_size={self.test_size}, "
            f"train_size={self.train_size}, random_state={self.random_state})"
        )


class TimeSeriesSplit(BaseCrossValidator):
    """
    Time Series cross-validator.

    Provides train/test indices to split time series data samples observed
    at fixed time intervals. In each split, test indices must be higher
    than training indices to prevent future leakage.

    Args:
        n_splits: Number of splits. Must be at least 2.
        max_train_size: Maximum size for a single training set.
        test_size: Number of samples per test set.
        gap: Number of samples to exclude between the end of train and start of test.
    """

    def __init__(
        self,
        n_splits: int = 5,
        max_train_size: Optional[int] = None,
        test_size: Optional[int] = None,
        gap: int = 0,
    ):
        if not isinstance(n_splits, (int, np.integer)) or n_splits <= 1:
            raise ValueError(f"TimeSeriesSplit requires at least 2 splits; got {n_splits}")
        self.n_splits = int(n_splits)
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = max(0, int(gap))

    def get_n_splits(
        self,
        X: Optional[Any] = None,
        y: Optional[Any] = None,
        groups: Optional[Any] = None,
    ) -> int:
        return self.n_splits

    def split(
        self,
        X: Any,
        y: Optional[Any] = None,
        groups: Optional[Any] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        gap = self.gap

        test_size = self.test_size or (n_samples // (n_splits + 1))
        if test_size <= 0:
            raise ValueError(f"test_size={test_size} must be positive.")

        if (n_splits + 1) * test_size + gap > n_samples:
            raise ValueError(
                f"Too many splits ({n_splits}) for {n_samples} samples with test_size={test_size} and gap={gap}."
            )

        indices = np.arange(n_samples)
        for i in range(n_splits):
            test_end = n_samples - (n_splits - 1 - i) * test_size
            test_start = test_end - test_size
            train_end = test_start - gap
            if train_end <= 0:
                raise ValueError("Training set size cannot be zero or negative.")

            train_start = max(0, train_end - self.max_train_size) if self.max_train_size else 0
            train_idx = indices[train_start:train_end]
            test_idx = indices[test_start:test_end]
            yield train_idx, test_idx

    def __repr__(self) -> str:
        return (
            f"TimeSeriesSplit(n_splits={self.n_splits}, max_train_size={self.max_train_size}, "
            f"test_size={self.test_size}, gap={self.gap})"
        )


def check_cv(
    cv: Union[int, BaseCrossValidator, Any] = 5,
    y: Optional[Any] = None,
    classifier: bool = False,
) -> BaseCrossValidator:
    """
    Checks and resolves a cross-validation generator.

    Args:
        cv: Integer fold count or cross-validator instance.
        y: Target values.
        classifier: Whether the estimator being evaluated is a classifier.

    Returns:
        BaseCrossValidator: Configured cross-validator instance.
    """
    if isinstance(cv, (int, np.integer)):
        if classifier and y is not None:
            return StratifiedKFold(n_splits=int(cv))
        return KFold(n_splits=int(cv))

    if isinstance(cv, BaseCrossValidator):
        return cv

    if hasattr(cv, "split"):
        return cv

    raise TypeError(f"Invalid cross-validation argument cv={cv}. Expected integer or BaseCrossValidator.")


def train_test_split(
    *arrays: Any,
    test_size: Optional[Union[float, int]] = None,
    train_size: Optional[Union[float, int]] = None,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    shuffle: bool = True,
    stratify: Optional[Any] = None,
) -> List[Any]:
    """
    Splits arrays or tensors into random train and test subsets.

    Preserves the type of each input (e.g. Nevula Tensor stays Tensor,
    NumPy ndarray stays ndarray, Python list stays list, Dataset returns Subset).

    Args:
        *arrays: Sequences of indexing-compatible objects with matching first dimension.
        test_size: Float fraction (0.0, 1.0) or int count. Default is 0.25 if train_size is None.
        train_size: Float fraction (0.0, 1.0) or int count.
        random_state: Seed for pseudo-random number generator.
        shuffle: Whether to shuffle data before splitting.
        stratify: Data array used for stratified splitting.

    Returns:
        List containing train-test split of inputs (e.g. X_train, X_test, y_train, y_test).
    """
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array or tensor must be provided to train_test_split.")

    n_samples = _num_samples(arrays[0])
    for i, arr in enumerate(arrays[1:], 1):
        arr_samples = _num_samples(arr)
        if arr_samples != n_samples:
            raise ValueError(
                f"Found input variables with inconsistent numbers of samples: "
                f"[{n_samples}, {arr_samples}]"
            )

    n_train, n_test = _validate_shuffle_split(n_samples, test_size, train_size, default_test_size=0.25)

    if stratify is not None:
        if not shuffle:
            raise ValueError("Stratified train_test_split requires shuffle=True.")
        cv = StratifiedShuffleSplit(n_splits=1, test_size=n_test, train_size=n_train, random_state=random_state)
        train_idx, test_idx = next(cv.split(arrays[0], stratify))
    elif shuffle:
        cv = ShuffleSplit(n_splits=1, test_size=n_test, train_size=n_train, random_state=random_state)
        train_idx, test_idx = next(cv.split(arrays[0]))
    else:
        indices = np.arange(n_samples)
        train_idx = indices[:n_train]
        test_idx = indices[n_train : n_train + n_test]

    result: List[Any] = []
    for arr in arrays:
        result.append(safe_indexing(arr, train_idx))
        result.append(safe_indexing(arr, test_idx))

    return result
