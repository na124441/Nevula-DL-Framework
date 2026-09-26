import os
import sys
import unittest
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import nevula as nv
from nevula.core.tensor import Tensor
from nevula.data.dataset import TensorDataset, Subset
from nevula.model_selection import (
    KFold,
    StratifiedKFold,
    LeaveOneOut,
    LeavePOut,
    ShuffleSplit,
    StratifiedShuffleSplit,
    TimeSeriesSplit,
    train_test_split,
    check_cv,
)


class TestDataSplitting(unittest.TestCase):
    """Unit tests for cross-validation generators and data splitting utilities."""

    def test_kfold_basic(self):
        """Verify KFold fold counts, disjoint splits, and complete sample coverage."""
        X = np.arange(20).reshape((10, 2))
        kf = KFold(n_splits=5, shuffle=False)

        self.assertEqual(kf.get_n_splits(X), 5)
        seen_test_indices = []

        for train_idx, test_idx in kf.split(X):
            # Train and test should be disjoint
            self.assertEqual(len(np.intersect1d(train_idx, test_idx)), 0)
            # Train + test should cover all samples
            self.assertEqual(len(train_idx) + len(test_idx), 10)
            self.assertEqual(len(test_idx), 2)
            seen_test_indices.extend(test_idx.tolist())

        # All samples must be evaluated once
        self.assertEqual(sorted(seen_test_indices), list(range(10)))

    def test_kfold_uneven_splits(self):
        """Verify KFold properly distributes remainder when N is not divisible by k."""
        X = np.arange(11)
        kf = KFold(n_splits=3, shuffle=False)
        splits = list(kf.split(X))
        fold_lens = [len(test_idx) for _, test_idx in splits]
        # 11 samples into 3 folds: sizes should be 4, 4, 3
        self.assertEqual(fold_lens, [4, 4, 3])

    def test_kfold_shuffle_reproducibility(self):
        """Verify shuffle with random_state produces deterministic identical splits."""
        X = np.arange(30)
        kf1 = KFold(n_splits=3, shuffle=True, random_state=42)
        kf2 = KFold(n_splits=3, shuffle=True, random_state=42)

        splits1 = list(kf1.split(X))
        splits2 = list(kf2.split(X))

        for (tr1, te1), (tr2, te2) in zip(splits1, splits2):
            np.testing.assert_array_equal(tr1, tr2)
            np.testing.assert_array_equal(te1, te2)

    def test_kfold_invalid_args(self):
        """Verify exceptions on invalid n_splits."""
        with self.assertRaises(ValueError):
            KFold(n_splits=1)
        with self.assertRaises(ValueError):
            list(KFold(n_splits=10).split(np.arange(5)))

    def test_stratified_kfold_class_balance(self):
        """Verify StratifiedKFold maintains class proportions in each fold."""
        # Dataset: 20 samples of class 0, 10 samples of class 1 (2:1 ratio)
        y = np.array([0] * 20 + [1] * 10)
        X = np.zeros((30, 2))

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        self.assertEqual(skf.get_n_splits(X, y), 5)

        for train_idx, test_idx in skf.split(X, y):
            test_y = y[test_idx]
            # Each test fold should have 4 samples of class 0 and 2 of class 1
            n_c0 = np.sum(test_y == 0)
            n_c1 = np.sum(test_y == 1)
            self.assertEqual(n_c0, 4)
            self.assertEqual(n_c1, 2)
            self.assertEqual(len(np.intersect1d(train_idx, test_idx)), 0)

    def test_stratified_kfold_requires_y(self):
        """Verify StratifiedKFold raises ValueError if y is None."""
        skf = StratifiedKFold(n_splits=3)
        with self.assertRaises(ValueError):
            list(skf.split(np.zeros((10, 2)), y=None))

    def test_leave_one_out(self):
        """Verify LeaveOneOut produces N folds of test size 1."""
        X = np.arange(8)
        loo = LeaveOneOut()
        self.assertEqual(loo.get_n_splits(X), 8)

        splits = list(loo.split(X))
        self.assertEqual(len(splits), 8)
        for i, (train_idx, test_idx) in enumerate(splits):
            self.assertEqual(len(test_idx), 1)
            self.assertEqual(test_idx[0], i)
            self.assertEqual(len(train_idx), 7)
            self.assertNotIn(i, train_idx)

    def test_leave_p_out(self):
        """Verify LeavePOut produces C(N, p) combinations of test size p."""
        X = np.arange(5)
        lpo = LeavePOut(p=2)
        # C(5, 2) = 10 splits
        self.assertEqual(lpo.get_n_splits(X), 10)
        splits = list(lpo.split(X))
        self.assertEqual(len(splits), 10)
        for train_idx, test_idx in splits:
            self.assertEqual(len(test_idx), 2)
            self.assertEqual(len(train_idx), 3)

    def test_shuffle_split(self):
        """Verify ShuffleSplit produces specified train and test sizes."""
        X = np.arange(100)
        ss = ShuffleSplit(n_splits=4, test_size=0.2, random_state=123)
        self.assertEqual(ss.get_n_splits(X), 4)

        for train_idx, test_idx in ss.split(X):
            self.assertEqual(len(test_idx), 20)
            self.assertEqual(len(train_idx), 80)
            self.assertEqual(len(np.intersect1d(train_idx, test_idx)), 0)

    def test_stratified_shuffle_split(self):
        """Verify StratifiedShuffleSplit maintains class distribution."""
        y = np.array([0] * 60 + [1] * 40)
        X = np.zeros((100, 2))
        sss = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=42)

        for train_idx, test_idx in sss.split(X, y):
            test_y = y[test_idx]
            self.assertEqual(len(test_idx), 20)
            self.assertEqual(np.sum(test_y == 0), 12)
            self.assertEqual(np.sum(test_y == 1), 8)

    def test_time_series_split(self):
        """Verify TimeSeriesSplit enforces temporal ordering (train < test)."""
        X = np.arange(20)
        tss = TimeSeriesSplit(n_splits=3)
        self.assertEqual(tss.get_n_splits(X), 3)

        for train_idx, test_idx in tss.split(X):
            self.assertGreater(len(train_idx), 0)
            self.assertGreater(len(test_idx), 0)
            # Max index in train must be strictly less than min index in test
            self.assertLess(np.max(train_idx), np.min(test_idx))

    def test_time_series_split_with_gap(self):
        """Verify TimeSeriesSplit gap parameter excludes intermediate samples."""
        X = np.arange(30)
        tss = TimeSeriesSplit(n_splits=3, test_size=5, gap=2)
        for train_idx, test_idx in tss.split(X):
            # Difference between first test index and last train index must be gap + 1
            self.assertEqual(np.min(test_idx) - np.max(train_idx), 3)

    def test_train_test_split_tensor(self):
        """Verify train_test_split on Nevula Tensors preserves Tensor type and shapes."""
        X_np = np.arange(40).reshape((20, 2)).astype(np.float64)
        y_np = np.arange(20).astype(np.float64)
        X = Tensor(X_np)
        y = Tensor(y_np)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, shuffle=True
        )

        self.assertIsInstance(X_train, Tensor)
        self.assertIsInstance(X_test, Tensor)
        self.assertIsInstance(y_train, Tensor)
        self.assertIsInstance(y_test, Tensor)

        self.assertEqual(X_train.shape, (15, 2))
        self.assertEqual(X_test.shape, (5, 2))
        self.assertEqual(y_train.shape, (15,))
        self.assertEqual(y_test.shape, (5,))

    def test_train_test_split_stratify(self):
        """Verify train_test_split with stratify preserves class proportions."""
        y = np.array([0] * 30 + [1] * 10)
        X = np.zeros((40, 3))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=7, stratify=y
        )

        self.assertEqual(len(y_test), 10)
        self.assertIn(int(np.sum(y_test == 0)), [7, 8])
        self.assertIn(int(np.sum(y_test == 1)), [2, 3])

    def test_train_test_split_dataset(self):
        """Verify train_test_split on Dataset returns Subsets."""
        t1 = Tensor(np.arange(10).reshape((10, 1)))
        ds = TensorDataset(t1)

        ds_train, ds_test = train_test_split(ds, test_size=0.3, shuffle=False)
        self.assertIsInstance(ds_train, Subset)
        self.assertIsInstance(ds_test, Subset)
        self.assertEqual(len(ds_train), 7)
        self.assertEqual(len(ds_test), 3)

    def test_check_cv_resolution(self):
        """Verify check_cv selects appropriate cross-validator."""
        cv_reg = check_cv(5, classifier=False)
        self.assertIsInstance(cv_reg, KFold)
        self.assertEqual(cv_reg.n_splits, 5)

        cv_cls = check_cv(3, y=[0, 1, 0, 1], classifier=True)
        self.assertIsInstance(cv_cls, StratifiedKFold)
        self.assertEqual(cv_cls.n_splits, 3)


if __name__ == "__main__":
    unittest.main()
