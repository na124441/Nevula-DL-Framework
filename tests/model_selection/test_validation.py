import os
import sys
import unittest
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import nevula as nv
from nevula.core.tensor import Tensor
from nevula.models.regression.ridge import RidgeRegression
from nevula.models.classification.logistic import LogisticRegression
from nevula.model_selection import (
    KFold,
    StratifiedKFold,
    ShuffleSplit,
    cross_val_score,
    cross_validate,
    cross_val_predict,
    learning_curve,
    validation_curve,
)


class TestValidation(unittest.TestCase):
    """Unit tests for cross-validation evaluation routines and curves."""

    def setUp(self):
        np.random.seed(42)
        # Synthetic regression dataset: y = 2*x0 - 3*x1 + 1.0 + noise
        self.n_reg = 60
        self.X_reg = np.random.uniform(-2.0, 2.0, size=(self.n_reg, 2))
        self.y_reg = 2.0 * self.X_reg[:, 0] - 3.0 * self.X_reg[:, 1] + 1.0 + np.random.normal(0, 0.05, self.n_reg)

        # Synthetic binary classification dataset: clearly separable clusters
        self.n_cls = 80
        c0 = np.random.randn(40, 2) * 0.5 + np.array([-1.5, -1.5])
        c1 = np.random.randn(40, 2) * 0.5 + np.array([1.5, 1.5])
        self.X_cls = np.vstack([c0, c1])
        self.y_cls = np.array([0.0] * 40 + [1.0] * 40)

    def test_cross_val_score_regression(self):
        """Verify cross_val_score on RidgeRegression with MSE and R2."""
        model = RidgeRegression(in_features=2, out_features=1, alpha=0.1)

        # Default scorer for regression should be r2
        scores_r2 = cross_val_score(model, self.X_reg, self.y_reg, cv=3)
        self.assertEqual(len(scores_r2), 3)
        # R2 should be high for this simple linear relationship
        self.assertTrue(np.all(scores_r2 > 0.8))

        # Explicit negative MSE metric
        scores_mse = cross_val_score(model, self.X_reg, self.y_reg, cv=3, scoring="neg_mean_squared_error")
        self.assertEqual(len(scores_mse), 3)
        self.assertTrue(np.all(scores_mse < 0.0))
        self.assertTrue(np.all(scores_mse > -1.0))

    def test_cross_val_score_classification(self):
        """Verify cross_val_score on LogisticRegression with StratifiedKFold."""
        clf = LogisticRegression(in_features=2, n_classes=2, C=1.0)
        scores = cross_val_score(
            clf, self.X_cls, self.y_cls, cv=4, scoring="accuracy", fit_params={"epochs": 50, "lr": 0.1, "verbose": False}
        )
        self.assertEqual(len(scores), 4)
        self.assertTrue(np.all(scores >= 0.85))

    def test_cross_val_score_custom_callable(self):
        """Verify cross_val_score with custom metric callable."""
        model = RidgeRegression(in_features=2, out_features=1, alpha=0.1)

        def custom_mae(y_true, y_pred):
            yt = np.array(y_true).ravel()
            yp = np.array(y_pred.to_list() if hasattr(y_pred, "to_list") else y_pred).ravel()
            return float(np.mean(np.abs(yt - yp)))

        scores = cross_val_score(model, self.X_reg, self.y_reg, cv=3, scoring=custom_mae)
        self.assertEqual(len(scores), 3)
        self.assertTrue(np.all(scores < 1.0))

    def test_cross_validate_details(self):
        """Verify cross_validate returns fit times, score times, and train scores."""
        clf = LogisticRegression(in_features=2, n_classes=2, C=1.0)
        res = cross_validate(
            clf,
            self.X_cls,
            self.y_cls,
            cv=3,
            scoring={"acc": "accuracy", "f1": "f1"},
            return_train_score=True,
            return_estimator=True,
            fit_params={"epochs": 30, "lr": 0.1, "verbose": False},
        )

        self.assertIn("fit_time", res)
        self.assertIn("score_time", res)
        self.assertIn("test_acc", res)
        self.assertIn("test_f1", res)
        self.assertIn("train_acc", res)
        self.assertIn("train_f1", res)
        self.assertIn("estimator", res)

        self.assertEqual(len(res["test_acc"]), 3)
        self.assertEqual(len(res["estimator"]), 3)
        self.assertTrue(res["estimator"][0].is_fitted)

    def test_cross_val_predict(self):
        """Verify out-of-fold cross_val_predict generates predictions for each sample."""
        clf = LogisticRegression(in_features=2, n_classes=2, C=1.0)
        preds = cross_val_predict(
            clf,
            self.X_cls,
            self.y_cls,
            cv=4,
            method="predict",
            fit_params={"epochs": 40, "lr": 0.1, "verbose": False},
        )
        self.assertEqual(len(preds), self.n_cls)
        # Check overall out-of-fold accuracy
        oof_acc = np.mean(preds.ravel() == self.y_cls.ravel())
        self.assertGreater(oof_acc, 0.8)

    def test_cross_val_predict_proba(self):
        """Verify out-of-fold predict_proba returns probability distributions."""
        clf = LogisticRegression(in_features=2, n_classes=2, C=1.0)
        probs = cross_val_predict(
            clf,
            self.X_cls,
            self.y_cls,
            cv=3,
            method="predict_proba",
            fit_params={"epochs": 40, "lr": 0.1, "verbose": False},
        )
        self.assertEqual(probs.shape, (self.n_cls, 2))
        probs_np = probs.numpy() if isinstance(probs, Tensor) else probs
        # Row sums should equal 1.0
        np.testing.assert_allclose(np.sum(probs_np, axis=1), 1.0, atol=1e-5)

    def test_cross_val_predict_invalid_cv(self):
        """Verify cross_val_predict raises error on non-partitioning splitters (e.g. ShuffleSplit)."""
        model = RidgeRegression(in_features=2, out_features=1)
        ss = ShuffleSplit(n_splits=5, test_size=0.2)
        with self.assertRaises(ValueError):
            cross_val_predict(model, self.X_reg, self.y_reg, cv=ss)

    def test_learning_curve(self):
        """Verify learning_curve calculates scores across sample size increments."""
        model = RidgeRegression(in_features=2, out_features=1, alpha=0.1)
        train_sizes, train_scores, test_scores = learning_curve(
            model,
            self.X_reg,
            self.y_reg,
            train_sizes=[0.3, 0.6, 1.0],
            cv=3,
            scoring="r2",
        )
        self.assertEqual(len(train_sizes), 3)
        self.assertEqual(train_scores.shape, (3, 3))
        self.assertEqual(test_scores.shape, (3, 3))

    def test_validation_curve(self):
        """Verify validation_curve computes scores across hyperparameter values."""
        model = RidgeRegression(in_features=2, out_features=1)
        train_scores, test_scores = validation_curve(
            model,
            self.X_reg,
            self.y_reg,
            param_name="alpha",
            param_range=[0.01, 0.1, 1.0],
            cv=3,
            scoring="r2",
        )
        self.assertEqual(train_scores.shape, (3, 3))
        self.assertEqual(test_scores.shape, (3, 3))


if __name__ == "__main__":
    unittest.main()
