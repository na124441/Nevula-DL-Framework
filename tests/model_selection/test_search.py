import os
import sys
import unittest
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import nevula as nv
from nevula.models.regression.ridge import RidgeRegression
from nevula.models.classification.logistic import LogisticRegression
from nevula.model_selection import (
    ParameterGrid,
    ParameterSampler,
    GridSearchCV,
    RandomizedSearchCV,
)


class TestSearch(unittest.TestCase):
    """Unit tests for ParameterGrid, ParameterSampler, GridSearchCV, and RandomizedSearchCV."""

    def setUp(self):
        np.random.seed(42)
        # Regression synthetic dataset
        self.n_samples = 50
        self.X_reg = np.random.uniform(-1.0, 1.0, size=(self.n_samples, 2))
        self.y_reg = 3.0 * self.X_reg[:, 0] - 2.0 * self.X_reg[:, 1] + 0.5 + np.random.normal(0, 0.05, self.n_samples)

        # Classification synthetic dataset: clearly separated clusters
        c0 = np.random.randn(self.n_samples // 2, 2) * 0.5 + np.array([-1.5, -1.5])
        c1 = np.random.randn(self.n_samples // 2, 2) * 0.5 + np.array([1.5, 1.5])
        self.X_cls = np.vstack([c0, c1])
        self.y_cls = np.array([0.0] * (self.n_samples // 2) + [1.0] * (self.n_samples // 2))

    def test_parameter_grid(self):
        """Verify ParameterGrid generates exact Cartesian product."""
        param_grid = {"alpha": [0.01, 0.1], "fit_intercept": [True, False]}
        grid = ParameterGrid(param_grid)
        self.assertEqual(len(grid), 4)

        items = list(grid)
        expected = [
            {"alpha": 0.01, "fit_intercept": True},
            {"alpha": 0.01, "fit_intercept": False},
            {"alpha": 0.1, "fit_intercept": True},
            {"alpha": 0.1, "fit_intercept": False},
        ]
        for exp in expected:
            self.assertIn(exp, items)

    def test_parameter_sampler(self):
        """Verify ParameterSampler produces requested number of candidate parameter dictionaries."""
        distributions = {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
        sampler = ParameterSampler(distributions, n_iter=3, random_state=42)
        self.assertEqual(len(sampler), 3)

        candidates = list(sampler)
        self.assertEqual(len(candidates), 3)
        for c in candidates:
            self.assertIn(c["alpha"], distributions["alpha"])

    def test_grid_search_cv_regression(self):
        """Verify GridSearchCV tunes RidgeRegression hyperparameters correctly."""
        model = RidgeRegression(in_features=2, out_features=1)
        param_grid = {"alpha": [0.001, 0.1, 10.0]}

        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=3,
            scoring="r2",
            refit=True,
        )
        search.fit(self.X_reg, self.y_reg, epochs=150, lr=0.03)

        self.assertIsNotNone(search.best_params_)
        self.assertIn("alpha", search.best_params_)
        self.assertIsNotNone(search.best_score_)
        self.assertGreater(search.best_score_, 0.65)
        self.assertIsNotNone(search.best_estimator_)
        self.assertTrue(search.best_estimator_.is_fitted)

        # Check cv_results_ keys
        self.assertIn("mean_test_score", search.cv_results_)
        self.assertIn("rank_test_score", search.cv_results_)
        self.assertEqual(len(search.cv_results_["params"]), 3)

        # Predict delegates to best_estimator_
        preds = search.predict(self.X_reg)
        self.assertEqual(len(preds), self.n_samples)

    def test_grid_search_cv_classification(self):
        """Verify GridSearchCV tunes LogisticRegression and delegates predict_proba and score."""
        clf = LogisticRegression(in_features=2, n_classes=2)
        param_grid = {"C": [0.1, 1.0, 10.0]}

        search = GridSearchCV(
            estimator=clf,
            param_grid=param_grid,
            cv=3,
            scoring="accuracy",
            refit=True,
        )
        search.fit(self.X_cls, self.y_cls, epochs=30, lr=0.1, verbose=False)

        self.assertIsNotNone(search.best_params_)
        self.assertIn("C", search.best_params_)
        self.assertGreater(search.best_score_, 0.75)

        # Verify predictions and probabilities
        preds = search.predict(self.X_cls)
        probs = search.predict_proba(self.X_cls)
        self.assertEqual(len(preds), self.n_samples)
        self.assertEqual(probs.shape, (self.n_samples, 2))

        # Check score method
        acc_score = search.score(self.X_cls, self.y_cls)
        self.assertGreater(acc_score, 0.75)

    def test_randomized_search_cv(self):
        """Verify RandomizedSearchCV samples candidate space and refits best model."""
        model = RidgeRegression(in_features=2, out_features=1)
        param_dists = {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0]}

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dists,
            n_iter=3,
            cv=3,
            scoring="r2",
            random_state=42,
            refit=True,
        )
        search.fit(self.X_reg, self.y_reg)

        self.assertIsNotNone(search.best_params_)
        self.assertIn("alpha", search.best_params_)
        self.assertGreater(search.best_score_, 0.6)
        self.assertTrue(search.best_estimator_.is_fitted)


if __name__ == "__main__":
    unittest.main()
