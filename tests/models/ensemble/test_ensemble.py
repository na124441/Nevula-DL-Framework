import os
import sys

# Ensure repository root is on sys.path when test is executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import unittest
import numpy as np

from nevula.core.tensor import Tensor
from nevula.models.base import BaseModel
from nevula.models.registry import get_model, list_models
from nevula.models.ensemble.voting import VotingRegressor
from nevula.models.ensemble.random_forest import RandomForestRegressor
from nevula.models.ensemble.gradient_boosting import GradientBoostingRegressor
from nevula.models.trees.decision_tree import DecisionTreeRegressor
from nevula.models.regression.linear import LinearRegression


class TestEnsembleRegistry(unittest.TestCase):
    """Verify registry lookups for all ensemble models and aliases."""

    def test_registry_registration(self):
        self.assertIs(get_model("voting_regressor"), VotingRegressor)
        self.assertIs(get_model("voting"), VotingRegressor)
        self.assertIs(get_model("random_forest_regressor"), RandomForestRegressor)
        self.assertIs(get_model("random_forest"), RandomForestRegressor)
        self.assertIs(get_model("gradient_boosting_regressor"), GradientBoostingRegressor)
        self.assertIs(get_model("gradient_boosting"), GradientBoostingRegressor)

        ensemble_models = list_models(category="ensemble")
        self.assertIn("VotingRegressor", ensemble_models)
        self.assertIn("RandomForestRegressor", ensemble_models)
        self.assertIn("GradientBoostingRegressor", ensemble_models)


class TestVotingRegressor(unittest.TestCase):
    """Independent unit tests for VotingRegressor."""

    def test_voting_initialization_and_weights(self):
        est1 = LinearRegression(in_features=2, out_features=1)
        est2 = LinearRegression(in_features=2, out_features=1)

        # Uniform weights by default
        vr_uniform = VotingRegressor(estimators=[("e1", est1), ("e2", est2)])
        self.assertEqual(len(vr_uniform), 2)
        self.assertAlmostEqual(vr_uniform.weights_[0], 0.5)
        self.assertAlmostEqual(vr_uniform.weights_[1], 0.5)

        # Normalized custom weights
        vr_weighted = VotingRegressor(estimators=[est1, est2], weights=[1.0, 3.0])
        self.assertAlmostEqual(vr_weighted.weights_[0], 0.25)
        self.assertAlmostEqual(vr_weighted.weights_[1], 0.75)

        # Negative weights rejected
        with self.assertRaises(ValueError):
            VotingRegressor(estimators=[est1, est2], weights=[-1.0, 2.0])

    def test_voting_fit_and_predict(self):
        np.random.seed(42)
        X = np.random.uniform(-2, 2, size=(50, 2))
        y = 2.0 * X[:, 0] + 1.0 * X[:, 1] + 0.5

        est1 = LinearRegression(in_features=2, out_features=1)
        est2 = DecisionTreeRegressor(max_depth=4)

        vr = VotingRegressor(estimators=[("lr", est1), ("dt", est2)], weights=[0.5, 0.5])
        vr.fit(X, y.reshape((-1, 1)), epochs=50, lr=0.05, verbose=False)

        preds = vr.predict(X)
        self.assertIsInstance(preds, Tensor)
        self.assertEqual(preds.shape, (50, 1))

        r2 = vr.evaluate(X, y.reshape((-1, 1)), metric="r2")
        self.assertGreater(r2, 0.85)

    def test_voting_serialization(self):
        est1 = LinearRegression(in_features=2)
        est2 = DecisionTreeRegressor(max_depth=3)
        vr = VotingRegressor(estimators=[("m1", est1), ("m2", est2)], weights=[0.4, 0.6])
        config = vr.get_config()
        self.assertIn("estimators", config)
        self.assertEqual(config["weights"], [0.4, 0.6])


class TestRandomForestRegressor(unittest.TestCase):
    """Independent unit tests for RandomForestRegressor."""

    def test_random_forest_fit_and_oob(self):
        np.random.seed(42)
        X = np.random.uniform(-3, 3, size=(80, 4))
        y = np.sin(X[:, 0]) + 0.5 * (X[:, 1] ** 2)

        rf = RandomForestRegressor(
            n_estimators=15,
            max_depth=5,
            max_features="sqrt",
            bootstrap=True,
            oob_score=True,
            random_state=42,
        )
        rf.fit(X, y)

        self.assertEqual(len(rf), 15)
        self.assertIsNotNone(rf.oob_score_)
        self.assertGreater(rf.oob_score_, 0.5)

        preds = rf.predict(X)
        self.assertEqual(preds.shape, (80, 1))

        r2 = rf.evaluate(X, y, metric="r2")
        self.assertGreater(r2, 0.85)

    def test_random_forest_feature_importances(self):
        np.random.seed(42)
        X = np.random.uniform(-2, 2, size=(60, 3))
        # Feature 0 is dominant, Feature 2 is pure noise
        y = 3.0 * X[:, 0] + np.random.normal(0, 0.05, size=(60,))

        rf = RandomForestRegressor(n_estimators=10, max_depth=4, random_state=42)
        rf.fit(X, y)

        importances = rf.feature_importances()
        self.assertEqual(len(importances), 3)
        self.assertAlmostEqual(float(np.sum(importances)), 1.0, places=4)
        # Feature 0 should have highest importance
        self.assertGreater(importances[0], importances[2])

    def test_random_forest_serialization(self):
        rf = RandomForestRegressor(n_estimators=20, max_depth=6, max_features="log2")
        config = rf.get_config()
        self.assertEqual(config["n_estimators"], 20)
        self.assertEqual(config["max_depth"], 6)
        self.assertEqual(config["max_features"], "log2")


class TestGradientBoostingRegressor(unittest.TestCase):
    """Independent unit tests for GradientBoostingRegressor."""

    def test_gradient_boosting_convergence(self):
        np.random.seed(42)
        X = np.linspace(-3, 3, 60).reshape((-1, 1))
        y = np.sin(X[:, 0])

        gbdt = GradientBoostingRegressor(
            n_estimators=25,
            learning_rate=0.15,
            max_depth=2,
            subsample=1.0,
            random_state=42,
        )
        gbdt.fit(X, y)

        self.assertEqual(len(gbdt), 25)
        # Verify training loss decreased over boosting stages
        self.assertLess(gbdt.train_score_[-1], gbdt.train_score_[0])

        r2 = gbdt.evaluate(X, y, metric="r2")
        self.assertGreater(r2, 0.90)

    def test_gradient_boosting_subsampling_and_importances(self):
        np.random.seed(42)
        X = np.random.uniform(-2, 2, size=(70, 4))
        # Feature 1 is dominant
        y = 2.5 * (X[:, 1] ** 2)

        gbdt = GradientBoostingRegressor(
            n_estimators=20,
            learning_rate=0.1,
            max_depth=3,
            subsample=0.8,
            random_state=42,
        )
        gbdt.fit(X, y)

        importances = gbdt.feature_importances()
        self.assertEqual(len(importances), 4)
        self.assertAlmostEqual(float(np.sum(importances)), 1.0, places=4)
        # Feature 1 should dominate
        self.assertGreater(importances[1], importances[0])

    def test_gradient_boosting_validation(self):
        with self.assertRaises(ValueError):
            GradientBoostingRegressor(n_estimators=0)
        with self.assertRaises(ValueError):
            GradientBoostingRegressor(learning_rate=-0.1)
        with self.assertRaises(ValueError):
            GradientBoostingRegressor(subsample=1.5)


if __name__ == "__main__":
    unittest.main()
