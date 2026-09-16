import os
import sys

# Ensure repository root is on sys.path when test is executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import unittest
import numpy as np

from nevula.core.tensor import Tensor
from nevula.models.base import BaseModel
from nevula.models.registry import get_model, list_models
from nevula.models.regression.linear import LinearRegression
from nevula.models.regression.svr import SVR
from nevula.metrics.regression import mean_squared_error, r2_score


class TestSVR(unittest.TestCase):
    """
    Independent test suite for Nevula Support Vector Regression.
    Covers registry lookup, forward arithmetic, epsilon-insensitive tube loss,
    support vector mask identification, outlier resilience, parameter validation, and serialization.
    """

    def test_registry_registration(self):
        """Verify SVR is registered under canonical name and alias."""
        model_cls_svr = get_model("svr")
        self.assertIs(model_cls_svr, SVR)

        model_cls_alias = get_model("support_vector_regression")
        self.assertIs(model_cls_alias, SVR)

        models_list = list_models(category="regression")
        self.assertIn("SVR", models_list)

    def test_forward_computation(self):
        """Verify forward pass evaluates y = X @ W + b precisely."""
        model = SVR(in_features=2, out_features=1, epsilon=0.1, C=1.0, fit_intercept=True, mode="tensor")
        model.weights.data[0] = 1.5
        model.weights.data[1] = -2.5
        model.bias.data[0] = 0.5

        X = Tensor([[2.0, 1.0], [0.0, 2.0]])
        # sample 0: 2*1.5 + 1*(-2.5) + 0.5 = 3.0 - 2.5 + 0.5 = 1.0
        # sample 1: 0*1.5 + 2*(-2.5) + 0.5 = 0.0 - 5.0 + 0.5 = -4.5
        preds = model(X)
        self.assertEqual(preds.shape, (2, 1))
        self.assertAlmostEqual(float(preds.data[0]), 1.0, places=5)
        self.assertAlmostEqual(float(preds.data[1]), -4.5, places=5)

    def test_epsilon_insensitive_tube_zero_loss(self):
        """Verify that points within epsilon distance incur zero tube loss."""
        epsilon = 0.5
        model = SVR(in_features=1, out_features=1, epsilon=epsilon, C=1.0, mode="tensor")
        model.weights.data[0] = 2.0
        model.bias.data[0] = 0.0

        # Point with residual within tube: x = 1.0 => y_hat = 2.0, true y = 2.3 (|diff| = 0.3 <= 0.5)
        X = Tensor([[1.0]])
        y = Tensor([[2.3]])

        pred = model(X)
        diff = (pred - y).abs()
        tube_loss = (diff - epsilon).relu()

        # Tube loss must be strictly 0.0
        self.assertAlmostEqual(float(tube_loss.data[0]), 0.0, places=5)

    def test_support_vectors_mask_and_ratio(self):
        """Verify support vectors mask correctly identifies points outside epsilon tube."""
        epsilon = 0.2
        model = SVR(in_features=1, out_features=1, epsilon=epsilon, C=1.0, mode="tensor")
        model.weights.data[0] = 1.0
        model.bias.data[0] = 0.0

        # Point 0: x=1.0 => y_hat=1.0, y=1.05 (|diff|=0.05 < 0.2) -> NOT a support vector
        # Point 1: x=2.0 => y_hat=2.0, y=2.50 (|diff|=0.50 >= 0.2) -> SUPPORT VECTOR
        X = np.array([[1.0], [2.0]])
        y = np.array([[1.05], [2.50]])

        mask = model.support_vectors_mask(X, y)
        self.assertEqual(len(mask), 2)
        self.assertFalse(mask[0])
        self.assertTrue(mask[1])

        ratio = model.support_vectors_ratio(X, y)
        self.assertAlmostEqual(ratio, 0.5, places=4)

    def test_outlier_resilience_comparison(self):
        """
        Verify that SVR achieves closer slope recovery than OLS on contaminated data.
        """
        np.random.seed(42)
        X_clean = np.random.uniform(-2.0, 2.0, size=(60, 1))
        y_clean = 2.0 * X_clean + 1.0

        # Contaminate with 5 severe outlier spikes
        X_out = np.ones((5, 1)) * 2.0
        y_out = np.ones((5, 1)) * -40.0

        X = np.vstack([X_clean, X_out])
        y = np.vstack([y_clean, y_out])

        ols = LinearRegression(in_features=1, out_features=1)
        ols.fit(X, y, epochs=150, lr=0.03, verbose=False)
        ols_slope = float(ols.weight.data[0])

        svr = SVR(in_features=1, out_features=1, epsilon=0.2, C=1.0)
        svr.fit(X, y, epochs=200, lr=0.03, verbose=False)
        svr_slope = float(svr.weight.data[0])

        # SVR slope error must be lower than OLS slope error
        ols_err = abs(ols_slope - 2.0)
        svr_err = abs(svr_slope - 2.0)
        self.assertLess(svr_err, ols_err)

    def test_hyperparameter_validation(self):
        """Verify negative epsilon or non-positive C raise ValueError."""
        with self.assertRaises(ValueError):
            SVR(in_features=1, epsilon=-0.1)
        with self.assertRaises(ValueError):
            SVR(in_features=1, C=0.0)
        with self.assertRaises(ValueError):
            SVR(in_features=1, C=-1.0)

    def test_config_and_serialization(self):
        """Verify get_config and state_dict save/load."""
        import tempfile
        model = SVR(in_features=2, out_features=1, epsilon=0.25, C=2.5, fit_intercept=True)
        config = model.get_config()
        self.assertEqual(config["epsilon"], 0.25)
        self.assertEqual(config["C"], 2.5)

        model.weight.data[0] = 7.77
        model.weight.data[1] = -8.88
        model.bias_param.data[0] = 4.44

        with tempfile.TemporaryDirectory() as tmpdir:
            chk_path = os.path.join(tmpdir, "svr.chk")
            model.save(chk_path)

            loaded = SVR(in_features=2, out_features=1, epsilon=0.25, C=2.5)
            loaded.load(chk_path)

            self.assertAlmostEqual(float(loaded.weight.data[0]), 7.77, places=4)
            self.assertAlmostEqual(float(loaded.weight.data[1]), -8.88, places=4)
            self.assertAlmostEqual(float(loaded.bias_param.data[0]), 4.44, places=4)


if __name__ == "__main__":
    unittest.main()
