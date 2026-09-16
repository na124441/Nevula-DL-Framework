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
from nevula.models.regression.ridge import RidgeRegression
from nevula.metrics.regression import mean_squared_error, r2_score


class TestRidgeRegression(unittest.TestCase):
    """
    Independent test suite for Nevula Ridge Regression.
    Covers registry lookup, forward correctness, L2 gradient computation,
    parameter shrinkage, alpha effects, evaluation metrics, and state serialization.
    """

    def test_registry_registration(self):
        """Verify RidgeRegression is registered under canonical name and alias."""
        model_cls_ridge = get_model("ridge")
        self.assertIs(model_cls_ridge, RidgeRegression)

        model_cls_alias = get_model("ridge_regression")
        self.assertIs(model_cls_alias, RidgeRegression)

        models_list = list_models(category="regression")
        self.assertIn("RidgeRegression", models_list)

    def test_forward_computation(self):
        """Verify forward pass evaluates y = X @ W + b precisely."""
        model = RidgeRegression(in_features=2, out_features=1, alpha=1.0, fit_intercept=True, mode="tensor")
        model.weights.data[0] = 3.0
        model.weights.data[1] = -2.0
        model.bias.data[0] = 1.0

        X = Tensor([[1.0, 1.0], [2.0, 0.0]])
        # sample 0: 1*3 + 1*(-2) + 1 = 2.0
        # sample 1: 2*3 + 0*(-2) + 1 = 7.0
        preds = model(X)
        self.assertEqual(preds.shape, (2, 1))
        self.assertAlmostEqual(float(preds.data[0]), 2.0, places=5)
        self.assertAlmostEqual(float(preds.data[1]), 7.0, places=5)

    def test_gradient_computation_with_l2_penalty(self):
        """Verify gradients computed by Autograd include MSE gradient plus 2 * alpha * W."""
        alpha = 2.5
        model = RidgeRegression(in_features=1, out_features=1, alpha=alpha, fit_intercept=True, mode="tensor")
        model.weights.data[0] = 2.0
        model.bias.data[0] = 0.0

        # One sample: x = 1.0, y = 4.0
        # y_hat = 1.0 * 2.0 = 2.0
        # mse = (2.0 - 4.0)^2 = 4.0
        # l2_penalty = alpha * (2.0)^2 = 2.5 * 4.0 = 10.0
        # total_loss = 14.0
        # d_mse / dW = 2 * (2.0 - 4.0) * 1.0 = -4.0
        # d_l2 / dW = 2 * alpha * W = 2 * 2.5 * 2.0 = 10.0
        # d_total / dW = -4.0 + 10.0 = +6.0
        # d_total / db = 2 * (2.0 - 4.0) = -4.0 (bias not penalized)
        X = Tensor([[1.0]])
        y = Tensor([[4.0]])

        pred = model(X)
        mse = ((pred - y) ** 2).mean()
        l2 = (model.weights ** 2).sum() * alpha
        loss = mse + l2
        loss.backward()

        self.assertIsNotNone(model.weights.grad)
        self.assertIsNotNone(model.bias.grad)
        self.assertAlmostEqual(float(model.weights.grad.data[0]), 6.0, places=4)
        self.assertAlmostEqual(float(model.bias.grad.data[0]), -4.0, places=4)

    def test_weight_shrinkage_property(self):
        """
        Verify the core property of Ridge Regression:
            ||W_ridge||_2 < ||W_ols||_2
        for identical collinear datasets when alpha > 0.
        """
        np.random.seed(42)
        n = 150
        x1 = np.random.uniform(-3.0, 3.0, size=(n, 1))
        x2 = x1 + np.random.normal(0.0, 0.05, size=(n, 1))  # strong collinearity
        X = np.hstack([x1, x2])
        y = 3.0 * x1 + 2.0 * x2 + 1.0

        # Fit OLS (alpha = 0 equivalent)
        ols = LinearRegression(in_features=2, out_features=1)
        ols.fit(X, y, epochs=150, lr=0.03, optimizer="sgd", verbose=False)
        ols_norm = float(np.linalg.norm([float(w) for w in ols.weight.data]))

        # Fit Ridge (alpha = 5.0)
        ridge = RidgeRegression(in_features=2, out_features=1, alpha=5.0)
        ridge.fit(X, y, epochs=150, lr=0.03, optimizer="sgd", verbose=False)
        ridge_norm = float(np.linalg.norm([float(w) for w in ridge.weight.data]))

        # Ridge weights must be strictly smaller in L2 norm
        self.assertLess(ridge_norm, ols_norm)
        self.assertTrue(ridge.is_fitted)

    def test_higher_alpha_produces_greater_shrinkage(self):
        """Verify that increasing alpha monotonically reduces weight norm."""
        np.random.seed(123)
        n = 100
        X = np.random.randn(n, 3)
        y = X @ np.array([[2.0], [-1.0], [1.5]]) + 0.5

        ridge_low = RidgeRegression(in_features=3, out_features=1, alpha=0.1)
        ridge_low.fit(X, y, epochs=120, lr=0.02, verbose=False)
        norm_low = float(np.linalg.norm([float(w) for w in ridge_low.weight.data]))

        ridge_high = RidgeRegression(in_features=3, out_features=1, alpha=10.0)
        ridge_high.fit(X, y, epochs=120, lr=0.02, verbose=False)
        norm_high = float(np.linalg.norm([float(w) for w in ridge_high.weight.data]))

        self.assertLess(norm_high, norm_low)

    def test_alpha_zero_validation(self):
        """Verify alpha must be non-negative."""
        with self.assertRaises(ValueError):
            RidgeRegression(in_features=2, alpha=-1.0)

    def test_prediction_and_evaluation(self):
        """Verify predictions have expected shape and evaluate metrics work."""
        X = Tensor([[1.0], [2.0], [3.0]])
        y = Tensor([[2.0], [4.0], [6.0]])

        model = RidgeRegression(in_features=1, out_features=1, alpha=0.5)
        model.fit(X, y, epochs=80, lr=0.05, verbose=False)

        preds = model.predict(X)
        self.assertEqual(preds.shape, (3, 1))
        self.assertFalse(preds.requires_grad)

        mse = model.evaluate(X, y, metric="mse")
        self.assertLess(mse, 0.5)

    def test_config_and_serialization(self):
        """Verify get_config and state_dict save/load."""
        import tempfile
        model = RidgeRegression(in_features=2, out_features=1, alpha=3.5, fit_intercept=True)
        config = model.get_config()
        self.assertEqual(config["alpha"], 3.5)
        self.assertEqual(config["in_features"], 2)

        model.weight.data[0] = 1.11
        model.weight.data[1] = 2.22
        model.bias_param.data[0] = 3.33

        with tempfile.TemporaryDirectory() as tmpdir:
            chk_path = os.path.join(tmpdir, "ridge.chk")
            model.save(chk_path)

            loaded_model = RidgeRegression(in_features=2, out_features=1, alpha=3.5)
            loaded_model.load(chk_path)

            self.assertAlmostEqual(float(loaded_model.weight.data[0]), 1.11, places=4)
            self.assertAlmostEqual(float(loaded_model.weight.data[1]), 2.22, places=4)
            self.assertAlmostEqual(float(loaded_model.bias_param.data[0]), 3.33, places=4)


if __name__ == "__main__":
    unittest.main()
