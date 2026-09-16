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
from nevula.models.regression.lasso import LassoRegression
from nevula.metrics.regression import mean_squared_error, r2_score


class TestLassoRegression(unittest.TestCase):
    """
    Independent test suite for Nevula Lasso Regression.
    Covers registry lookup, differentiable Tensor.abs(), forward arithmetic,
    L1 subgradient computation, feature sparsity, alpha sensitivity, and serialization.
    """

    def test_registry_registration(self):
        """Verify LassoRegression is registered under canonical name and alias."""
        model_cls_lasso = get_model("lasso")
        self.assertIs(model_cls_lasso, LassoRegression)

        model_cls_alias = get_model("lasso_regression")
        self.assertIs(model_cls_alias, LassoRegression)

        models_list = list_models(category="regression")
        self.assertIn("LassoRegression", models_list)

    def test_tensor_abs_subgradient(self):
        """Verify Tensor.abs() computes correct values and subgradients."""
        x = Tensor([-4.0, 5.0, 0.0], requires_grad=True)
        y = x.abs().sum()
        y.backward()

        self.assertEqual(x.abs().to_list(), [4.0, 5.0, 0.0])
        self.assertEqual(x.grad.to_list(), [-1.0, 1.0, 0.0])

    def test_forward_computation(self):
        """Verify forward pass evaluates y = X @ W + b precisely."""
        model = LassoRegression(in_features=2, out_features=1, alpha=0.5, fit_intercept=True, mode="tensor")
        model.weights.data[0] = 4.0
        model.weights.data[1] = -1.0
        model.bias.data[0] = 2.0

        X = Tensor([[1.0, 2.0], [0.0, 3.0]])
        # sample 0: 1*4 + 2*(-1) + 2 = 4 - 2 + 2 = 4.0
        # sample 1: 0*4 + 3*(-1) + 2 = 0 - 3 + 2 = -1.0
        preds = model(X)
        self.assertEqual(preds.shape, (2, 1))
        self.assertAlmostEqual(float(preds.data[0]), 4.0, places=5)
        self.assertAlmostEqual(float(preds.data[1]), -1.0, places=5)

    def test_gradient_computation_with_l1_penalty(self):
        """Verify gradients computed by Autograd include MSE gradient plus alpha * sign(W)."""
        alpha = 1.5
        model = LassoRegression(in_features=1, out_features=1, alpha=alpha, fit_intercept=True, mode="tensor")
        model.weights.data[0] = 3.0
        model.bias.data[0] = 0.0

        # One sample: x = 2.0, y = 8.0
        # y_hat = 2.0 * 3.0 = 6.0
        # mse = (6.0 - 8.0)^2 = 4.0
        # l1_penalty = alpha * |3.0| = 1.5 * 3.0 = 4.5
        # total_loss = 8.5
        # d_mse / dW = 2 * (6.0 - 8.0) * 2.0 = -8.0
        # d_l1 / dW = alpha * sign(3.0) = 1.5 * 1.0 = +1.5
        # d_total / dW = -8.0 + 1.5 = -6.5
        # d_total / db = 2 * (6.0 - 8.0) = -4.0 (bias not penalized)
        X = Tensor([[2.0]])
        y = Tensor([[8.0]])

        pred = model(X)
        mse = ((pred - y) ** 2).mean()
        l1 = (model.weights.abs()).sum() * alpha
        loss = mse + l1
        loss.backward()

        self.assertIsNotNone(model.weights.grad)
        self.assertIsNotNone(model.bias.grad)
        self.assertAlmostEqual(float(model.weights.grad.data[0]), -6.5, places=4)
        self.assertAlmostEqual(float(model.bias.grad.data[0]), -4.0, places=4)

    def test_feature_sparsity_selection(self):
        """
        Verify that Lasso zeroes out irrelevant noise features.
        Dataset: y = 3*x0 - 2*x1 + noise (features 2, 3, 4 are pure noise).
        """
        np.random.seed(42)
        n = 180
        X = np.random.uniform(-2.0, 2.0, size=(n, 5))
        y = 3.0 * X[:, 0:1] - 2.0 * X[:, 1:2] + 1.0

        lasso = LassoRegression(in_features=5, out_features=1, alpha=0.8)
        lasso.fit(X, y, epochs=200, lr=0.03, tol=1e-2, verbose=False)

        weights = [float(w) for w in lasso.weight.data]
        self.assertTrue(lasso.is_fitted)

        # Features 0 and 1 must be retained (non-zero)
        self.assertGreater(abs(weights[0]), 0.5)
        self.assertGreater(abs(weights[1]), 0.5)

        # Sparsity score must indicate pruned features (at least 2 noise features zeroed out)
        sparsity_score = lasso.sparsity(tol=1e-2)
        self.assertGreaterEqual(sparsity_score, 0.4)

    def test_alpha_zero_matches_ols(self):
        """Verify alpha = 0 behaves like unregularized linear regression."""
        X = Tensor([[1.0], [2.0], [3.0]])
        y = Tensor([[2.0], [4.0], [6.0]])

        lasso = LassoRegression(in_features=1, out_features=1, alpha=0.0)
        lasso.fit(X, y, epochs=100, lr=0.05, verbose=False)

        preds = lasso.predict(X)
        self.assertAlmostEqual(float(lasso.weight.data[0]), 2.0, delta=0.2)

    def test_negative_alpha_raises_value_error(self):
        """Verify alpha cannot be negative."""
        with self.assertRaises(ValueError):
            LassoRegression(in_features=2, alpha=-0.5)

    def test_config_and_serialization(self):
        """Verify get_config and state_dict save/load."""
        import tempfile
        model = LassoRegression(in_features=3, out_features=1, alpha=2.0, fit_intercept=True)
        config = model.get_config()
        self.assertEqual(config["alpha"], 2.0)
        self.assertEqual(config["in_features"], 3)

        model.weight.data[0] = 5.55
        model.weight.data[1] = 0.0
        model.weight.data[2] = -3.33
        model.bias_param.data[0] = 1.23

        with tempfile.TemporaryDirectory() as tmpdir:
            chk_path = os.path.join(tmpdir, "lasso.chk")
            model.save(chk_path)

            loaded = LassoRegression(in_features=3, out_features=1, alpha=2.0)
            loaded.load(chk_path)

            self.assertAlmostEqual(float(loaded.weight.data[0]), 5.55, places=4)
            self.assertAlmostEqual(float(loaded.weight.data[1]), 0.0, places=4)
            self.assertAlmostEqual(float(loaded.weight.data[2]), -3.33, places=4)
            self.assertAlmostEqual(float(loaded.bias_param.data[0]), 1.23, places=4)


if __name__ == "__main__":
    unittest.main()
