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
from nevula.metrics.regression import mean_squared_error, r2_score


class TestLinearRegression(unittest.TestCase):
    """
    Comprehensive test suite for Nevula Linear Regression.
    Covers mathematical correctness, autograd gradient calculation,
    training convergence, multi-feature learning, edge cases, and registry.
    """

    def test_registry_registration(self):
        """Verify LinearRegression is properly registered in MODEL_REGISTRY."""
        model_cls = get_model("linear_regression")
        self.assertIs(model_cls, LinearRegression)

        # Case-insensitive / CamelCase lookup
        model_cls_camel = get_model("LinearRegression")
        self.assertIs(model_cls_camel, LinearRegression)

        # Verify listed in regression category
        regression_models = list_models(category="regression")
        self.assertIn("LinearRegression", regression_models)

    def test_forward_mathematical_correctness(self):
        """Verify forward pass evaluates y = X @ W + b precisely."""
        model = LinearRegression(in_features=2, out_features=1, fit_intercept=True, mode="tensor")
        
        # Set deterministic weights: W = [[2.0], [-1.0]], b = [0.5]
        model.weights.data[0] = 2.0
        model.weights.data[1] = -1.0
        model.bias.data[0] = 0.5

        X = Tensor([[1.0, 3.0], [2.0, 0.0]])
        # Expected:
        # sample 0: 1*2.0 + 3*(-1.0) + 0.5 = 2.0 - 3.0 + 0.5 = -0.5
        # sample 1: 2*2.0 + 0*(-1.0) + 0.5 = 4.0 + 0.0 + 0.5 = 4.5
        preds = model(X)
        self.assertEqual(preds.shape, (2, 1))
        self.assertAlmostEqual(float(preds.data[0]), -0.5, places=5)
        self.assertAlmostEqual(float(preds.data[1]), 4.5, places=5)

    def test_gradient_computation(self):
        """Verify gradients computed by Autograd match analytical MSE loss derivatives."""
        model = LinearRegression(in_features=1, out_features=1, fit_intercept=True, mode="tensor")
        model.weights.data[0] = 1.0
        model.bias.data[0] = 0.0

        # One sample: x = 2.0, true_y = 5.0
        # y_hat = 2.0 * 1.0 + 0.0 = 2.0
        # loss = (2.0 - 5.0)^2 = 9.0
        # dloss/dy_hat = 2 * (2.0 - 5.0) = -6.0
        # dloss/dW = dloss/dy_hat * x = -6.0 * 2.0 = -12.0
        # dloss/db = dloss/dy_hat * 1 = -6.0
        X = Tensor([[2.0]])
        y = Tensor([[5.0]])

        pred = model(X)
        loss = ((pred - y) ** 2).sum()
        loss.backward()

        self.assertIsNotNone(model.weights.grad)
        self.assertIsNotNone(model.bias.grad)
        self.assertAlmostEqual(float(model.weights.grad.data[0]), -12.0, places=4)
        self.assertAlmostEqual(float(model.bias.grad.data[0]), -6.0, places=4)

    def test_synthetic_convergence_1d(self):
        """
        Verify convergence on synthetic linear relation:
            y = 3x + 2
        Learned parameters must converge to W approx 3 and b approx 2.
        """
        np.random.seed(42)
        n = 200
        X_np = np.random.uniform(-3.0, 3.0, size=(n, 1))
        y_np = 3.0 * X_np + 2.0

        model = LinearRegression(in_features=1, out_features=1, fit_intercept=True)
        model.fit(X_np, y_np, epochs=150, lr=0.05, optimizer="sgd", verbose=False)

        w_learned = float(model.weight.data[0])
        b_learned = float(model.bias_param.data[0])

        self.assertAlmostEqual(w_learned, 3.0, delta=0.1)
        self.assertAlmostEqual(b_learned, 2.0, delta=0.1)
        self.assertTrue(model.is_fitted)

        # Check loss decreased
        self.assertLess(model.loss_history[-1], model.loss_history[0])
        self.assertLess(model.loss_history[-1], 0.01)

    def test_synthetic_convergence_multivariate(self):
        """
        Verify convergence on multi-feature synthetic dataset:
            y = 2*x1 - 4*x2 + 5
        """
        np.random.seed(123)
        n = 300
        X_np = np.random.uniform(-2.0, 2.0, size=(n, 2))
        y_np = 2.0 * X_np[:, 0:1] - 4.0 * X_np[:, 1:2] + 5.0

        model = LinearRegression(in_features=2, out_features=1, fit_intercept=True, mode="tensor")
        model.fit(X_np, y_np, epochs=160, lr=0.04, optimizer="sgd", verbose=False)

        w1 = float(model.weight.data[0])
        w2 = float(model.weight.data[1])
        b = float(model.bias_param.data[0])

        self.assertAlmostEqual(w1, 2.0, delta=0.15)
        self.assertAlmostEqual(w2, -4.0, delta=0.15)
        self.assertAlmostEqual(b, 5.0, delta=0.15)

    def test_minibatch_dataloader_training(self):
        """Verify training succeeds with mini-batch DataLoader enabled."""
        np.random.seed(99)
        X_np = np.random.randn(80, 2)
        y_np = X_np[:, 0:1] + X_np[:, 1:2]

        model = LinearRegression(in_features=2, out_features=1)
        model.fit(X_np, y_np, epochs=50, lr=0.02, batch_size=16, optimizer="sgd", verbose=False)

        self.assertTrue(model.is_fitted)
        self.assertEqual(len(model.loss_history), 50)
        self.assertLess(model.loss_history[-1], model.loss_history[0])

    def test_prediction_and_evaluation_metrics(self):
        """Verify predict and evaluate work with R2 and MSE metrics."""
        X = Tensor([[1.0], [2.0], [3.0], [4.0]])
        y = Tensor([[2.0], [4.0], [6.0], [8.0]])

        model = LinearRegression(in_features=1, out_features=1)
        model.fit(X, y, epochs=100, lr=0.05, verbose=False)

        preds = model.predict(X)
        self.assertEqual(preds.shape, (4, 1))

        # Under inference, preds shouldn't require grad
        self.assertFalse(preds.requires_grad)

        r2 = model.evaluate(X, y, metric="r2")
        mse = model.evaluate(X, y, metric="mse")
        self.assertGreater(r2, 0.95)
        self.assertLess(mse, 0.1)

    def test_edge_case_single_sample(self):
        """Verify prediction and forward pass handle single sample (N=1)."""
        model = LinearRegression(in_features=3, out_features=1)
        single_x = Tensor([[1.0, 0.0, -1.0]])
        pred = model.predict(single_x)
        self.assertEqual(pred.shape, (1, 1))

    def test_edge_case_zeros_and_negatives(self):
        """Verify model handles all-zeros and negative input values gracefully."""
        model = LinearRegression(in_features=2, out_features=1)
        X_zero = Tensor([[0.0, 0.0], [-1.0, -2.0]])
        pred = model(X_zero)
        self.assertEqual(pred.shape, (2, 1))

    def test_dimension_mismatch_errors(self):
        """Verify descriptive errors on invalid feature dimensions."""
        model = LinearRegression(in_features=3, out_features=1)
        wrong_X = Tensor([[1.0, 2.0]])  # 2 features instead of 3
        with self.assertRaises(ValueError):
            model.predict(wrong_X)

    def test_state_dict_save_and_load(self):
        """Verify state_dict can be saved and loaded accurately."""
        import tempfile
        import os

        model1 = LinearRegression(in_features=2, out_features=1)
        model1.weight.data[0] = 1.234
        model1.weight.data[1] = 5.678
        model1.bias_param.data[0] = 9.999

        with tempfile.TemporaryDirectory() as tmpdir:
            chk_path = os.path.join(tmpdir, "model.chk")
            model1.save(chk_path)

            model2 = LinearRegression(in_features=2, out_features=1)
            model2.load(chk_path)

            self.assertAlmostEqual(float(model2.weight.data[0]), 1.234, places=4)
            self.assertAlmostEqual(float(model2.weight.data[1]), 5.678, places=4)
            self.assertAlmostEqual(float(model2.bias_param.data[0]), 9.999, places=4)


if __name__ == "__main__":
    unittest.main()
