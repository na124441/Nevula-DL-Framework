import os
import sys

# Ensure repository root is on sys.path when test is executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import unittest
import numpy as np

import nevula as nv
from nevula.core.tensor import Tensor
from nevula.models.registry import get_model, list_models
from nevula.models.classification.logistic import LogisticRegression
from nevula.nn import functional as F


class TestLogisticRegression(unittest.TestCase):
    """
    Independent unit test suite for LogisticRegression and Sigmoid infrastructure.
    """

    def test_registry_registration(self):
        """Verify model is registered in category 'classification' under canonical name and alias."""
        model_cls = get_model("logistic_regression")
        self.assertIs(model_cls, LogisticRegression)

        alias_cls = get_model("logistic")
        self.assertIs(alias_cls, LogisticRegression)

        models_list = list_models(category="classification")
        self.assertIn("LogisticRegression", models_list)

    def test_sigmoid_and_log_sigmoid(self):
        """Verify sigmoid and log_sigmoid functional evaluations."""
        x = nv.tensor([0.0])
        sig = nv.sigmoid(x)
        self.assertAlmostEqual(float(sig.item()), 0.5, places=5)

        log_sig = F.log_sigmoid(x)
        self.assertAlmostEqual(float(log_sig.item()), float(np.log(0.5)), places=5)

    def test_softmax_and_log_softmax(self):
        """Verify softmax and log_softmax output probabilities, sum-to-one, and autograd."""
        x = nv.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        probs = nv.softmax(x, dim=-1)
        self.assertEqual(probs.shape, (1, 3))

        probs_np = np.array(probs.to_list(), dtype=np.float64)
        # Sum along last dimension must be 1.0
        self.assertAlmostEqual(float(np.sum(probs_np)), 1.0, places=5)

        # LogSoftmax equality with log(probs)
        log_probs = nv.log_softmax(x, dim=-1)
        log_probs_np = np.array(log_probs.to_list(), dtype=np.float64)
        np.testing.assert_allclose(log_probs_np, np.log(probs_np), atol=1e-5)

        # Autograd backward: gradient of sum(probs) w.r.t input must be 0
        probs.sum().backward()
        np.testing.assert_allclose(np.array(x.grad.to_list()), 0.0, atol=1e-6)

    def test_binary_classification_fit_and_predict(self):
        """Verify binary logistic regression achieves high accuracy on separable data."""
        np.random.seed(42)
        n = 50
        c0 = np.random.normal(loc=[-2.0, -2.0], scale=0.5, size=(n, 2))
        c1 = np.random.normal(loc=[2.0, 2.0], scale=0.5, size=(n, 2))
        X = np.vstack([c0, c1])
        y = np.array([0] * n + [1] * n)

        clf = LogisticRegression(in_features=2, n_classes=2, penalty="l2", C=1.0)
        clf.fit(X, y, epochs=80, lr=0.1, verbose=False)

        self.assertTrue(clf.is_fitted)

        # Decision function shape (2*n, 1)
        logits = clf.decision_function(X)
        self.assertEqual(logits.shape, (2 * n, 1))

        # Predict proba shape (2*n, 2) and row sums == 1.0
        probs = clf.predict_proba(X)
        self.assertEqual(probs.shape, (2 * n, 2))
        probs_np = np.array(probs.to_list(), dtype=np.float64)
        np.testing.assert_allclose(np.sum(probs_np, axis=1), 1.0, atol=1e-5)

        acc = clf.evaluate(X, y, metric="accuracy")
        self.assertGreaterEqual(acc, 0.98)

        f1 = clf.evaluate(X, y, metric="f1")
        self.assertGreaterEqual(f1, 0.98)

    def test_multiclass_softmax_classification(self):
        """Verify multiclass logistic regression with Softmax achieves high accuracy."""
        np.random.seed(42)
        n = 40
        c0 = np.random.normal(loc=[-3.0, 0.0], scale=0.5, size=(n, 2))
        c1 = np.random.normal(loc=[3.0, 0.0], scale=0.5, size=(n, 2))
        c2 = np.random.normal(loc=[0.0, 3.0], scale=0.5, size=(n, 2))
        X = np.vstack([c0, c1, c2])
        y = np.array([0] * n + [1] * n + [2] * n)

        clf = LogisticRegression(in_features=2, n_classes=3, penalty="l2", C=1.0)
        clf.fit(X, y, epochs=100, lr=0.1, verbose=False)

        probs = clf.predict_proba(X)
        self.assertEqual(probs.shape, (3 * n, 3))
        probs_np = np.array(probs.to_list(), dtype=np.float64)
        np.testing.assert_allclose(np.sum(probs_np, axis=1), 1.0, atol=1e-5)

        acc = clf.evaluate(X, y, metric="accuracy")
        self.assertGreaterEqual(acc, 0.95)

    def test_l1_and_l2_regularization(self):
        """Verify L1 and L2 regularized models train successfully."""
        X = np.random.normal(0, 1, size=(40, 3))
        y = np.random.randint(0, 2, size=(40,))

        clf_l1 = LogisticRegression(in_features=3, penalty="l1", C=0.5)
        clf_l1.fit(X, y, epochs=20, lr=0.05, verbose=False)
        self.assertTrue(clf_l1.is_fitted)

        clf_l2 = LogisticRegression(in_features=3, penalty="l2", C=0.5)
        clf_l2.fit(X, y, epochs=20, lr=0.05, verbose=False)
        self.assertTrue(clf_l2.is_fitted)

    def test_serialization(self):
        """Verify model configuration and state serialization."""
        clf = LogisticRegression(in_features=4, n_classes=2, penalty="l2", C=2.0)
        config = clf.get_config()
        self.assertEqual(config["in_features"], 4)
        self.assertEqual(config["n_classes"], 2)
        self.assertEqual(config["penalty"], "l2")
        self.assertEqual(config["C"], 2.0)

    def test_binary_and_multiclass_cross_entropy_losses(self):
        """Verify BCELoss, BCEWithLogitsLoss, CrossEntropyLoss, and NLLLoss."""
        # 1. Binary Cross Entropy with Logits
        z_bin = nv.tensor([-1.0, 1.0], requires_grad=True)
        y_bin = nv.tensor([0.0, 1.0])
        bce_logits = nv.losses.BCEWithLogitsLoss()(z_bin, y_bin)
        bce_logits.backward()
        self.assertAlmostEqual(bce_logits.item(), 0.31326168, places=5)
        # Gradient should be (sigma(z) - y) / 2
        grad_expected = (np.array([1.0 / (1.0 + np.exp(1.0)) - 0.0, 1.0 / (1.0 + np.exp(-1.0)) - 1.0])) / 2.0
        np.testing.assert_allclose(np.array(z_bin.grad.to_list()), grad_expected, atol=1e-5)

        # 2. Multiclass Cross Entropy and NLLLoss equivalence
        z_multi = nv.tensor([[1.0, 2.0, 0.5]], requires_grad=True)
        y_multi = nv.tensor([1])
        ce_loss = nv.losses.CrossEntropyLoss()(z_multi, y_multi)
        ce_loss.backward()

        log_p = nv.log_softmax(z_multi)
        nll_loss = nv.losses.NLLLoss()(log_p, y_multi)
        self.assertAlmostEqual(ce_loss.item(), nll_loss.item(), places=5)

        # 3. Pos_weight in BCE
        z_pos = nv.tensor([0.0], requires_grad=True)
        y_pos = nv.tensor([1.0])
        loss_unweighted = nv.losses.BCEWithLogitsLoss()(z_pos, y_pos)
        loss_weighted = nv.losses.BCEWithLogitsLoss(pos_weight=nv.tensor([2.0]))(z_pos, y_pos)
        self.assertAlmostEqual(loss_weighted.item(), 2.0 * loss_unweighted.item(), places=5)


if __name__ == "__main__":
    unittest.main()
