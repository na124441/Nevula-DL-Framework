import os
import sys

# Ensure repository root is on sys.path when test is executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import unittest
import numpy as np

import nevula as nv
from nevula.core.tensor import Tensor
from nevula.models.registry import get_model, list_models
from nevula.models.classification.svm import SVC, SVM
from nevula.nn.losses import HingeLoss
from nevula.nn import functional as F


class TestSupportVectorClassifier(unittest.TestCase):
    """
    Independent unit test suite for Support Vector Classifier (SVC / SVM) and Hinge Loss.
    """

    def test_registry_registration(self):
        """Verify SVC and SVM are registered under classification category."""
        svc_cls = get_model("svc")
        self.assertIs(svc_cls, SVC)

        svm_cls = get_model("svm")
        self.assertIs(svm_cls, SVC)

        alias_cls = get_model("support_vector_classifier")
        self.assertIs(alias_cls, SVC)

        models_list = list_models(category="classification")
        self.assertIn("SVC", models_list)

    def test_hinge_loss_module_and_functional(self):
        """Verify HingeLoss module and functional hinge_loss output and subgradients."""
        pred = nv.tensor([0.5, -0.5, 2.0], requires_grad=True)
        target = nv.tensor([1.0, 1.0, -1.0])

        # Hinge loss: max(0, 1 - y * z)
        # For sample 0: 1 - 1.0 * 0.5 = 0.5 -> max(0, 0.5) = 0.5
        # For sample 1: 1 - 1.0 * (-0.5) = 1.5 -> max(0, 1.5) = 1.5
        # For sample 2: 1 - (-1.0) * 2.0 = 3.0 -> max(0, 3.0) = 3.0
        # Mean = (0.5 + 1.5 + 3.0) / 3 = 5.0 / 3 = 1.66667
        loss_mod = HingeLoss(reduction="mean")
        loss_val = loss_mod(pred, target)
        self.assertAlmostEqual(float(loss_val.item()), 5.0 / 3.0, places=4)

        loss_fn = F.hinge_loss(pred, target, reduction="mean")
        self.assertAlmostEqual(float(loss_fn.item()), 5.0 / 3.0, places=4)

        # Autograd backward test
        loss_val.backward()
        # dL/dz_0 = -1/3, dL/dz_1 = -1/3, dL/dz_2 = +1/3
        grad_np = np.array(pred.grad.to_list())
        expected_grad = np.array([-1.0 / 3.0, -1.0 / 3.0, 1.0 / 3.0])
        np.testing.assert_allclose(grad_np, expected_grad, atol=1e-4)

    def test_invalid_parameters(self):
        """Verify validation of constructor parameters."""
        with self.assertRaises(ValueError):
            SVC(in_features=0)
        with self.assertRaises(ValueError):
            SVC(in_features=2, n_classes=1)
        with self.assertRaises(ValueError):
            SVC(in_features=2, C=-1.0)
        with self.assertRaises(ValueError):
            SVC(in_features=2, loss="invalid_loss")
        with self.assertRaises(ValueError):
            SVC(in_features=2, mode="invalid_mode")

    def test_binary_classification_fit_predict(self):
        """Verify binary SVM classification on synthetic separable 2D data."""
        np.random.seed(42)
        n = 50
        X0 = np.random.randn(n, 2) * 0.5 + np.array([-2.0, -2.0])
        X1 = np.random.randn(n, 2) * 0.5 + np.array([2.0, 2.0])
        X = np.vstack([X0, X1])
        y = np.array([0] * n + [1] * n)

        clf = SVC(in_features=2, n_classes=2, C=5.0, loss="hinge")
        clf.fit(X, y, epochs=100, lr=0.08, optimizer="adam")

        preds = clf.predict(X)
        self.assertEqual(preds.shape, (len(X), 1))

        acc = clf.evaluate(X, y, metric="accuracy")
        self.assertGreaterEqual(acc, 0.95)

        # Check support vectors
        n_sv = clf.n_support_(X, y)
        sv_ratio = clf.support_vectors_ratio(X, y)
        self.assertGreater(n_sv, 0)
        self.assertLessEqual(sv_ratio, 1.0)

    def test_multiclass_ovr_classification(self):
        """Verify multiclass One-vs-Rest SVM classification."""
        np.random.seed(42)
        n = 40
        X0 = np.random.randn(n, 2) * 0.5 + np.array([-3.0, 0.0])
        X1 = np.random.randn(n, 2) * 0.5 + np.array([0.0, 3.0])
        X2 = np.random.randn(n, 2) * 0.5 + np.array([3.0, 0.0])
        X = np.vstack([X0, X1, X2])
        y = np.array([0] * n + [1] * n + [2] * n)

        clf = SVM(in_features=2, n_classes=3, C=2.0, loss="hinge")
        clf.fit(X, y, epochs=100, lr=0.06, optimizer="adam")

        self.assertEqual(clf.coef_.shape, (3, 2))
        self.assertEqual(clf.intercept_.shape, (3,))

        preds = clf.predict(X)
        self.assertEqual(preds.shape, (len(X), 1))

        acc = clf.evaluate(X, y, metric="accuracy")
        self.assertGreaterEqual(acc, 0.90)

    def test_squared_hinge_loss(self):
        """Verify training with squared hinge loss."""
        np.random.seed(42)
        n = 30
        X = np.vstack([np.random.randn(n, 2) - 2.0, np.random.randn(n, 2) + 2.0])
        y = np.array([0] * n + [1] * n)

        clf = SVC(in_features=2, n_classes=2, C=1.0, loss="squared_hinge")
        clf.fit(X, y, epochs=80, lr=0.05, optimizer="adam")

        acc = clf.evaluate(X, y, metric="accuracy")
        self.assertGreaterEqual(acc, 0.90)

    def test_regularization_effect(self):
        """Verify that smaller C results in a smaller weight norm (stronger regularization)."""
        np.random.seed(42)
        n = 40
        X = np.vstack([np.random.randn(n, 2) - 1.0, np.random.randn(n, 2) + 1.0])
        y = np.array([0] * n + [1] * n)

        clf_small_c = SVC(in_features=2, n_classes=2, C=0.05).fit(X, y, epochs=80, lr=0.05, optimizer="adam")
        clf_large_c = SVC(in_features=2, n_classes=2, C=20.0).fit(X, y, epochs=80, lr=0.05, optimizer="adam")

        norm_small_c = float(np.linalg.norm(clf_small_c.coef_))
        norm_large_c = float(np.linalg.norm(clf_large_c.coef_))

        # Smaller C penalizes ||w||^2 more heavily, yielding smaller norm
        self.assertLess(norm_small_c, norm_large_c)

    def test_state_dict_and_config(self):
        """Verify get_config and state_dict roundtrip."""
        clf = SVC(in_features=3, n_classes=2, C=2.5, loss="hinge", fit_intercept=True)
        config = clf.get_config()
        self.assertEqual(config["in_features"], 3)
        self.assertEqual(config["n_classes"], 2)
        self.assertEqual(config["C"], 2.5)

        sd = clf.state_dict()
        self.assertIn("linear.weight", sd)
        self.assertIn("linear.bias", sd)


if __name__ == "__main__":
    unittest.main()
