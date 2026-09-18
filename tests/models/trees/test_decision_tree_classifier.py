import os
import sys

# Ensure repository root is on sys.path when test is executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import unittest
import numpy as np

import nevula as nv
from nevula.core.tensor import Tensor
from nevula.models.registry import get_model, list_models
from nevula.models.trees.decision_tree import DecisionTreeClassifier, TreeNode


class TestDecisionTreeClassifier(unittest.TestCase):
    """
    Independent unit test suite for CART DecisionTreeClassifier.
    """

    def test_registry_registration(self):
        """Verify DecisionTreeClassifier is registered in trees and classification."""
        dtc_cls = get_model("decision_tree_classifier")
        self.assertIs(dtc_cls, DecisionTreeClassifier)

        dtc_alias = get_model("dtc")
        self.assertIs(dtc_alias, DecisionTreeClassifier)

        dtc_classif = get_model("decision_tree_classification")
        self.assertIs(dtc_classif, DecisionTreeClassifier)

        tree_models = list_models(category="trees")
        self.assertIn("DecisionTreeClassifier", tree_models)

        classif_models = list_models(category="classification")
        self.assertIn("DecisionTreeClassifier", classif_models)

    def test_invalid_parameters(self):
        """Verify parameter validation."""
        with self.assertRaises(ValueError):
            DecisionTreeClassifier(criterion="invalid")
        with self.assertRaises(ValueError):
            DecisionTreeClassifier(max_depth=0)
        with self.assertRaises(ValueError):
            DecisionTreeClassifier(min_samples_split=1)
        with self.assertRaises(ValueError):
            DecisionTreeClassifier(min_samples_leaf=0)
        with self.assertRaises(ValueError):
            DecisionTreeClassifier(min_impurity_decrease=-0.5)

    def test_pure_node_creates_single_leaf(self):
        """Verify that a single pure class dataset creates a single leaf node."""
        X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        y = np.array([1, 1, 1])

        clf = DecisionTreeClassifier()
        clf.fit(X, y)

        self.assertIsNotNone(clf.root)
        self.assertTrue(clf.root.is_leaf)
        self.assertEqual(clf.root.value, 0)  # mapped index 0
        self.assertEqual(clf.tree_depth(), 0)
        self.assertEqual(clf.n_leaves(), 1)

        preds = clf.predict(X)
        self.assertTrue((np.array(preds.to_list()).ravel() == 1).all())

    def test_binary_separable_classification(self):
        """Verify accurate binary classification on 2D linearly separable clusters."""
        np.random.seed(42)
        n = 40
        X0 = np.random.randn(n, 2) * 0.4 + np.array([-2.0, -2.0])
        X1 = np.random.randn(n, 2) * 0.4 + np.array([2.0, 2.0])
        X = np.vstack([X0, X1])
        y = np.array([0] * n + [1] * n)

        clf = DecisionTreeClassifier(criterion="gini", max_depth=3)
        clf.fit(X, y)

        preds = clf.predict(X)
        self.assertEqual(preds.shape, (len(X), 1))

        acc = clf.evaluate(X, y, metric="accuracy")
        self.assertAlmostEqual(acc, 1.0, places=4)

        f1 = clf.evaluate(X, y, metric="f1")
        self.assertAlmostEqual(f1, 1.0, places=4)

    def test_entropy_criterion_and_predict_proba(self):
        """Verify entropy criterion and calibrated predict_proba output."""
        np.random.seed(42)
        n = 30
        X = np.vstack([np.random.randn(n, 2) - 1.5, np.random.randn(n, 2) + 1.5])
        y = np.array([0] * n + [1] * n)

        clf = DecisionTreeClassifier(criterion="entropy", max_depth=4)
        clf.fit(X, y)

        probs = clf.predict_proba(X)
        probs_np = np.array(probs.to_list())

        self.assertEqual(probs.shape, (len(X), 2))
        # Each probability row must sum to 1.0
        np.testing.assert_allclose(np.sum(probs_np, axis=1), np.ones(len(X)), atol=1e-5)
        # Probabilities must be bounded in [0, 1]
        self.assertTrue((probs_np >= 0.0).all() and (probs_np <= 1.0).all())

        log_loss = clf.evaluate(X, y, metric="log_loss")
        self.assertLess(log_loss, 0.5)

    def test_multiclass_classification(self):
        """Verify multi-class (3 classes) classification and metrics."""
        np.random.seed(42)
        n = 40
        X0 = np.random.randn(n, 2) * 0.3 + np.array([-3.0, -3.0])
        X1 = np.random.randn(n, 2) * 0.3 + np.array([0.0, 3.0])
        X2 = np.random.randn(n, 2) * 0.3 + np.array([3.0, -3.0])
        X = np.vstack([X0, X1, X2])
        y = np.array([0] * n + [1] * n + [2] * n)

        clf = DecisionTreeClassifier(criterion="gini", max_depth=5)
        clf.fit(X, y)

        self.assertEqual(clf.n_classes_, 3)

        preds = clf.predict(X)
        self.assertEqual(preds.shape, (len(X), 1))

        acc = clf.evaluate(X, y, metric="accuracy")
        self.assertGreaterEqual(acc, 0.95)

        probs = clf.predict_proba(X)
        self.assertEqual(probs.shape, (len(X), 3))

    def test_non_linear_classification(self):
        """Verify tree partitions non-linear rectangular interior decision boundaries."""
        np.random.seed(42)
        n = 30
        x0 = np.linspace(-2, 2, n)
        x1 = np.linspace(-2, 2, n)
        X0, X1 = np.meshgrid(x0, x1)
        X = np.column_stack([X0.ravel(), X1.ravel()])
        # Non-linear 2D box region
        y = ((X[:, 0] > -0.6) & (X[:, 0] < 0.7) & (X[:, 1] > -0.7) & (X[:, 1] < 0.6)).astype(int)

        clf = DecisionTreeClassifier(criterion="gini", max_depth=5)
        clf.fit(X, y)

        acc = clf.evaluate(X, y, metric="accuracy")
        self.assertGreaterEqual(acc, 0.98)
        self.assertGreater(clf.tree_depth(), 1)

    def test_stopping_conditions_and_depth_limits(self):
        """Verify max_depth and min_samples_leaf constraints."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] * X[:, 1] > 0).astype(int)

        # Restricted depth
        clf_shallow = DecisionTreeClassifier(max_depth=2).fit(X, y)
        self.assertLessEqual(clf_shallow.tree_depth(), 2)

        # Min samples leaf
        clf_leaf = DecisionTreeClassifier(min_samples_leaf=20).fit(X, y)
        self.assertIsNotNone(clf_leaf.root)

    def test_feature_importances(self):
        """Verify that informative feature receives highest importance and sums to 1."""
        np.random.seed(42)
        n = 100
        # Feature 0 is perfectly predictive, Feature 1 is pure random noise
        x0 = np.random.uniform(-3, 3, size=(n, 1))
        x1 = np.random.randn(n, 1)
        X = np.hstack([x0, x1])
        y = (x0[:, 0] > 0).astype(int)

        clf = DecisionTreeClassifier(max_depth=3).fit(X, y)
        imp = clf.feature_importances()

        self.assertEqual(len(imp), 2)
        self.assertAlmostEqual(float(np.sum(imp)), 1.0, places=4)
        self.assertGreater(imp[0], imp[1])

    def test_state_dict_and_config_serialization(self):
        """Verify model configuration and tree state dict serialization."""
        np.random.seed(42)
        X = np.array([[1.0, 2.0], [-1.0, -2.0], [2.0, 1.0], [-2.0, -1.0]])
        y = np.array([1, 0, 1, 0])

        clf = DecisionTreeClassifier(criterion="entropy", max_depth=3, min_samples_split=2)
        clf.fit(X, y)

        preds_orig = clf.predict(X)

        cfg = clf.get_config()
        self.assertEqual(cfg["criterion"], "entropy")
        self.assertEqual(cfg["max_depth"], 3)
        self.assertEqual(cfg["n_classes"], 2)

        sd = clf.state_dict()
        self.assertIn("tree_root", sd)
        self.assertIn("classes_", sd)

        # Recreate model and load state dict
        clf_loaded = DecisionTreeClassifier(criterion="entropy")
        clf_loaded.load_state_dict(sd)

        self.assertTrue(clf_loaded.is_fitted)
        preds_loaded = clf_loaded.predict(X)
        np.testing.assert_array_equal(preds_orig.to_list(), preds_loaded.to_list())


if __name__ == "__main__":
    unittest.main()
