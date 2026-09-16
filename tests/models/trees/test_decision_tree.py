import os
import sys

# Ensure repository root is on sys.path when test is executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import unittest
import numpy as np

from nevula.core.tensor import Tensor
from nevula.models.base import BaseModel
from nevula.models.registry import get_model, list_models
from nevula.models.trees.decision_tree import DecisionTreeRegressor, TreeNode


class TestDecisionTreeRegressor(unittest.TestCase):
    """
    Independent test suite for Nevula DecisionTreeRegressor.
    Covers registry lookup, recursive tree building, non-linear fitting,
    stopping criteria, feature importances, and serialization.
    """

    def test_registry_registration(self):
        """Verify DecisionTreeRegressor is registered in category 'trees' under canonical name and aliases."""
        model_cls_tree = get_model("decision_tree_regressor")
        self.assertIs(model_cls_tree, DecisionTreeRegressor)

        model_cls_alias = get_model("decision_tree")
        self.assertIs(model_cls_alias, DecisionTreeRegressor)

        models_list = list_models(category="trees")
        self.assertIn("DecisionTreeRegressor", models_list)

    def test_constant_target_creates_single_leaf(self):
        """Verify that constant target creates a single pure leaf node."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = np.array([5.0, 5.0, 5.0])

        tree = DecisionTreeRegressor(max_depth=3)
        tree.fit(X, y)

        self.assertIsNotNone(tree.root)
        self.assertTrue(tree.root.is_leaf)
        self.assertAlmostEqual(tree.root.value, 5.0)
        self.assertEqual(tree.tree_depth(), 0)
        self.assertEqual(tree.n_leaves(), 1)

    def test_non_linear_step_function_fit(self):
        """Verify DecisionTreeRegressor accurately fits a non-linear step function."""
        np.random.seed(42)
        X = np.linspace(-5, 5, 80).reshape((-1, 1))
        # Step function: y = 0 if x <= 0 else 10
        y = np.where(X[:, 0] <= 0, 0.0, 10.0)

        tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=1)
        tree.fit(X, y)

        r2 = tree.evaluate(X, y, metric="r2")
        self.assertGreater(r2, 0.98)

        # Test inference produces Tensor without grad
        preds = tree.predict(X)
        self.assertIsInstance(preds, Tensor)
        self.assertFalse(preds.requires_grad)

    def test_max_depth_constraint_strictly_enforced(self):
        """Verify tree_depth() never exceeds max_depth."""
        np.random.seed(123)
        X = np.random.randn(100, 3)
        y = np.random.randn(100)

        for depth_limit in [1, 2, 4]:
            tree = DecisionTreeRegressor(max_depth=depth_limit)
            tree.fit(X, y)
            self.assertLessEqual(tree.tree_depth(), depth_limit)

    def test_min_samples_leaf_constraint(self):
        """Verify leaf nodes strictly satisfy min_samples_leaf constraint."""
        np.random.seed(99)
        X = np.random.randn(50, 2)
        y = np.random.randn(50)

        tree = DecisionTreeRegressor(max_depth=5, min_samples_leaf=5)
        tree.fit(X, y)

        def _check_leaves(node: TreeNode):
            if node.is_leaf:
                self.assertGreaterEqual(node.n_samples, 5)
            else:
                _check_leaves(node.left)
                _check_leaves(node.right)

        _check_leaves(tree.root)

    def test_feature_importances_sum_to_one(self):
        """Verify feature importances are non-negative and sum to 1.0."""
        np.random.seed(42)
        X = np.random.randn(100, 4)
        # Feature 0 dominates target: y = 10 * x0
        y = 10.0 * X[:, 0] + np.random.normal(0.0, 0.1, size=(100,))

        tree = DecisionTreeRegressor(max_depth=4)
        tree.fit(X, y)

        importances = tree.feature_importances()
        self.assertEqual(len(importances), 4)
        self.assertTrue(all(imp >= 0.0 for imp in importances))
        self.assertAlmostEqual(float(np.sum(importances)), 1.0, places=5)
        # Feature 0 must have the highest importance score
        self.assertEqual(np.argmax(importances), 0)

    def test_serialization_and_restoration(self):
        """Verify state_dict accurately serializes and reconstructs the tree."""
        import tempfile
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([2.0, 4.0, 6.0, 8.0])

        tree1 = DecisionTreeRegressor(max_depth=2)
        tree1.fit(X, y)
        preds1 = tree1.predict(X).to_list()

        with tempfile.TemporaryDirectory() as tmpdir:
            chk_path = os.path.join(tmpdir, "tree.chk")
            tree1.save(chk_path)

            tree2 = DecisionTreeRegressor(max_depth=2)
            tree2.load(chk_path)
            preds2 = tree2.predict(X).to_list()

            self.assertEqual(preds1, preds2)
            self.assertEqual(tree1.tree_depth(), tree2.tree_depth())
            self.assertEqual(tree1.n_leaves(), tree2.n_leaves())


if __name__ == "__main__":
    unittest.main()
