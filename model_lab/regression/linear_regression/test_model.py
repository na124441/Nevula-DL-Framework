"""
Isolated experimental test for Linear Regression prototype.
"""

import os
import sys

# Ensure repository root is on sys.path when script is executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import unittest
import numpy as np
from nevula.core.tensor import Tensor
from model_lab.regression.linear_regression.model import ExperimentalLinearRegression


class TestExperimentalLinearModel(unittest.TestCase):
    def test_forward_shape(self):
        X = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        model = ExperimentalLinearRegression(in_features=2, out_features=1)
        pred = model(X)
        self.assertEqual(pred.shape, (3, 1))

    def test_predict_no_grad(self):
        X = Tensor([[1.0], [2.0]])
        model = ExperimentalLinearRegression(in_features=1, out_features=1)
        pred = model.predict(X)
        self.assertEqual(pred.shape, (2, 1))
        self.assertFalse(pred.requires_grad)


if __name__ == "__main__":
    unittest.main()
