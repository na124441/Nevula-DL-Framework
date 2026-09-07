import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from nevula.core.tensor import Tensor


class TestSubtraction(unittest.TestCase):
    def test_sub_forward_scalar(self):
        """Forward test: z = x - y produces 3.0"""
        x = Tensor(5.0)
        y = Tensor(2.0)
        z = x - y
        self.assertEqual(z.item(), 3.0)
        self.assertEqual(z.shape, ())

    def test_sub_backward_scalar(self):
        """Backward test: z = x - y => x.grad = 1.0, y.grad = -1.0"""
        x = Tensor(5.0, requires_grad=True)
        y = Tensor(2.0, requires_grad=True)
        z = x - y
        z.backward()

        self.assertEqual(z.item(), 3.0)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)
        self.assertEqual(x.grad.item(), 1.0)
        self.assertEqual(y.grad.item(), -1.0)

    def test_sub_constant_right(self):
        """Backward test: z = x - 2.0 => x.grad = 1.0"""
        x = Tensor(5.0, requires_grad=True)
        z = x - 2.0
        z.backward()

        self.assertEqual(z.item(), 3.0)
        self.assertEqual(x.grad.item(), 1.0)

    def test_sub_constant_left(self):
        """Backward test: z = 3.0 - x => x.grad = -1.0"""
        x = Tensor(5.0, requires_grad=True)
        z = 3.0 - x
        z.backward()

        self.assertEqual(z.item(), -2.0)
        self.assertEqual(x.grad.item(), -1.0)

    def test_sub_multidimensional(self):
        """Forward and backward for multidimensional tensors"""
        x = Tensor([[5.0, 7.0], [9.0, 11.0]], requires_grad=True)
        y = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        z = (x - y).sum()
        z.backward()

        self.assertEqual(x.grad.to_list(), [[1.0, 1.0], [1.0, 1.0]])
        self.assertEqual(y.grad.to_list(), [[-1.0, -1.0], [-1.0, -1.0]])

    def test_sub_broadcasting(self):
        """Broadcasting subtraction: bias subtraction and gradient accumulation"""
        x = Tensor([[10.0, 20.0], [30.0, 40.0]], requires_grad=True)  # (2, 2)
        b = Tensor([1.0, 2.0], requires_grad=True)                   # (2,)
        z = (x - b).sum()
        z.backward()

        # x receives 1.0 at every element
        self.assertEqual(x.grad.to_list(), [[1.0, 1.0], [1.0, 1.0]])
        # b was broadcasted across 2 rows, so its gradient sums to -2.0 for each element
        self.assertEqual(b.grad.to_list(), [-2.0, -2.0])

    def test_sub_self_branching(self):
        """Multi-path DAG test: z = x - x => x.grad = 0.0"""
        x = Tensor(4.0, requires_grad=True)
        z = x - x
        z.backward()

        self.assertEqual(z.item(), 0.0)
        self.assertEqual(x.grad.item(), 0.0)

    def test_sub_numerical_gradient(self):
        """Numerical gradient check: [f(x + eps) - f(x - eps)] / (2 * eps)"""
        eps = 1e-6
        x_val = 5.0
        y_val = 2.0

        # Analytical gradients
        x = Tensor(x_val, requires_grad=True)
        y = Tensor(y_val, requires_grad=True)
        z = x - y
        z.backward()

        # Numerical gradient for x
        f_plus_x = (x_val + eps) - y_val
        f_minus_x = (x_val - eps) - y_val
        num_grad_x = (f_plus_x - f_minus_x) / (2 * eps)

        # Numerical gradient for y
        f_plus_y = x_val - (y_val + eps)
        f_minus_y = x_val - (y_val - eps)
        num_grad_y = (f_plus_y - f_minus_y) / (2 * eps)

        self.assertAlmostEqual(x.grad.item(), num_grad_x, places=5)
        self.assertAlmostEqual(y.grad.item(), num_grad_y, places=5)


if __name__ == '__main__':
    unittest.main()
