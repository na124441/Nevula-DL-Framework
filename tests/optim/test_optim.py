import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from nevula.core.tensor import Tensor
from nevula.nn.parameter import Parameter
from nevula.nn.linear import Linear
from nevula.nn.losses import MSELoss
from nevula.optim.sgd import SGD
from nevula.optim.adam import Adam
from nevula.optim.adamw import AdamW


class TestOptim(unittest.TestCase):

    def test_zero_grad(self):
        """test_zero_grad resets parameter gradients to None"""
        weight = Parameter(Tensor([2.0]))
        weight.grad = Tensor([4.0])

        optimizer = SGD([weight], lr=0.1)
        optimizer.zero_grad()

        self.assertIsNone(weight.grad)

    def test_sgd_step(self):
        """Analytical SGD step: 2.0 - 0.1 * 4.0 = 1.6"""
        weight = Parameter(Tensor([2.0]))
        weight.grad = Tensor([4.0])

        optimizer = SGD([weight], lr=0.1)
        optimizer.step()

        self.assertAlmostEqual(weight.data[0], 1.6, places=5)

    def test_sgd_momentum(self):
        """SGD with momentum accelerates in the direction of persistent gradients"""
        weight = Parameter(Tensor([10.0]))
        optimizer = SGD([weight], lr=0.1, momentum=0.9)

        # Step 1: grad = 1.0, v1 = 1.0 -> w = 10 - 0.1 * 1.0 = 9.9
        weight.grad = Tensor([1.0])
        optimizer.step()
        self.assertAlmostEqual(weight.data[0], 9.9, places=5)

        # Step 2: grad = 1.0, v2 = 0.9 * 1.0 + 1.0 = 1.9 -> w = 9.9 - 0.1 * 1.9 = 9.71
        weight.grad = Tensor([1.0])
        optimizer.step()
        self.assertAlmostEqual(weight.data[0], 9.71, places=5)

    def test_adam_step(self):
        """Adam optimizer step updates parameter and maintains moment buffers"""
        weight = Parameter(Tensor([1.0, 2.0]))
        weight.grad = Tensor([0.5, -0.5])

        optimizer = Adam([weight], lr=0.01)
        optimizer.step()

        # Weight should move in the opposite direction of gradient
        self.assertTrue(weight.data[0] < 1.0)
        self.assertTrue(weight.data[1] > 2.0)

    def test_linear_regression_sgd(self):
        """Learning experiment: y = 2x with SGD optimizer"""
        # Data: y = 2x
        x = Tensor([[1.0], [2.0], [3.0], [4.0]])
        target = Tensor([[2.0], [4.0], [6.0], [8.0]])

        model = Linear(1, 1)
        loss_fn = MSELoss()
        optimizer = SGD(model.parameters(), lr=0.01)

        initial_loss = loss_fn(model(x), target).item()

        # Training loop
        for _ in range(600):
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()

        final_loss = loss_fn(model(x), target).item()

        # Loss should decrease significantly
        self.assertTrue(final_loss < initial_loss)
        self.assertTrue(final_loss < 0.05)

        # Learned weight should be close to 2.0, bias close to 0.0
        self.assertAlmostEqual(model.weight.item(), 2.0, places=1)
        self.assertAlmostEqual(model.bias.item(), 0.0, places=1)

    def test_linear_regression_adam(self):
        """Learning experiment: y = 2x with Adam optimizer"""
        x = Tensor([[1.0], [2.0], [3.0], [4.0]])
        target = Tensor([[2.0], [4.0], [6.0], [8.0]])

        model = Linear(1, 1)
        loss_fn = MSELoss()
        optimizer = Adam(model.parameters(), lr=0.05)

        initial_loss = loss_fn(model(x), target).item()

        for _ in range(500):
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()

        final_loss = loss_fn(model(x), target).item()

        self.assertTrue(final_loss < initial_loss)
        self.assertTrue(final_loss < 0.05)
        self.assertAlmostEqual(model.weight.item(), 2.0, places=1)


if __name__ == '__main__':
    unittest.main()
