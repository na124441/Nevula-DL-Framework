import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from nevula.core.tensor import Tensor
from nevula.nn.parameter import Parameter
from nevula.nn.module import Module
from nevula.nn.container import Sequential
from nevula.nn.linear import Linear
from nevula.nn.activations import ReLU, Sigmoid, Tanh
from nevula.nn.losses import MSELoss, CrossEntropyLoss


class TestNN(unittest.TestCase):

    def test_parameter(self):
        """Parameter is a Tensor with requires_grad=True by default"""
        p = Parameter([1.0, 2.0, 3.0])
        self.assertTrue(isinstance(p, Tensor))
        self.assertTrue(p.requires_grad)
        self.assertEqual(p.shape, (3,))
        self.assertIn("Parameter containing:", repr(p))

    def test_module_parameter_registration(self):
        """Module automatically discovers and registers Parameter attributes"""
        class MyModule(Module):
            def __init__(self):
                super().__init__()
                self.w = Parameter([[1.0, 2.0], [3.0, 4.0]])
                self.b = Parameter([0.1, 0.2])

        m = MyModule()
        params = m.parameters()
        self.assertEqual(len(params), 2)
        self.assertIs(params[0], m.w)
        self.assertIs(params[1], m.b)

        named = dict(m.named_parameters())
        self.assertIn("w", named)
        self.assertIn("b", named)

    def test_module_zero_grad_and_train_eval(self):
        """Module zero_grad() and train()/eval() modes"""
        class SubMod(Module):
            def __init__(self):
                super().__init__()
                self.p = Parameter([1.0])

        class MainMod(Module):
            def __init__(self):
                super().__init__()
                self.sub = SubMod()

        m = MainMod()
        self.assertTrue(m.training)
        self.assertTrue(m.sub.training)

        m.eval()
        self.assertFalse(m.training)
        self.assertFalse(m.sub.training)

        m.train()
        self.assertTrue(m.training)
        self.assertTrue(m.sub.training)

        # zero_grad
        m.sub.p.grad = Tensor([5.0])
        m.zero_grad()
        self.assertIsNone(m.sub.p.grad)

    def test_linear_forward_backward(self):
        """Linear layer forward pass shape and backward gradients"""
        layer = Linear(3, 2, bias=True)
        self.assertEqual(layer.weight.shape, (3, 2))
        self.assertEqual(layer.bias.shape, (2,))

        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)  # (2, 3)
        y = layer(x)
        self.assertEqual(y.shape, (2, 2))

        # Backward
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(layer.weight.grad)
        self.assertEqual(layer.weight.grad.shape, (3, 2))
        self.assertIsNotNone(layer.bias.grad)
        self.assertEqual(layer.bias.grad.shape, (2,))
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, (2, 3))

    def test_activations(self):
        """ReLU, Sigmoid, and Tanh module wrappers"""
        x = Tensor([-2.0, 0.0, 2.0], requires_grad=True)
        relu = ReLU()
        y = relu(x)
        self.assertEqual(y.to_list(), [0.0, 0.0, 2.0])

        sig = Sigmoid()
        s = sig(x)
        self.assertEqual(s.shape, (3,))

        tanh = Tanh()
        t = tanh(x)
        self.assertEqual(t.shape, (3,))

    def test_sequential_container(self):
        """Sequential chains layers and discovers all nested parameters"""
        model = Sequential(
            Linear(2, 4),
            ReLU(),
            Linear(4, 1)
        )
        self.assertEqual(len(model), 3)
        params = model.parameters()
        # Linear1 weight, bias + Linear2 weight, bias = 4 parameters
        self.assertEqual(len(params), 4)

        x = Tensor([[1.0, 2.0], [3.0, 4.0]])
        out = model(x)
        self.assertEqual(out.shape, (2, 1))

    def test_mse_loss(self):
        """MSELoss calculates mean squared error and backpropagates"""
        pred = Tensor([[2.0], [4.0]], requires_grad=True)
        target = Tensor([[1.0], [2.0]])
        criterion = MSELoss()

        loss = criterion(pred, target)
        # diff = [1.0, 2.0], sq = [1.0, 4.0], mean = 2.5
        self.assertAlmostEqual(loss.item(), 2.5, places=5)

        loss.backward()
        # d/d_pred = 2 * (pred - target) / N = 2 * [1.0, 2.0] / 2 = [1.0, 2.0]
        self.assertIsNotNone(pred.grad)
        self.assertEqual(pred.grad.shape, (2, 1))
        self.assertAlmostEqual(pred.grad[0, 0], 1.0, places=5)
        self.assertAlmostEqual(pred.grad[1, 0], 2.0, places=5)

    def test_cross_entropy_loss(self):
        """CrossEntropyLoss computes stable loss and backpropagates"""
        logits = Tensor([[2.0, 1.0, 0.1]], requires_grad=True)
        target = Tensor([0])
        criterion = CrossEntropyLoss()

        loss = criterion(logits, target)
        self.assertTrue(loss.item() > 0)

        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertEqual(logits.grad.shape, (1, 3))

    def test_end_to_end_mlp(self):
        """The Phase 3 Finish Line: full MLP forward, loss, backward, and parameter gradients"""
        model = Sequential(
            Linear(2, 4),
            ReLU(),
            Linear(4, 1)
        )

        x = Tensor([[0.5, -1.0], [1.5, 2.0]])
        target = Tensor([[1.0], [0.0]])

        prediction = model(x)
        loss_fn = MSELoss()
        loss = loss_fn(prediction, target)

        model.zero_grad()
        loss.backward()

        # Check that every single parameter has valid gradients populated
        params = model.parameters()
        self.assertEqual(len(params), 4)
        for i, p in enumerate(params):
            self.assertIsNotNone(p.grad, f"Parameter {i} missing gradient")
            self.assertEqual(p.shape, p.grad.shape)


if __name__ == '__main__':
    unittest.main()
