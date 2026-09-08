import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from nevula.core.tensor import Tensor
from nevula.nn.parameter import Parameter
from nevula.nn.linear import Linear
from nevula.nn.container import Sequential
from nevula.nn.activations import ReLU
from nevula.optim.adam import Adam
from nevula.nn import init
from nevula.nn.utils import clip_grad_norm_
from nevula.utils.serialization import save, load, save_checkpoint, load_checkpoint
from nevula.utils.diagnostics import check_for_nan, check_for_inf, inspect_gradients


class TestPhase6(unittest.TestCase):

    def test_init_functions(self):
        """Initializers fill tensor data properly"""
        t = Tensor.zeros((4, 4))
        init.constant_(t, 7.5)
        self.assertEqual(t.data, [7.5] * 16)

        init.zeros_(t)
        self.assertEqual(t.data, [0.0] * 16)

        init.ones_(t)
        self.assertEqual(t.data, [1.0] * 16)

        init.uniform_(t, a=-2.0, b=2.0)
        self.assertTrue(all(-2.0 <= x <= 2.0 for x in t.data))

        init.xavier_uniform_(t)
        # For 4x4 matrix: fan_in=4, fan_out=4 -> std = sqrt(2/8) = 0.5, bound = sqrt(3)*0.5 ~ 0.866
        bound = math.sqrt(3.0 * (2.0 / 8.0))
        self.assertTrue(all(-bound <= x <= bound for x in t.data))

        init.kaiming_uniform_(t, nonlinearity="relu")
        self.assertEqual(len(t.data), 16)

    def test_detach(self):
        """detach() severs connection to the autograd computation graph"""
        x = Tensor([2.0, 3.0], requires_grad=True)
        y = x * 2.0
        self.assertTrue(y.requires_grad)

        z = y.detach()
        self.assertFalse(z.requires_grad)
        self.assertIsNone(z.grad_fn)

        # Operations on z do not backpropagate to x
        out = (z * 5.0).sum()
        out.backward()

        self.assertIsNone(x.grad)

    def test_clip_grad_norm(self):
        """clip_grad_norm_ scales down gradients when total norm exceeds max_norm"""
        w = Parameter(Tensor([3.0, 4.0]))
        w.grad = Tensor([3.0, 4.0])  # Norm = sqrt(9 + 16) = 5.0

        total_norm = clip_grad_norm_([w], max_norm=2.5)
        self.assertAlmostEqual(total_norm, 5.0, places=5)

        # Gradients should be halved: 3.0 * (2.5/5.0) = 1.5, 4.0 * (2.5/5.0) = 2.0
        self.assertAlmostEqual(w.grad.data[0], 1.5, places=5)
        self.assertAlmostEqual(w.grad.data[1], 2.0, places=5)

    def test_state_dict_and_load_state_dict(self):
        """Model state_dict captures parameters and load_state_dict restores them"""
        model_a = Sequential(Linear(2, 4), ReLU(), Linear(4, 1))
        # Mutate weights of model_a
        for p in model_a.parameters():
            for i in range(len(p.data)):
                p.data[i] = 42.0

        state = model_a.state_dict()
        self.assertIn("0.weight", state)
        self.assertIn("0.bias", state)
        self.assertIn("2.weight", state)
        self.assertIn("2.bias", state)

        model_b = Sequential(Linear(2, 4), ReLU(), Linear(4, 1))
        model_b.load_state_dict(state)

        # Verify model_b now has identical parameter values
        for p in model_b.parameters():
            self.assertTrue(all(val == 42.0 for val in p.data))

        # Forward output of both models must match exactly
        inp = Tensor([[1.0, 2.0]])
        out_a = model_a(inp)
        out_b = model_b(inp)
        self.assertAlmostEqual(out_a.item(), out_b.item(), places=5)

    def test_checkpointing(self):
        """save_checkpoint and load_checkpoint full round-trip"""
        model = Sequential(Linear(3, 2), Linear(2, 1))
        optimizer = Adam(model.parameters(), lr=0.02)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "model.ckpt")
            save_checkpoint(ckpt_path, model=model, optimizer=optimizer, epoch=10, accuracy=0.95)
            self.assertTrue(os.path.exists(ckpt_path))

            fresh_model = Sequential(Linear(3, 2), Linear(2, 1))
            fresh_opt = Adam(fresh_model.parameters(), lr=0.001)

            ckpt_loaded = load_checkpoint(ckpt_path, model=fresh_model, optimizer=fresh_opt)
            self.assertEqual(ckpt_loaded["epoch"], 10)
            self.assertEqual(ckpt_loaded["accuracy"], 0.95)
            self.assertAlmostEqual(fresh_opt.lr, 0.02, places=5)

            # Check that weights were restored
            inp = Tensor([[1.0, 2.0, 3.0]])
            self.assertAlmostEqual(model(inp).item(), fresh_model(inp).item(), places=5)

    def test_diagnostics(self):
        """NaN and Inf detection and gradient inspection"""
        normal_t = Tensor([1.0, 2.0, 3.0])
        nan_t = Tensor([1.0, float("nan"), 3.0])
        inf_t = Tensor([1.0, float("inf"), 3.0])

        self.assertFalse(check_for_nan(normal_t))
        self.assertTrue(check_for_nan(nan_t))

        self.assertFalse(check_for_inf(normal_t))
        self.assertTrue(check_for_inf(inf_t))

        # Test gradient inspection
        m = Linear(2, 1)
        m.weight.grad = Tensor([0.1, 0.2], shape=(2, 1))
        m.bias.grad = Tensor([0.05])

        stats = inspect_gradients(m)
        self.assertIn("weight", stats)
        self.assertTrue(stats["weight"]["has_grad"])
        self.assertFalse(stats["weight"]["has_nan"])
        self.assertFalse(stats["weight"]["has_inf"])
        self.assertTrue(stats["weight"]["norm"] > 0)


if __name__ == '__main__':
    unittest.main()
