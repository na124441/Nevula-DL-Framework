import unittest
from nevula.backend.device import Device, cuda_available
from nevula.backend.registry import get_backend
from nevula.backend.cpu.backend import CPUBackend
from nevula.backend.cuda.backend import CUDABackend
from nevula.core.tensor import Tensor
from nevula.nn.linear import Linear
from nevula.nn.activations import ReLU
from nevula.nn.container import Sequential


class TestBackend(unittest.TestCase):
    def test_device_creation_and_equality(self):
        d_cpu = Device("cpu")
        self.assertEqual(d_cpu.type, "cpu")
        self.assertEqual(repr(d_cpu), "Device('cpu')")
        self.assertEqual(str(d_cpu), "cpu")
        self.assertEqual(d_cpu, "cpu")
        self.assertEqual(d_cpu, Device("cpu"))

        d_cuda = Device("cuda")
        self.assertEqual(d_cuda.type, "cuda")
        self.assertEqual(d_cuda, "cuda")
        self.assertEqual(d_cuda, Device("cuda:0"))

        with self.assertRaises(ValueError):
            _ = Device("tpu")

        with self.assertRaises(TypeError):
            _ = Device(1234)

    def test_cuda_available_utility(self):
        is_avail = cuda_available()
        self.assertIsInstance(is_avail, bool)

    def test_backend_registry(self):
        cpu_b = get_backend("cpu")
        self.assertIsInstance(cpu_b, CPUBackend)

        cuda_b = get_backend("cuda")
        self.assertIsInstance(cuda_b, CUDABackend)

        with self.assertRaises(RuntimeError):
            _ = get_backend("unsupported_backend_device")

    def test_tensor_device_cpu(self):
        x = Tensor([1.0, 2.0, 3.0], device="cpu")
        self.assertEqual(x.device.type, "cpu")
        self.assertEqual(x.device, Device("cpu"))
        self.assertEqual(x.data.tolist(), [1.0, 2.0, 3.0])

    def test_tensor_math_cpu_backend(self):
        a = Tensor([1.0, 2.0, 3.0], device="cpu")
        b = Tensor([4.0, 5.0, 6.0], device="cpu")

        # Elementwise addition
        c = a + b
        self.assertEqual(c.device.type, "cpu")
        self.assertEqual(c.data.tolist(), [5, 7, 9])
        self.assertEqual(c.to_list(), [5.0, 7.0, 9.0])

        # Subtraction, multiplication, division
        sub = b - a
        self.assertEqual(sub.to_list(), [3.0, 3.0, 3.0])

        mul = a * b
        self.assertEqual(mul.to_list(), [4.0, 10.0, 18.0])

        div = b / a
        self.assertEqual(div.to_list(), [4.0, 2.5, 2.0])

        # Matrix multiplication
        m1 = Tensor([[1.0, 2.0], [3.0, 4.0]], device="cpu")
        m2 = Tensor([[2.0, 0.0], [1.0, 2.0]], device="cpu")
        res_mm = m1 @ m2
        self.assertEqual(res_mm.shape, (2, 2))
        self.assertEqual(res_mm.to_list(), [[4.0, 4.0], [10.0, 8.0]])

        # Sum and mean
        s = m1.sum()
        self.assertEqual(s.shape, ())
        self.assertEqual(s.item(), 10.0)

        m = m1.mean()
        self.assertEqual(m.shape, ())
        self.assertEqual(m.item(), 2.5)

    def test_device_mismatch(self):
        a = Tensor([1.0, 2.0, 3.0], device="cpu")
        b = Tensor([4.0, 5.0, 6.0], device="cuda")

        with self.assertRaises(RuntimeError) as ctx:
            _ = a + b
        self.assertIn("Expected all tensors to be on the same device, but found cpu and cuda.", str(ctx.exception))

        with self.assertRaises(RuntimeError) as ctx:
            _ = a - b
        self.assertIn("Expected all tensors to be on the same device, but found cpu and cuda.", str(ctx.exception))

        with self.assertRaises(RuntimeError) as ctx:
            _ = a @ b
        self.assertIn("Expected all tensors to be on the same device, but found cpu and cuda.", str(ctx.exception))

    def test_tensor_to_device(self):
        x = Tensor([1.0, 2.0, 3.0], device="cpu")
        same_x = x.to("cpu")
        self.assertIs(same_x, x)

        x_cuda = x.to("cuda")
        self.assertEqual(x_cuda.device.type, "cuda")

        x_back = x_cuda.to("cpu")
        self.assertEqual(x_back.device.type, "cpu")
        self.assertEqual(x_back.to_list(), [1.0, 2.0, 3.0])

    def test_module_to_device(self):
        model = Sequential(
            Linear(4, 8),
            ReLU(),
            Linear(8, 2),
        )

        # Initially all parameters on CPU
        for p in model.parameters():
            self.assertEqual(p.device.type, "cpu")

        # Forward pass on CPU
        x = Tensor([[1.0, 2.0, 3.0, 4.0]], device="cpu")
        out = model(x)
        self.assertEqual(out.device.type, "cpu")
        self.assertEqual(out.shape, (1, 2))

        # Move model to CUDA
        model.to("cuda")
        for p in model.parameters():
            self.assertEqual(p.device.type, "cuda")

        # Passing CPU input to CUDA model must raise device mismatch
        with self.assertRaises(RuntimeError) as ctx:
            _ = model(x)
        self.assertIn("Expected all tensors to be on the same device", str(ctx.exception))

        # Move model back to CPU
        model.to("cpu")
        for p in model.parameters():
            self.assertEqual(p.device.type, "cpu")
        out2 = model(x)
        self.assertEqual(out2.device.type, "cpu")

    def test_autograd_with_cpu_backend(self):
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True, device="cpu")
        y = (x ** 2).sum()
        y.backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.device.type, "cpu")
        self.assertEqual(x.grad.to_list(), [[2.0, 4.0], [6.0, 8.0]])


if __name__ == "__main__":
    unittest.main()
