"""
Nevula DL Framework - Phase 7 Backend & Device System Demonstration
Demonstrates device allocation, backend routing, device validation, and model migration.
"""

from nevula.backend.device import Device, cuda_available
from nevula.backend.registry import get_backend
from nevula.core.tensor import Tensor
from nevula.nn.linear import Linear
from nevula.nn.activations import ReLU
from nevula.nn.container import Sequential


def main():
    print("=================================================================")
    print("      🚀 Nevula DL Framework — Phase 7: Device & Backend         ")
    print("=================================================================\n")

    # 1. Device Inspection
    print("1. Compute Devices:")
    cpu_dev = Device("cpu")
    cuda_dev = Device("cuda")
    print(f"   CPU Device   : {cpu_dev}")
    print(f"   CUDA Device  : {cuda_dev}")
    print(f"   CUDA Avail?  : {cuda_available()}\n")

    # 2. Tensor Allocation on CPU
    print("2. CPU Tensor Allocation & Backend Arithmetic:")
    a = Tensor([1.0, 2.0, 3.0], device="cpu")
    b = Tensor([4.0, 5.0, 6.0], device="cpu")
    c = a + b
    print(f"   a = {a}")
    print(f"   b = {b}")
    print(f"   c = a + b -> {c}")
    print(f"   c.data storage: {type(c.data).__name__}, list: {c.data.tolist()}\n")

    # 3. Matrix Multiplication on CPU
    print("3. High-Performance Matrix Multiplication:")
    m1 = Tensor([[1.0, 2.0], [3.0, 4.0]], device="cpu")
    m2 = Tensor([[5.0, 6.0], [7.0, 8.0]], device="cpu")
    out_mm = m1 @ m2
    print(f"   m1 @ m2 ->\n{out_mm}\n")

    # 4. Device Mismatch Enforcement
    print("4. Device Mismatch Safety:")
    tensor_cpu = Tensor([1.0, 2.0, 3.0], device="cpu")
    tensor_cuda = Tensor([1.0, 2.0, 3.0], device="cuda")
    print(f"   Tensor on CPU : {tensor_cpu}")
    print(f"   Tensor on CUDA: {tensor_cuda}")
    try:
        _ = tensor_cpu + tensor_cuda
    except RuntimeError as e:
        print(f"   Caught Expected Safety Error: {e}\n")

    # 5. Tensor Migration with .to()
    print("5. Tensor Device Migration (.to()):")
    t_cpu = Tensor([10.0, 20.0, 30.0], device="cpu")
    print(f"   Original CPU Tensor: {t_cpu}")
    t_cuda = t_cpu.to("cuda")
    print(f"   Migrated to CUDA   : {t_cuda}")
    t_back = t_cuda.to("cpu")
    print(f"   Migrated Back to CPU: {t_back}\n")

    # 6. Neural Network Model Migration
    print("6. Neural Network Model Migration (Module.to()):")
    model = Sequential(
        Linear(4, 8),
        ReLU(),
        Linear(8, 2),
    )
    print(f"   Model created. First layer device: {model[0].weight.device}")
    
    # Forward on CPU
    x = Tensor([[1.0, 2.0, 3.0, 4.0]], device="cpu")
    out = model(x)
    print(f"   Forward output on CPU: {out}")

    # Move model to CUDA
    model.to("cuda")
    print(f"   Model moved to CUDA. First layer device: {model[0].weight.device}")

    try:
        # Should raise error because input is still on CPU
        _ = model(x)
    except RuntimeError as e:
        print(f"   Caught Expected Input/Model Mismatch: {e}")

    # Move back to CPU
    model.to("cpu")
    print(f"   Model moved back to CPU. First layer device: {model[0].weight.device}")
    out2 = model(x)
    print(f"   Forward output after return to CPU: {out2}\n")

    # 7. Autograd with Backend
    print("7. Autograd Engine via Backend:")
    param = Tensor([2.0, 3.0, 4.0], requires_grad=True, device="cpu")
    loss = (param ** 2).sum()
    loss.backward()
    print(f"   Loss: {loss.item()}")
    print(f"   Gradients (d(x^2)/dx = 2x): {param.grad}")
    print("\n✅ Phase 7 Device & Backend System successfully demonstrated!")


if __name__ == "__main__":
    main()
