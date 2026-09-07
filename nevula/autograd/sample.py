import nevula as nv
from nevula.autograd import render_graph, no_grad

print("=== 1. Basic Scalar Automatic Differentiation ===")
# f(x) = x^2 + 3x + 5 => df/dx = 2x + 3
# At x = 2.0: df/dx = 2(2) + 3 = 7.0
x = nv.tensor(2.0, requires_grad=True)
y = x**2 + 3 * x + 5
print(f"Forward output: {y}")
print(f"Graph root: {y.grad_fn}")

y.backward()
print(f"Analytical gradient dx = {x.grad} (Expected: 7.0)\n")

print("=== 2. Multi-Path Branching (DAG Accumulation) ===")
# z = a * b + a => dz/da = b + 1, dz/db = a
a = nv.tensor(3.0, requires_grad=True)
b = nv.tensor(4.0, requires_grad=True)
z = a * b + a
z.backward()
print(f"z = a * b + a -> a.grad = {a.grad} (Expected: 5.0), b.grad = {b.grad} (Expected: 3.0)\n")

print("=== 3. Computational Graph Visualization ===")
print("Rendered Graph for z = a * b + a:")
print(render_graph(z))
print()

print("=== 4. Matrix Multiplication Autodiff ===")
# C = A @ B
# A: (2, 3), B: (3, 2)
A = nv.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
B = nv.Tensor([[7.0, 8.0], [9.0, 1.0], [2.0, 3.0]], requires_grad=True)
C = A @ B
loss = C.sum()
loss.backward()
print(f"A.grad:\n{A.grad}")
print(f"B.grad:\n{B.grad}\n")

print("=== 5. Broadcast Reduction Gradient ===")
# Y = X + bias
# X: (2, 2), bias: (2,)
X = nv.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
bias = nv.Tensor([10.0, 20.0], requires_grad=True)
out = (X + bias).sum()
out.backward()
print(f"X.grad (all 1s):\n{X.grad}")
print(f"bias.grad (summed along batch dim, expect [2.0, 2.0]):\n{bias.grad}\n")

print("=== 6. Mean Squared Error (MSE) Optimization Step ===")
# w * x - target
w = nv.tensor(2.0, requires_grad=True)
x_in = nv.tensor(3.0)
target = nv.tensor(10.0)

pred = w * x_in
mse = (pred - target) ** 2
print(f"Initial Prediction: {pred.to_list()}, Target: {target.to_list()}, Loss: {mse.to_list()}")
mse.backward()
print(f"w.grad: {w.grad}")

# Optimizer update under no_grad()
with no_grad():
    lr = 0.01
    w = w - lr * w.grad
    w.requires_grad = True

new_pred = w * x_in
new_loss = (new_pred - target) ** 2
print(f"Updated w: {w.to_list()}, New Loss: {new_loss.to_list()}\n")

print("=== 7. Non-linear Activations (ReLU, Sigmoid, Tanh) ===")
act_in = nv.Tensor([-2.0, -0.5, 0.5, 2.0], requires_grad=True)
act_out = act_in.relu().sum()
act_out.backward()
print(f"ReLU gradient on [-2, -0.5, 0.5, 2]: {act_in.grad.to_list()} (Expected: [0.0, 0.0, 1.0, 1.0])")
