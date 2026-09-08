import nevula as nv
from nevula import Tensor
from nevula.nn import Linear, MSELoss
from nevula.optim import SGD, Adam

print("==================================================================")
print("  🌌 NEVULA PHASE 4 — OPTIMIZERS & LEARNING ENGINE")
print("==================================================================\n")

print("--- Experiment: Learning y = 2x with SGD ---")

# 1. Dataset: y = 2x
x = Tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0]
])

y = Tensor([
    [2.0],
    [4.0],
    [6.0],
    [8.0]
])

# 2. Model: Single Linear Neuron (y = w*x + b)
model = Linear(1, 1)
print(f"Initial Weight: {model.weight.item():.4f}")
print(f"Initial Bias:   {model.bias.item():.4f}\n")

# 3. Loss & Optimizer
loss_fn = MSELoss()
optimizer = SGD(model.parameters(), lr=0.01)

# 4. Training Loop
print("Training progress (SGD):")
for epoch in range(1, 601):
    # a. Forward pass
    prediction = model(x)

    # b. Compute loss
    loss = loss_fn(prediction, y)

    # c. Zero out accumulated gradients from previous iteration
    optimizer.zero_grad()

    # d. Backward pass: compute dL/dw and dL/db
    loss.backward()

    # e. Step: update parameters w = w - lr * grad
    optimizer.step()

    if epoch % 100 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d} | Loss: {loss.item():.6f} | Weight: {model.weight.item():.4f} | Bias: {model.bias.item():.4f}")

print("\nResult after 600 epochs:")
print(f"  Target: y = 2.0 * x + 0.0")
print(f"  Learned: y = {model.weight.item():.4f} * x + {model.bias.item():.4f}")

# Test on unseen input x = 10.0 (expected y = 20.0)
test_x = Tensor([[10.0]])
test_pred = model(test_x)
print(f"  Prediction for x = 10.0: {test_pred.item():.2f} (Expected: 20.0)\n")

print("------------------------------------------------------------------")
print("--- Experiment: Learning y = 2x with Adam ---")
model_adam = Linear(1, 1)
adam_opt = Adam(model_adam.parameters(), lr=0.05)

for epoch in range(1, 501):
    adam_opt.zero_grad()
    pred = model_adam(x)
    loss = loss_fn(pred, y)
    loss.backward()
    adam_opt.step()

    if epoch % 100 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d} | Loss: {loss.item():.6f} | Weight: {model_adam.weight.item():.4f} | Bias: {model_adam.bias.item():.4f}")

print(f"\nAdam Final Result: y = {model_adam.weight.item():.4f} * x + {model_adam.bias.item():.4f}")
print("==================================================================")
