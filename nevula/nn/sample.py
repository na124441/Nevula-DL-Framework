import nevula as nv
from nevula import Tensor
from nevula.autograd import no_grad
from nevula.nn import Linear, ReLU, Sequential, MSELoss

print("==================================================")
print("  🌌 NEVULA PHASE 3 — NEURAL NETWORK FRAMEWORK")
print("==================================================\n")

# 1. Defining a multi-layer Perceptron (MLP)
model = Sequential(
    Linear(2, 4),
    ReLU(),
    Linear(4, 1)
)

print("Architecture:")
print(model)
print()

# 2. Inspecting model parameters
print("Registered Parameters:")
for name, param in model.named_parameters():
    print(f"  - {name}: shape={param.shape}")
print(f"Total Parameters: {len(model.parameters())}\n")

# 3. Input batch and target labels
# 2 samples, 2 features each
x = Tensor([
    [0.5, -1.0],
    [1.5,  2.0]
])

target = Tensor([
    [1.0],
    [0.0]
])

print("Input X:")
print(x)
print("\nTarget Y:")
print(target)
print()

# 4. Forward Pass
prediction = model(x)
print("Initial Prediction:")
print(prediction)
print()

# 5. Loss Computation
criterion = MSELoss()
loss = criterion(prediction, target)
print(f"Initial MSE Loss: {loss.item():.6f}\n")

# 6. Backward Pass (Automatic Differentiation through the entire network)
model.zero_grad()
loss.backward()

print("Computed Gradients:")
for name, param in model.named_parameters():
    print(f"\nGradient for {name} ({param.shape}):")
    print(param.grad)
print()

# 7. A Single Optimization Step under no_grad()
learning_rate = 0.05
with no_grad():
    for param in model.parameters():
        param.data = (param - learning_rate * param.grad).data

new_pred = model(x)
new_loss = criterion(new_pred, target)
print("==================================================")
print(f"After 1 Step (lr={learning_rate}):")
print(f"  - New Prediction:\n{new_pred}")
print(f"  - New MSE Loss: {new_loss.item():.6f}")
print("==================================================")
