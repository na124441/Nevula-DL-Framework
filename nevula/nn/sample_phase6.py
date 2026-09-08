import os
import nevula as nv
from nevula import Tensor, no_grad, save_checkpoint, load_checkpoint
from nevula.nn import Sequential, Linear, ReLU, MSELoss, init
from nevula.nn.utils import clip_grad_norm_
from nevula.optim import Adam
from nevula.utils.diagnostics import print_gradient_diagnostics

print("==================================================================")
print("  🌌 NEVULA PHASE 6 — MODEL LIFECYCLE, UTILITIES & CHECKPOINTING")
print("==================================================================\n")

# 1. Build Model & Apply Kaiming / Xavier Initialization
model = Sequential(
    Linear(4, 16),
    ReLU(),
    Linear(16, 1)
)

# Apply Xavier uniform to first linear layer, Kaiming to second
init.xavier_uniform_(model[0].weight)
init.zeros_(model[0].bias)
init.kaiming_uniform_(model[2].weight, nonlinearity="linear")
init.zeros_(model[2].bias)

print("Initialized Model Architecture:")
print(model)
print()

# 2. Setup Optimizer & Loss Function
optimizer = Adam(model.parameters(), lr=0.01)
criterion = MSELoss()

# Dummy batch: 8 samples of 4 features
x = Tensor([
    [1.0, 2.0, 3.0, 4.0],
    [2.0, 3.0, 4.0, 5.0],
    [3.0, 4.0, 5.0, 6.0],
    [4.0, 5.0, 6.0, 7.0],
    [5.0, 6.0, 7.0, 8.0],
    [6.0, 7.0, 8.0, 9.0],
    [7.0, 8.0, 9.0, 10.0],
    [8.0, 9.0, 10.0, 11.0],
])
target = Tensor([[10.0], [14.0], [18.0], [22.0], [26.0], [30.0], [34.0], [38.0]])

# 3. Training Loop with Gradient Clipping
model.train()
print("--- Training Step with Gradient Clipping ---")
optimizer.zero_grad()
pred = model(x)
loss = criterion(pred, target)
loss.backward()

# Clip gradient norm to 1.0 to ensure numerical stability
grad_norm_before = clip_grad_norm_(model.parameters(), max_norm=1.0)
print(f"Total Gradient Norm before clipping: {grad_norm_before:.6f}")

# Print numerical diagnostics of gradients
print_gradient_diagnostics(model)

optimizer.step()
print(f"Loss after step: {loss.item():.6f}\n")

# 4. Inference Mode using no_grad()
model.eval()
test_x = Tensor([[10.0, 11.0, 12.0, 13.0]])

with no_grad():
    inference_output = model(test_x)

print(f"Inference output for test input: {inference_output.item():.4f}")
print(f"Graph tracking active during inference? {inference_output.grad_fn is not None} (Expected: False)\n")

# 5. Checkpointing: Saving & Restoring Model State
checkpoint_file = "nevula_checkpoint.ckpt"
print(f"--- Saving Checkpoint to '{checkpoint_file}' ---")
save_checkpoint(
    checkpoint_file,
    model=model,
    optimizer=optimizer,
    epoch=1,
    loss=loss.item()
)
print("Checkpoint saved successfully!")

print("\n--- Loading Checkpoint into Fresh Model ---")
fresh_model = Sequential(
    Linear(4, 16),
    ReLU(),
    Linear(16, 1)
)

fresh_optimizer = Adam(fresh_model.parameters(), lr=0.001)

loaded_data = load_checkpoint(checkpoint_file, model=fresh_model, optimizer=fresh_optimizer)
print(f"Restored Metadata: epoch={loaded_data['epoch']}, loss={loaded_data['loss']:.6f}")

with no_grad():
    fresh_output = fresh_model(test_x)

print(f"Restored Model output: {fresh_output.item():.4f}")
assert abs(inference_output.item() - fresh_output.item()) < 1e-5
print("Verification: Outputs between original and restored model match exactly!\n")

# Cleanup checkpoint file
if os.path.exists(checkpoint_file):
    os.remove(checkpoint_file)

print("==================================================================")
print("  🎉 Phase 6 Model Lifecycle & Utilities Completed Successfully!")
print("==================================================================")
