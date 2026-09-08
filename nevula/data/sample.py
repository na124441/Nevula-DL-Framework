import random
import nevula as nv
from nevula import Tensor
from nevula.nn import Sequential, Linear, ReLU, MSELoss
from nevula.optim import Adam
from nevula.data import TensorDataset, DataLoader, Compose, ToTensor, Normalize

print("==================================================================")
print("  🌌 NEVULA PHASE 5 — DATA ENGINE & MINI-BATCH PIPELINE")
print("==================================================================\n")

# 1. Generate Synthetic Dataset
# 128 samples, 10 features -> target is a linear combination + non-linear term
num_samples = 128
in_features = 10

raw_features = []
raw_targets = []
for _ in range(num_samples):
    row = [random.uniform(-1.0, 1.0) for _ in range(in_features)]
    target_val = sum(row[:5]) * 2.0 - sum(row[5:]) * 1.5
    raw_features.append(row)
    raw_targets.append([target_val])

features = Tensor(raw_features)
targets = Tensor(raw_targets)

print(f"Total Dataset Size: {len(features)} samples")
print(f"Feature Matrix Shape: {features.shape}")
print(f"Target Matrix Shape:  {targets.shape}\n")

# 2. Wrap in TensorDataset
dataset = TensorDataset(features, targets)
print(f"Dataset sample 0 features: {dataset[0][0].to_list()[:3]}... (showing first 3)")
print(f"Dataset sample 0 target:   {dataset[0][1].to_list()}\n")

# 3. Create DataLoader with Mini-Batching and Shuffling
batch_size = 32
train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=False,
)

print(f"DataLoader Config: batch_size={batch_size}, shuffle=True")
print(f"Total Batches per Epoch: {len(train_loader)} (128 / 32 = 4 batches)\n")

# 4. Neural Network Architecture
model = Sequential(
    Linear(in_features, 32),
    ReLU(),
    Linear(32, 1)
)

print("Model Architecture:")
print(model)
print()

# 5. Optimizer & Loss Function
optimizer = Adam(model.parameters(), lr=0.01)
criterion = MSELoss()

# 6. Mini-Batch Training Loop
print("--- Starting Mini-Batch Training Loop ---")
num_epochs = 15

for epoch in range(1, num_epochs + 1):
    epoch_loss = 0.0
    batch_count = 0

    for batch_idx, (batch_x, batch_y) in enumerate(train_loader, start=1):
        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass through model
        prediction = model(batch_x)

        # Compute MSE loss
        loss = criterion(prediction, batch_y)

        # Reverse-mode Autograd backpropagation
        loss.backward()

        # Update weights with Adam
        optimizer.step()

        epoch_loss += loss.item()
        batch_count += 1

    avg_loss = epoch_loss / batch_count
    if epoch % 3 == 0 or epoch == 1:
        print(f"  Epoch {epoch:2d}/{num_epochs:2d} | Average Loss: {avg_loss:.6f} | Batches Processed: {batch_count}")

print("\n==================================================================")
print("  🎉 Mini-batch data pipeline and training completed successfully!")
print("==================================================================")
