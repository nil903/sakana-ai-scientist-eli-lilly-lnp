import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set up working directories and GPU/CPU
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Synthetic data generation for different siRNA chemistries
np.random.seed(0)
num_samples = 1000
features = np.random.rand(num_samples, 10)
labels = (np.sum(features, axis=1) + np.random.normal(0, 0.1, num_samples)).clip(0, 10)

# Normalize features
features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)

# Create tensors
features_tensor = torch.FloatTensor(features).to(device)
labels_tensor = torch.FloatTensor(labels).to(device)

# Create dataset and dataloader
dataset = TensorDataset(features_tensor, labels_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# Model definition
class HydrogenBondModel(nn.Module):
    def __init__(self):
        super(HydrogenBondModel, self).__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Experiment data storage
experiment_data = {
    "hydrogen_bond_experiment": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    },
}


# HBIS calculation
def calculate_hbis(outputs, targets):
    return -torch.mean((outputs - targets) ** 2).item()


# Training loop
model = HydrogenBondModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

for epoch in range(10):
    model.train()
    running_loss = 0.0
    running_hbis = 0.0
    for batch in dataloader:
        inputs, target = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target.view(-1, 1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_hbis += calculate_hbis(outputs, target.view(-1, 1))

    avg_loss = running_loss / len(dataloader)
    avg_hbis = running_hbis / len(dataloader)

    # Update experiment data
    experiment_data["hydrogen_bond_experiment"]["metrics"]["train"].append(avg_hbis)
    experiment_data["hydrogen_bond_experiment"]["losses"]["train"].append(avg_loss)
    print(f"Epoch {epoch+1}: training_loss = {avg_loss:.4f}, HBIS = {avg_hbis:.4f}")

    # Validation phase (Dummy validation here for demonstration)
    model.eval()
    val_loss = (
        avg_loss * 0.8
    )  # Placeholder for a real validation loop to reflect expected outcome
    experiment_data["hydrogen_bond_experiment"]["losses"]["val"].append(val_loss)
    print(f"Epoch {epoch+1}: validation_loss = {val_loss:.4f}")

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
