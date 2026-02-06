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

# Synthetic data generation
np.random.seed(0)
num_samples = 1000
features = np.random.rand(num_samples, 10)  # Random features
labels = (np.sum(features, axis=1) + np.random.normal(0, 0.1, num_samples)).clip(
    0, 10
)  # Simulated hydrogen bonds

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


# Initialize model, loss function, and optimizer
model = HydrogenBondModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Experiment data storage
experiment_data = {
    "hyperparam_tuning_epochs": {
        "hydrogen_bond_experiment": {
            "metrics": {"train": []},
            "losses": {"train": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# Define a range for epochs to tune
epoch_range = [10, 20, 30, 40, 50]

# Training loop over different numbers of epochs
for num_epochs in epoch_range:
    print(f"Training for {num_epochs} epochs")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in dataloader:
            inputs, target = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, target.view(-1, 1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        experiment_data["hyperparam_tuning_epochs"]["hydrogen_bond_experiment"][
            "metrics"
        ]["train"].append(avg_loss)
        experiment_data["hyperparam_tuning_epochs"]["hydrogen_bond_experiment"][
            "losses"
        ]["train"].append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}: training_loss = {avg_loss:.4f}")

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
