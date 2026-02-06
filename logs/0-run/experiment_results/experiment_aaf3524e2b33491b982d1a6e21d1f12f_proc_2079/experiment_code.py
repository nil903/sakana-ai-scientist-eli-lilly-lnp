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
features = np.random.rand(num_samples, 10)
labels = (np.sum(features, axis=1) + np.random.normal(0, 0.1, num_samples)).clip(0, 10)

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
    "weight_decay_tuning": {
        "hydrogen_bond_experiment": {
            "metrics": {"train": []},
            "losses": {"train": []},
            "predictions": [],
            "ground_truth": [],
        },
    }
}

# Hyperparameter tuning for weight decay
weight_decay_values = [
    0.0,
    1e-5,
    1e-4,
    1e-3,
]  # Different weight decay values for tuning

for weight_decay in weight_decay_values:
    model = HydrogenBondModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)

    # Training loop
    for epoch in range(10):
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
        experiment_data["weight_decay_tuning"]["hydrogen_bond_experiment"]["metrics"][
            "train"
        ].append(avg_loss)
        experiment_data["weight_decay_tuning"]["hydrogen_bond_experiment"]["losses"][
            "train"
        ].append(avg_loss)
        print(
            f"Weight Decay: {weight_decay}, Epoch {epoch+1}: training_loss = {avg_loss:.4f}"
        )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
