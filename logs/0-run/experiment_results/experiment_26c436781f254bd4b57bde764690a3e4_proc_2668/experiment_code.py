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
num_samples = 2000  # increased dataset size
features = np.random.rand(num_samples, 10)  # 10 features
labels = (np.sum(features, axis=1) + np.random.normal(0, 0.1, num_samples)).clip(0, 10)

# Create tensors
features_tensor = torch.FloatTensor(features).to(device)
labels_tensor = torch.FloatTensor(labels).to(device).view(-1, 1)

# Create dataset and dataloaders
dataset = TensorDataset(features_tensor, labels_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Model definition with increased complexity
class HydrogenBondModel(nn.Module):
    def __init__(self):
        super(HydrogenBondModel, self).__init__()
        self.fc1 = nn.Linear(10, 64)  # increased layer size
        self.fc2 = nn.Linear(64, 32)  # increased layer size
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)  # additional layer for complexity

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))  # added activation for additional layer
        return self.fc4(x)


# Experiment data storage
experiment_data = {
    "hydrogen_bond_experiment": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    },
}


# Hydrogen Bonding Interaction Score (HBIS) function
def calculate_hbis(y_true, y_pred):
    return 1 - nn.MSELoss()(y_pred, y_true).item()


# Hyperparameter tuning for weight decay
weight_decay_values = [0.0, 1e-5, 1e-4, 1e-3]

for weight_decay in weight_decay_values:
    model = HydrogenBondModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)

    # Training loop
    for epoch in range(50):  # increased epochs
        model.train()
        running_loss_train = 0.0
        hbis_train = 0.0

        for batch in train_dataloader:
            inputs, target = [t.to(device) for t in batch]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss_train += loss.item()
            hbis_train += calculate_hbis(target, outputs)

        avg_loss_train = running_loss_train / len(train_dataloader)
        avg_hbis_train = hbis_train / len(train_dataloader)
        experiment_data["hydrogen_bond_experiment"]["losses"]["train"].append(
            avg_loss_train
        )
        experiment_data["hydrogen_bond_experiment"]["metrics"]["train"].append(
            avg_hbis_train
        )
        print(
            f"Weight Decay: {weight_decay}, Epoch {epoch+1}: training_loss = {avg_loss_train:.4f}, HBIS = {avg_hbis_train:.4f}"
        )

        # Validation Phase
        model.eval()
        running_loss_val = 0.0
        hbis_val = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                inputs, target = [t.to(device) for t in batch]
                outputs = model(inputs)
                loss = criterion(outputs, target)
                running_loss_val += loss.item()
                hbis_val += calculate_hbis(target, outputs)

        avg_loss_val = running_loss_val / len(val_dataloader)
        avg_hbis_val = hbis_val / len(val_dataloader)
        experiment_data["hydrogen_bond_experiment"]["losses"]["val"].append(
            avg_loss_val
        )
        experiment_data["hydrogen_bond_experiment"]["metrics"]["val"].append(
            avg_hbis_val
        )
        print(
            f"Weight Decay: {weight_decay}, Epoch {epoch + 1}: validation_loss = {avg_loss_val:.4f}, HBIS = {avg_hbis_val:.4f}"
        )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
