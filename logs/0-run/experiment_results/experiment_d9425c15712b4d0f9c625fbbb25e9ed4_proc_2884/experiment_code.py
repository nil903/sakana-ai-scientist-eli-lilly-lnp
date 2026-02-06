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


# Synthetic data generation functions
def generate_uniform_data(num_samples):
    features = np.random.rand(num_samples, 10)
    labels = (np.sum(features, axis=1) + np.random.normal(0, 0.1, num_samples)).clip(
        0, 10
    )
    return features, labels


def generate_gaussian_data(num_samples):
    features = np.random.normal(0, 1, (num_samples, 10))
    labels = (np.sum(features, axis=1) + np.random.normal(0, 0.1, num_samples)).clip(
        0, 10
    )
    return features, labels


def generate_exponential_data(num_samples):
    features = np.random.exponential(scale=1.0, size=(num_samples, 10))
    labels = (np.sum(features, axis=1) + np.random.normal(0, 0.1, num_samples)).clip(
        0, 10
    )
    return features, labels


# Model definition
class HydrogenBondModel(nn.Module):
    def __init__(self):
        super(HydrogenBondModel, self).__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


# Hydrogen Bonding Interaction Score (HBIS) function
def calculate_hbis(y_true, y_pred):
    return 1 - nn.MSELoss()(y_pred, y_true).item()


# Experiment data storage
experiment_data = {
    "MULTI_DISTRIBUTION_LEARNING": {
        "uniform": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "gaussian": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "exponential": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    }
}

weight_decay_values = [0.0, 1e-5, 1e-4, 1e-3]
num_samples = 2000  # dataset size
datasets = {
    "uniform": generate_uniform_data(num_samples),
    "gaussian": generate_gaussian_data(num_samples),
    "exponential": generate_exponential_data(num_samples),
}


def normalize(features):
    return (features - np.mean(features, axis=0)) / np.std(features, axis=0)


for dataset_name, (features, labels) in datasets.items():
    features_normalized = normalize(features)
    features_tensor = torch.FloatTensor(features_normalized).to(device)
    labels_tensor = torch.FloatTensor(labels).to(device).view(-1, 1)
    dataset_tensor = TensorDataset(features_tensor, labels_tensor)
    train_size = int(0.8 * len(dataset_tensor))
    val_size = len(dataset_tensor) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset_tensor, [train_size, val_size]
    )
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    for weight_decay in weight_decay_values:
        model = HydrogenBondModel().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)

        for epoch in range(50):
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
            experiment_data["MULTI_DISTRIBUTION_LEARNING"][dataset_name]["losses"][
                "train"
            ].append(avg_loss_train)
            experiment_data["MULTI_DISTRIBUTION_LEARNING"][dataset_name]["metrics"][
                "train"
            ].append(avg_hbis_train)
            print(
                f"[{dataset_name}] Weight Decay: {weight_decay}, Epoch {epoch + 1}: training_loss = {avg_loss_train:.4f}, HBIS = {avg_hbis_train:.4f}"
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
            experiment_data["MULTI_DISTRIBUTION_LEARNING"][dataset_name]["losses"][
                "val"
            ].append(avg_loss_val)
            experiment_data["MULTI_DISTRIBUTION_LEARNING"][dataset_name]["metrics"][
                "val"
            ].append(avg_hbis_val)
            print(
                f"[{dataset_name}] Weight Decay: {weight_decay}, Epoch {epoch + 1}: validation_loss = {avg_loss_val:.4f}, HBIS = {avg_hbis_val:.4f}"
            )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
