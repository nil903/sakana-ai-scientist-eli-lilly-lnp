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
num_samples = 2000
features = np.random.rand(num_samples, 10)
labels = (np.sum(features, axis=1) + np.random.normal(0, 0.1, num_samples)).clip(0, 10)

# Normalize features
features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)

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


# Model definition
class HydrogenBondModel(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super(HydrogenBondModel, self).__init__()
        self.fc1 = nn.Linear(10, 64)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


# Experiment data storage
experiment_data = {
    "multiple_hyperparameter_settings": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predict": [],
        "ground_truth": [],
        "hbd": [],  # Hydrogen Bonding Density
    }
}

# Hyperparameter tuning
hyperparameter_configs = [
    {"learning_rate": 0.001, "weight_decay": 0.0, "batch_size": 32, "dropout": 0.0},
    {"learning_rate": 0.001, "weight_decay": 1e-5, "batch_size": 32, "dropout": 0.2},
    {"learning_rate": 0.0005, "weight_decay": 1e-4, "batch_size": 64, "dropout": 0.0},
    {"learning_rate": 0.0005, "weight_decay": 1e-3, "batch_size": 64, "dropout": 0.2},
]

for config in hyperparameter_configs:
    model = HydrogenBondModel(dropout_rate=config["dropout"]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    # Adjust DataLoader batch size for each configuration
    train_dataloader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False
    )

    # Training loop
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
            hbis_train += 1 - nn.MSELoss()(outputs, target).item()

        avg_loss_train = running_loss_train / len(train_dataloader)
        avg_hbis_train = hbis_train / len(train_dataloader)
        experiment_data["multiple_hyperparameter_settings"]["losses"]["train"].append(
            avg_loss_train
        )
        experiment_data["multiple_hyperparameter_settings"]["metrics"]["train"].append(
            avg_hbis_train
        )
        print(
            f"Config: {config}, Epoch {epoch+1}: training_loss = {avg_loss_train:.4f}, HBIS = {avg_hbis_train:.4f}"
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
                hbis_val += 1 - nn.MSELoss()(outputs, target).item()

        avg_loss_val = running_loss_val / len(val_dataloader)
        avg_hbis_val = hbis_val / len(val_dataloader)
        experiment_data["multiple_hyperparameter_settings"]["losses"]["val"].append(
            avg_loss_val
        )
        experiment_data["multiple_hyperparameter_settings"]["metrics"]["val"].append(
            avg_hbis_val
        )
        experiment_data["multiple_hyperparameter_settings"]["hbd"].append(
            avg_hbis_val
        )  # Track HBD score
        print(
            f"Config: {config}, Epoch {epoch + 1}: validation_loss = {avg_loss_val:.4f}, HBIS = {avg_hbis_val:.4f}"
        )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
