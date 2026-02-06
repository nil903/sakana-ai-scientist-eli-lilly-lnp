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

# Normalize the features
features = (features - features.mean(axis=0)) / features.std(axis=0)

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


# Model definition with increased complexity and dropout
class HydrogenBondModel(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super(HydrogenBondModel, self).__init__()
        self.fc1 = nn.Linear(10, 64)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


# Experiment data storage
experiment_data = {
    "dropout_ablation": {
        "with_dropout": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "without_dropout": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    },
}


# Hydrogen Bonding Interaction Density (HBD) function
def calculate_hbd(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


# Training loop for models with and without dropout
for usage in ["without_dropout", "with_dropout"]:
    dropout_rate = 0.3 if usage == "with_dropout" else 0.0
    model = HydrogenBondModel(dropout_rate).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(50):
        model.train()
        running_loss_train = 0.0
        hbd_train = 0.0

        for batch in train_dataloader:
            inputs, target = [t.to(device) for t in batch]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss_train += loss.item()
            hbd_train += calculate_hbd(
                target.cpu().numpy(), outputs.cpu().detach().numpy()
            )

        avg_loss_train = running_loss_train / len(train_dataloader)
        avg_hbd_train = hbd_train / len(train_dataloader)
        experiment_data["dropout_ablation"][usage]["losses"]["train"].append(
            avg_loss_train
        )
        experiment_data["dropout_ablation"][usage]["metrics"]["train"].append(
            avg_hbd_train
        )
        print(
            f"{usage.capitalize()}: Epoch {epoch+1}: training_loss = {avg_loss_train:.4f}, HBD = {avg_hbd_train:.4f}"
        )

        # Validation Phase
        model.eval()
        running_loss_val = 0.0
        hbd_val = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                inputs, target = [t.to(device) for t in batch]
                outputs = model(inputs)
                loss = criterion(outputs, target)
                running_loss_val += loss.item()
                hbd_val += calculate_hbd(
                    target.cpu().numpy(), outputs.cpu().detach().numpy()
                )

        avg_loss_val = running_loss_val / len(val_dataloader)
        avg_hbd_val = hbd_val / len(val_dataloader)
        experiment_data["dropout_ablation"][usage]["losses"]["val"].append(avg_loss_val)
        experiment_data["dropout_ablation"][usage]["metrics"]["val"].append(avg_hbd_val)
        print(
            f"{usage.capitalize()}: Epoch {epoch + 1}: validation_loss = {avg_loss_val:.4f}, HBD = {avg_hbd_val:.4f}"
        )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
