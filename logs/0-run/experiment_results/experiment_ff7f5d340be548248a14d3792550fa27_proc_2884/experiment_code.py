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

# Synthetic data generation with different noise levels
np.random.seed(0)
num_samples = 2000  # increased dataset size
features = np.random.rand(num_samples, 10)  # 10 features

noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2]
experiment_data = {"DATA_NOISE_LEVEL_ABLATION": {}}

for noise_level in noise_levels:
    labels = (
        np.sum(features, axis=1) + np.random.normal(0, noise_level, num_samples)
    ).clip(0, 10)

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

    # Store metrics for this noise level
    experiment_data["DATA_NOISE_LEVEL_ABLATION"][f"noise_{noise_level}"] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    # Training loop
    model = HydrogenBondModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

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

        avg_loss_train = running_loss_train / len(train_dataloader)
        experiment_data["DATA_NOISE_LEVEL_ABLATION"][f"noise_{noise_level}"]["losses"][
            "train"
        ].append(avg_loss_train)

        # Validation Phase
        model.eval()
        running_loss_val = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                inputs, target = [t.to(device) for t in batch]
                outputs = model(inputs)
                loss = criterion(outputs, target)
                running_loss_val += loss.item()

        avg_loss_val = running_loss_val / len(val_dataloader)
        experiment_data["DATA_NOISE_LEVEL_ABLATION"][f"noise_{noise_level}"]["losses"][
            "val"
        ].append(avg_loss_val)

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
