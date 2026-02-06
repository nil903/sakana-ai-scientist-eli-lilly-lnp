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


# Experiment data storage
experiment_data = {
    "hyperparam_tuning_type_1": {
        "learning_rate_tuning": {
            "metrics": {"train": []},
            "losses": {"train": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# Hyperparameter tuning for learning rates
learning_rates = [0.01, 0.001, 0.0005, 0.0001]

for lr in learning_rates:
    # Initialize model, loss function, and optimizer for each learning rate
    model = HydrogenBondModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(10):  # Limited to 10 epochs for preliminary testing
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
        experiment_data["hyperparam_tuning_type_1"]["learning_rate_tuning"]["metrics"][
            "train"
        ].append(avg_loss)
        experiment_data["hyperparam_tuning_type_1"]["learning_rate_tuning"]["losses"][
            "train"
        ].append(avg_loss)
        print(f"Learning rate: {lr}, Epoch {epoch+1}: training_loss = {avg_loss:.4f}")

    # Save predictions and ground truth after training
    with torch.no_grad():
        model.eval()
        predictions = model(features_tensor).cpu().numpy()
        experiment_data["hyperparam_tuning_type_1"]["learning_rate_tuning"][
            "predictions"
        ].append(predictions)
        experiment_data["hyperparam_tuning_type_1"]["learning_rate_tuning"][
            "ground_truth"
        ].append(labels)

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
