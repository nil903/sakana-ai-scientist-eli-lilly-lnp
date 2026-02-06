import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot training and validation losses
try:
    train_losses = experiment_data["multiple_hyperparameter_settings"]["losses"][
        "train"
    ]
    val_losses = experiment_data["multiple_hyperparameter_settings"]["losses"]["val"]
    plt.figure()
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Training and Validation Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "Experiment_Loss_Plot.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plot training and validation metrics (HBD scores)
try:
    train_metrics = experiment_data["multiple_hyperparameter_settings"]["metrics"][
        "train"
    ]
    val_metrics = experiment_data["multiple_hyperparameter_settings"]["metrics"]["val"]
    plt.figure()
    plt.plot(train_metrics, label="Training HBD")
    plt.plot(val_metrics, label="Validation HBD")
    plt.title("Training and Validation HBD Scores")
    plt.xlabel("Epoch")
    plt.ylabel("HBD Score")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "Experiment_HBD_Plot.png"))
    plt.close()
except Exception as e:
    print(f"Error creating HBD metrics plot: {e}")
    plt.close()
