import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Extract metrics and prepare for plotting
try:
    epochs = [10, 20, 30, 40, 50]
    training_losses = experiment_data["hyperparam_tuning_epochs"][
        "hydrogen_bond_experiment"
    ]["losses"]["train"]

    plt.figure()
    plt.plot(
        epochs[: len(training_losses)],
        training_losses,
        label="Training Loss",
        marker="o",
    )
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "hydrogen_bond_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()
