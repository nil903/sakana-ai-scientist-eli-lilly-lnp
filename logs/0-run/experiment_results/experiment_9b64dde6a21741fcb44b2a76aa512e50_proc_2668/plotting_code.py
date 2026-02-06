import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

try:
    # Plot training loss
    plt.figure()
    plt.plot(
        experiment_data["hydrogen_bond_experiment"]["losses"]["train"],
        label="Training Loss",
    )
    plt.title("Training Loss Over Epochs - Hydrogen Bond Experiment")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "hydrogen_bond_experiment_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

try:
    # Plot training metrics
    plt.figure()
    plt.plot(
        experiment_data["hydrogen_bond_experiment"]["metrics"]["train"],
        label="Training Metric",
    )
    plt.title("Training Metric Over Epochs - Hydrogen Bond Experiment")
    plt.xlabel("Epochs")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "hydrogen_bond_experiment_training_metric.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating training metric plot: {e}")
    plt.close()

try:
    # Check for validation losses and plot
    if "val" in experiment_data["hydrogen_bond_experiment"]["losses"]:
        plt.figure()
        plt.plot(
            experiment_data["hydrogen_bond_experiment"]["losses"]["val"],
            label="Validation Loss",
        )
        plt.title("Validation Loss Over Epochs - Hydrogen Bond Experiment")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, "hydrogen_bond_experiment_validation_loss.png")
        )
        plt.close()
except Exception as e:
    print(f"Error creating validation loss plot: {e}")
    plt.close()
