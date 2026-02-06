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
    # Plot training and validation loss
    plt.figure()
    plt.plot(
        experiment_data["hydrogen_bond_experiment"]["losses"]["train"],
        label="Training Loss",
    )
    plt.plot(
        experiment_data["hydrogen_bond_experiment"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "hydrogen_bond_experiment_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

try:
    # Plot training and validation metrics (HBIS)
    plt.figure()
    plt.plot(
        experiment_data["hydrogen_bond_experiment"]["metrics"]["train"],
        label="Training HBIS",
    )
    plt.plot(
        experiment_data["hydrogen_bond_experiment"]["metrics"]["val"],
        label="Validation HBIS",
    )
    plt.title("HBIS Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("HBIS Score")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "hydrogen_bond_experiment_hbis.png"))
    plt.close()
except Exception as e:
    print(f"Error creating HBIS plot: {e}")
    plt.close()
