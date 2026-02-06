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
    plt.plot(
        experiment_data["hydrogen_bond_experiment"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Loss Curves Over Epochs - Hydrogen Bond Experiment")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "hydrogen_bond_experiment_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

try:
    # Plot HBIS
    plt.figure()
    plt.plot(
        experiment_data["hydrogen_bond_experiment"]["metrics"]["train"],
        label="HBIS - Train",
    )
    plt.title("Training HBIS Over Epochs - Hydrogen Bond Experiment")
    plt.xlabel("Epochs")
    plt.ylabel("HBIS Value")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "hydrogen_bond_experiment_hbis_train.png"))
    plt.close()
except Exception as e:
    print(f"Error creating HBIS plot: {e}")
    plt.close()
