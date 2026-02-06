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
    plt.title("Training Loss Over Epochs: Hydrogen Bond Experiment")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "hydrogen_bond_experiment_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

try:
    # Plot validation loss
    plt.figure()
    plt.plot(
        experiment_data["hydrogen_bond_experiment"]["losses"]["val"],
        label="Validation Loss",
        color="orange",
    )
    plt.title("Validation Loss Over Epochs: Hydrogen Bond Experiment")
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

try:
    # Plot training metrics (HBIS)
    plt.figure()
    plt.plot(
        experiment_data["hydrogen_bond_experiment"]["metrics"]["train"],
        label="Training HBIS",
        color="green",
    )
    plt.title("Training HBIS Over Epochs: Hydrogen Bond Experiment")
    plt.xlabel("Epochs")
    plt.ylabel("HBIS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "hydrogen_bond_experiment_training_hbis.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training HBIS plot: {e}")
    plt.close()

try:
    # Plot validation metrics (HBIS)
    plt.figure()
    plt.plot(
        experiment_data["hydrogen_bond_experiment"]["metrics"]["val"],
        label="Validation HBIS",
        color="red",
    )
    plt.title("Validation HBIS Over Epochs: Hydrogen Bond Experiment")
    plt.xlabel("Epochs")
    plt.ylabel("HBIS")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "hydrogen_bond_experiment_validation_hbis.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating validation HBIS plot: {e}")
    plt.close()
