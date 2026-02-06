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
        experiment_data["weight_decay_tuning"]["hydrogen_bond_experiment"]["losses"][
            "train"
        ],
        label="Training Loss",
    )
    plt.plot(
        experiment_data["weight_decay_tuning"]["hydrogen_bond_experiment"]["losses"][
            "val"
        ],
        label="Validation Loss",
    )
    plt.title("Training vs Validation Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "hydrogen_bond_experiment_loss_comparison.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating training vs validation loss plot: {e}")
    plt.close()

try:
    # Additional implementation for other plots can be added here based on available data
    pass
except Exception as e:
    print(f"Error generating additional plots: {e}")
    plt.close()
