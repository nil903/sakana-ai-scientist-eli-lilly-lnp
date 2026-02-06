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

# Plot training and validation losses
for dataset_name in experiment_data["MULTI_DISTRIBUTION_LEARNING"]:
    try:
        plt.figure()
        plt.plot(
            experiment_data["MULTI_DISTRIBUTION_LEARNING"][dataset_name]["losses"][
                "train"
            ],
            label="Train Loss",
        )
        plt.plot(
            experiment_data["MULTI_DISTRIBUTION_LEARNING"][dataset_name]["losses"][
                "val"
            ],
            label="Validation Loss",
        )
        plt.title(f"{dataset_name.capitalize()} Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating {dataset_name} loss plot: {e}")
        plt.close()

    try:
        plt.figure()
        plt.plot(
            experiment_data["MULTI_DISTRIBUTION_LEARNING"][dataset_name]["metrics"][
                "train"
            ],
            label="Train HBIS",
        )
        plt.plot(
            experiment_data["MULTI_DISTRIBUTION_LEARNING"][dataset_name]["metrics"][
                "val"
            ],
            label="Validation HBIS",
        )
        plt.title(f"{dataset_name.capitalize()} HBIS Curves")
        plt.xlabel("Epochs")
        plt.ylabel("HBIS Score")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_hbis_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating {dataset_name} HBIS plot: {e}")
        plt.close()
