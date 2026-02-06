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

for dataset_name in experiment_data["MULTI_DISTRIBUTION_EVALUATION"].keys():
    try:
        epochs = range(
            len(
                experiment_data["MULTI_DISTRIBUTION_EVALUATION"][dataset_name][
                    "losses"
                ]["train"]
            )
        )
        plt.figure()
        plt.plot(
            epochs,
            experiment_data["MULTI_DISTRIBUTION_EVALUATION"][dataset_name]["losses"][
                "train"
            ],
            label="Train Loss",
        )
        plt.plot(
            epochs,
            experiment_data["MULTI_DISTRIBUTION_EVALUATION"][dataset_name]["losses"][
                "val"
            ],
            label="Validation Loss",
        )
        plt.title(f"{dataset_name.capitalize()} Dataset Losses")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_losses.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dataset_name}: {e}")
        plt.close()

    try:
        plt.figure()
        plt.plot(
            epochs,
            experiment_data["MULTI_DISTRIBUTION_EVALUATION"][dataset_name]["metrics"][
                "train"
            ],
            label="Train HBIS",
        )
        plt.plot(
            epochs,
            experiment_data["MULTI_DISTRIBUTION_EVALUATION"][dataset_name]["metrics"][
                "val"
            ],
            label="Validation HBIS",
        )
        plt.title(f"{dataset_name.capitalize()} Dataset Metrics")
        plt.xlabel("Epochs")
        plt.ylabel("HBIS")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_metrics.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating metric plot for {dataset_name}: {e}")
        plt.close()
