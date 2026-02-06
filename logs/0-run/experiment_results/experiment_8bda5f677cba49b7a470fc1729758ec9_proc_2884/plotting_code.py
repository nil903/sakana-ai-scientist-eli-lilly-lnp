import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

for dataset_name in experiment_data["MULTIPLE_SYNTHETIC_DATASETS"]:
    try:
        plt.figure()
        plt.plot(
            experiment_data["MULTIPLE_SYNTHETIC_DATASETS"][dataset_name]["metrics"][
                "train"
            ],
            label="Training HBIS",
        )
        plt.plot(
            experiment_data["MULTIPLE_SYNTHETIC_DATASETS"][dataset_name]["metrics"][
                "val"
            ],
            label="Validation HBIS",
        )
        plt.title(f"{dataset_name}: HBIS Metric Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("HBIS Score")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_hbis_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating HBIS plot for {dataset_name}: {e}")
        plt.close()

    try:
        plt.figure()
        plt.plot(
            experiment_data["MULTIPLE_SYNTHETIC_DATASETS"][dataset_name]["losses"][
                "train"
            ],
            label="Training Loss",
        )
        plt.plot(
            experiment_data["MULTIPLE_SYNTHETIC_DATASETS"][dataset_name]["losses"][
                "val"
            ],
            label="Validation Loss",
        )
        plt.title(f"{dataset_name}: Loss Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dataset_name}: {e}")
        plt.close()
