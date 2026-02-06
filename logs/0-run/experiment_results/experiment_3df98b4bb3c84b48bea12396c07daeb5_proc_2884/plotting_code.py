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

for usage in ["without_dropout", "with_dropout"]:
    try:
        epochs = np.arange(
            len(experiment_data["dropout_ablation"][usage]["losses"]["train"])
        )
        plt.figure()
        plt.plot(
            epochs,
            experiment_data["dropout_ablation"][usage]["losses"]["train"],
            label="Train Loss",
        )
        plt.plot(
            epochs,
            experiment_data["dropout_ablation"][usage]["losses"]["val"],
            label="Validation Loss",
        )
        plt.title(f"{usage.capitalize()}: Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{usage}_loss_plot.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {usage}: {e}")
        plt.close()

    try:
        epochs = np.arange(
            len(experiment_data["dropout_ablation"][usage]["metrics"]["train"])
        )
        plt.figure()
        plt.plot(
            epochs,
            experiment_data["dropout_ablation"][usage]["metrics"]["train"],
            label="Train HBD",
        )
        plt.plot(
            epochs,
            experiment_data["dropout_ablation"][usage]["metrics"]["val"],
            label="Validation HBD",
        )
        plt.title(f"{usage.capitalize()}: Training and Validation HBD Metric")
        plt.xlabel("Epochs")
        plt.ylabel("Metric")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{usage}_hbd_metric_plot.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating HBD metric plot for {usage}: {e}")
        plt.close()
