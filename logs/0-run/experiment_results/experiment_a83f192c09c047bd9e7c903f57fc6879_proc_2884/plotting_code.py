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
    plt.figure()
    plt.plot(
        experiment_data["model_ablation_study"]["2_layers"]["metrics"]["train"],
        label="Train HBIS",
    )
    plt.plot(
        experiment_data["model_ablation_study"]["2_layers"]["metrics"]["val"],
        label="Val HBIS",
    )
    plt.title("Model: 2 Layers - Performance Metrics")
    plt.xlabel("Epochs")
    plt.ylabel("HBIS Score")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "2_layers_metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot for 2 layers metrics: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(
        experiment_data["model_ablation_study"]["3_layers"]["metrics"]["train"],
        label="Train HBIS",
    )
    plt.plot(
        experiment_data["model_ablation_study"]["3_layers"]["metrics"]["val"],
        label="Val HBIS",
    )
    plt.title("Model: 3 Layers - Performance Metrics")
    plt.xlabel("Epochs")
    plt.ylabel("HBIS Score")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "3_layers_metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot for 3 layers metrics: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(
        experiment_data["model_ablation_study"]["4_layers"]["metrics"]["train"],
        label="Train HBIS",
    )
    plt.plot(
        experiment_data["model_ablation_study"]["4_layers"]["metrics"]["val"],
        label="Val HBIS",
    )
    plt.title("Model: 4 Layers - Performance Metrics")
    plt.xlabel("Epochs")
    plt.ylabel("HBIS Score")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "4_layers_metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot for 4 layers metrics: {e}")
    plt.close()
