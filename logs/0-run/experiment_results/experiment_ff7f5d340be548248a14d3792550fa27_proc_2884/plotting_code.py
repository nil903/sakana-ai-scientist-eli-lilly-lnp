import matplotlib.pyplot as plt
import numpy as np
import os

# Define working directory
working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

for noise_level in experiment_data["DATA_NOISE_LEVEL_ABLATION"].keys():
    noise_data = experiment_data["DATA_NOISE_LEVEL_ABLATION"][noise_level]

    try:
        plt.figure()
        plt.plot(noise_data["losses"]["train"], label="Training Loss")
        plt.plot(noise_data["losses"]["val"], label="Validation Loss")
        plt.title(f"Loss Curves for {noise_level} Noise Level")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{noise_level}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve for {noise_level}: {e}")
        plt.close()
