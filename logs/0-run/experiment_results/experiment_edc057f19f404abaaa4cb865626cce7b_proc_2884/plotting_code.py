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

for optimizer in ["Adam", "SGD", "RMSprop"]:
    try:
        plt.figure()
        plt.plot(experiment_data[optimizer]["metrics"]["train"], label="Training HBIS")
        plt.plot(experiment_data[optimizer]["metrics"]["val"], label="Validation HBIS")
        plt.title(f"Optimizer: {optimizer} - HBIS Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("HBSI Score")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{optimizer}_hbis.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {optimizer}: {e}")
        plt.close()

for optimizer in ["Adam", "SGD", "RMSprop"]:
    try:
        plt.figure()
        plt.plot(experiment_data[optimizer]["losses"]["train"], label="Training Loss")
        plt.plot(experiment_data[optimizer]["losses"]["val"], label="Validation Loss")
        plt.title(f"Optimizer: {optimizer} - Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{optimizer}_loss.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {optimizer}: {e}")
        plt.close()
