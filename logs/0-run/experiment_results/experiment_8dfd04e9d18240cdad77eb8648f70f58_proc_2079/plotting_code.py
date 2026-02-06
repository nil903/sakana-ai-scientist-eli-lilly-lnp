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

for momentum in np.arange(0.5, 1.0, 0.1):
    try:
        labels = experiment_data["hyperparam_tuning_momentum"][
            "hydrogen_bond_experiment"
        ]["losses"]["train"]
        plt.figure()
        plt.plot(labels, label=f"Momentum={momentum:.1f}")
        plt.title("Training Loss vs Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            os.path.join(
                working_dir,
                f"hydrogen_bond_experiment_loss_momentum_{momentum:.1f}.png",
            )
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot for momentum={momentum:.1f}: {e}")
        plt.close()
