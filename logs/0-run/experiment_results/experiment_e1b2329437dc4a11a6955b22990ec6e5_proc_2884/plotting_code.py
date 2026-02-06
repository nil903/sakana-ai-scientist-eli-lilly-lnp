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

for act_name in experiment_data["activation_function_variation"]:
    loss_train = experiment_data["activation_function_variation"][act_name]["losses"][
        "train"
    ]
    loss_val = experiment_data["activation_function_variation"][act_name]["losses"][
        "val"
    ]
    metrics_train = experiment_data["activation_function_variation"][act_name][
        "metrics"
    ]["train"]
    metrics_val = experiment_data["activation_function_variation"][act_name]["metrics"][
        "val"
    ]

    try:
        plt.figure()
        plt.plot(loss_train, label="Training Loss")
        plt.plot(loss_val, label="Validation Loss")
        plt.title(f"{act_name} Activation Function - Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{act_name}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {act_name}: {e}")
        plt.close()

    try:
        plt.figure()
        plt.plot(metrics_train, label="Training HBIS")
        plt.plot(metrics_val, label="Validation HBIS")
        plt.title(f"{act_name} Activation Function - HBIS Curves")
        plt.xlabel("Epochs")
        plt.ylabel("HBIS Score")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{act_name}_hbis_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating HBIS plot for {act_name}: {e}")
        plt.close()
