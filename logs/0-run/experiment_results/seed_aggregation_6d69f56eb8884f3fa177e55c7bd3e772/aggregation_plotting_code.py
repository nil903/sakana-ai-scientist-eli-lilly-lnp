import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-02-06_07-43-32_eli_lilly_lnp_1_attempt_0/logs/0-run/experiment_results/experiment_59e30fb90f354675b7dc7fd0f2480572_proc_2884/experiment_data.npy",
        "experiments/2026-02-06_07-43-32_eli_lilly_lnp_1_attempt_0/logs/0-run/experiment_results/experiment_22fbad50219247c8bb9f60b6f5c114ac_proc_2884/experiment_data.npy",
        "experiments/2026-02-06_07-43-32_eli_lilly_lnp_1_attempt_0/logs/0-run/experiment_results/experiment_1bdbf16a828e49b58dcb4c59319e5133_proc_2884/experiment_data.npy",
    ]
    all_experiment_data = []
    for experiment_data_path in experiment_data_path_list:
        experiment_data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), experiment_data_path),
            allow_pickle=True,
        ).item()
        all_experiment_data.append(experiment_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")

for dataset_name in experiment_data["MULTI_DISTRIBUTION_EVALUATION"].keys():
    try:
        losses_train = []
        losses_val = []
        metrics_train = []
        metrics_val = []

        for experiment in all_experiment_data:
            losses_train.append(
                experiment["MULTI_DISTRIBUTION_EVALUATION"][dataset_name]["losses"][
                    "train"
                ]
            )
            losses_val.append(
                experiment["MULTI_DISTRIBUTION_EVALUATION"][dataset_name]["losses"][
                    "val"
                ]
            )
            metrics_train.append(
                experiment["MULTI_DISTRIBUTION_EVALUATION"][dataset_name]["metrics"][
                    "train"
                ]
            )
            metrics_val.append(
                experiment["MULTI_DISTRIBUTION_EVALUATION"][dataset_name]["metrics"][
                    "val"
                ]
            )

        epochs = range(len(losses_train[0]))

        plt.figure()
        plt.errorbar(
            epochs,
            np.mean(losses_train, axis=0),
            yerr=np.std(losses_train, axis=0) / np.sqrt(len(losses_train)),
            label="Mean Train Loss ± SE",
            capsize=5,
        )
        plt.errorbar(
            epochs,
            np.mean(losses_val, axis=0),
            yerr=np.std(losses_val, axis=0) / np.sqrt(len(losses_val)),
            label="Mean Validation Loss ± SE",
            capsize=5,
        )
        plt.title(f"{dataset_name.capitalize()} Dataset Losses with Error Bars")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_losses_aggregated.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dataset_name}: {e}")
        plt.close()

    try:
        plt.figure()
        plt.errorbar(
            epochs,
            np.mean(metrics_train, axis=0),
            yerr=np.std(metrics_train, axis=0) / np.sqrt(len(metrics_train)),
            label="Mean Train HBIS ± SE",
            capsize=5,
        )
        plt.errorbar(
            epochs,
            np.mean(metrics_val, axis=0),
            yerr=np.std(metrics_val, axis=0) / np.sqrt(len(metrics_val)),
            label="Mean Validation HBIS ± SE",
            capsize=5,
        )
        plt.title(f"{dataset_name.capitalize()} Dataset Metrics with Error Bars")
        plt.xlabel("Epochs")
        plt.ylabel("HBIS")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_metrics_aggregated.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating metric plot for {dataset_name}: {e}")
        plt.close()
