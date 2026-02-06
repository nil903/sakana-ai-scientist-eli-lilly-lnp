markdown
import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data_paths = [
        "experiments/2026-02-06_07-43-32_eli_lilly_lnp_1_attempt_0/logs/0-run/experiment_results/experiment_0de9a7367cc2483d85394b3622a37b37_proc_2668/experiment_data.npy",
        "experiments/2026-02-06_07-43-32_eli_lilly_lnp_1_attempt_0/logs/0-run/experiment_results/experiment_6230d29865fb43fba5b0097e4f5f9d95_proc_2668/experiment_data.npy",
        "experiments/2026-02-06_07-43-32_eli_lilly_lnp_1_attempt_0/logs/0-run/experiment_results/experiment_eb6f3e73e93648f88287d32ffb6a61de_proc_2668/experiment_data.npy",
    ]
    all_experiment_data = []
    for path in experiment_data_paths:
        experiment_data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), path), allow_pickle=True
        ).item()
        all_experiment_data.append(experiment_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")

try:
    # Aggregate Training Loss
    training_losses = [
        data["hydrogen_bond_experiment"]["losses"]["train"]
        for data in all_experiment_data
    ]
    mean_training_loss = np.mean(training_losses, axis=0)
    se_training_loss = np.std(training_losses, axis=0) / np.sqrt(len(training_losses))

    plt.figure()
    plt.plot(mean_training_loss, label="Mean Training Loss")
    plt.fill_between(
        np.arange(len(mean_training_loss)),
        mean_training_loss - se_training_loss,
        mean_training_loss + se_training_loss,
        color="blue",
        alpha=0.2,
        label="Standard Error",
    )
    plt.title("Mean Training Loss Over Epochs (Hydrogen Bond Experiment)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "hydrogen_bond_experiment_mean_training_loss.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating mean training loss plot: {e}")
    plt.close()

try:
    # Aggregate Validation Loss
    validation_losses = [
        data["hydrogen_bond_experiment"]["losses"]["val"]
        for data in all_experiment_data
    ]
    mean_validation_loss = np.mean(validation_losses, axis=0)
    se_validation_loss = np.std(validation_losses, axis=0) / np.sqrt(
        len(validation_losses)
    )

    plt.figure()
    plt.plot(mean_validation_loss, label="Mean Validation Loss", color="orange")
    plt.fill_between(
        np.arange(len(mean_validation_loss)),
        mean_validation_loss - se_validation_loss,
        mean_validation_loss + se_validation_loss,
        color="orange",
        alpha=0.2,
        label="Standard Error",
    )
    plt.title("Mean Validation Loss Over Epochs (Hydrogen Bond Experiment)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "hydrogen_bond_experiment_mean_validation_loss.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating mean validation loss plot: {e}")
    plt.close()

try:
    # Aggregate Training Metrics
    training_metrics = [
        data["hydrogen_bond_experiment"]["metrics"]["train"]
        for data in all_experiment_data
    ]
    mean_training_hbis = np.mean(training_metrics, axis=0)
    se_training_hbis = np.std(training_metrics, axis=0) / np.sqrt(len(training_metrics))

    plt.figure()
    plt.plot(mean_training_hbis, label="Mean Training HBIS", color="green")
    plt.fill_between(
        np.arange(len(mean_training_hbis)),
        mean_training_hbis - se_training_hbis,
        mean_training_hbis + se_training_hbis,
        color="green",
        alpha=0.2,
        label="Standard Error",
    )
    plt.title("Mean Training HBIS Over Epochs (Hydrogen Bond Experiment)")
    plt.xlabel("Epochs")
    plt.ylabel("HBIS")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "hydrogen_bond_experiment_mean_training_hbis.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating mean training HBIS plot: {e}")
    plt.close()

try:
    # Aggregate Validation Metrics
    validation_metrics = [
        data["hydrogen_bond_experiment"]["metrics"]["val"]
        for data in all_experiment_data
    ]
    mean_validation_hbis = np.mean(validation_metrics, axis=0)
    se_validation_hbis = np.std(validation_metrics, axis=0) / np.sqrt(
        len(validation_metrics)
    )

    plt.figure()
    plt.plot(mean_validation_hbis, label="Mean Validation HBIS", color="red")
    plt.fill_between(
        np.arange(len(mean_validation_hbis)),
        mean_validation_hbis - se_validation_hbis,
        mean_validation_hbis + se_validation_hbis,
        color="red",
        alpha=0.2,
        label="Standard Error",
    )
    plt.title("Mean Validation HBIS Over Epochs (Hydrogen Bond Experiment)")
    plt.xlabel("Epochs")
    plt.ylabel("HBIS")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "hydrogen_bond_experiment_mean_validation_hbis.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating mean validation HBIS plot: {e}")
    plt.close()
