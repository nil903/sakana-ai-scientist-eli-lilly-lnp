import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data_path_list = [
        "experiments/2026-02-06_07-43-32_eli_lilly_lnp_1_attempt_0/logs/0-run/experiment_results/experiment_84143691da8c42bf9ab90c1eddcd9c08_proc_2079/experiment_data.npy",
        "experiments/2026-02-06_07-43-32_eli_lilly_lnp_1_attempt_0/logs/0-run/experiment_results/experiment_e339d85ff2d7421ab488c1f09f1a52a4_proc_2079/experiment_data.npy",
        "experiments/2026-02-06_07-43-32_eli_lilly_lnp_1_attempt_0/logs/0-run/experiment_results/experiment_60b935ad16224596bad15e44bdeae72d_proc_2079/experiment_data.npy",
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

try:
    # Calculate mean and standard error for training losses
    train_losses = [
        exp["weight_decay_tuning"]["hydrogen_bond_experiment"]["losses"]["train"]
        for exp in all_experiment_data
    ]
    train_losses_mean = np.mean(train_losses, axis=0)
    train_losses_se = np.std(train_losses, axis=0) / np.sqrt(len(train_losses))

    plt.figure()
    plt.plot(train_losses_mean, label="Mean Training Loss")
    plt.fill_between(
        range(len(train_losses_mean)),
        train_losses_mean - train_losses_se,
        train_losses_mean + train_losses_se,
        alpha=0.2,
        label="Standard Error",
    )
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "hydrogen_bond_experiment_training_loss_mean_se.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

try:
    # Calculate mean and standard error for validation losses if exists
    val_losses = [
        exp["weight_decay_tuning"]["hydrogen_bond_experiment"]["losses"]["val"]
        for exp in all_experiment_data
        if "val" in exp["weight_decay_tuning"]["hydrogen_bond_experiment"]["losses"]
    ]
    val_losses_mean = np.mean(val_losses, axis=0)
    val_losses_se = np.std(val_losses, axis=0) / np.sqrt(len(val_losses))

    plt.figure()
    plt.plot(val_losses_mean, label="Mean Validation Loss")
    plt.fill_between(
        range(len(val_losses_mean)),
        val_losses_mean - val_losses_se,
        val_losses_mean + val_losses_se,
        alpha=0.2,
        label="Standard Error",
    )
    plt.title("Validation Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(
            working_dir, "hydrogen_bond_experiment_validation_loss_mean_se.png"
        )
    )
    plt.close()
except Exception as e:
    print(f"Error creating validation loss plot: {e}")
    plt.close()
