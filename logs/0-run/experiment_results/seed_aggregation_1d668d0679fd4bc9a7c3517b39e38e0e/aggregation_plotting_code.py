import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data_paths = [
        "experiments/2026-02-06_07-43-32_eli_lilly_lnp_1_attempt_0/logs/0-run/experiment_results/experiment_479ca1797490410cbe54f73dd5b7cc46_proc_2017/experiment_data.npy",
        "experiments/2026-02-06_07-43-32_eli_lilly_lnp_1_attempt_0/logs/0-run/experiment_results/experiment_74660dd8bcbc4262b9bb2f15044c4bc2_proc_2017/experiment_data.npy",
        "experiments/2026-02-06_07-43-32_eli_lilly_lnp_1_attempt_0/logs/0-run/experiment_results/experiment_7760187caea64505ac4dca0f3e69c15d_proc_2017/experiment_data.npy",
    ]

    all_experiment_data = []
    for experiment_data_path in experiment_data_paths:
        experiment_data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), experiment_data_path),
            allow_pickle=True,
        ).item()
        all_experiment_data.append(experiment_data)

except Exception as e:
    print(f"Error loading experiment data: {e}")

try:
    # Calculate mean and standard error for training loss
    training_losses = [exp["metrics"]["train"] for exp in all_experiment_data]
    training_losses_mean = np.mean(training_losses, axis=0)
    training_losses_se = np.std(training_losses, axis=0) / np.sqrt(len(training_losses))

    plt.figure()
    epochs = np.arange(len(training_losses_mean))
    plt.plot(epochs, training_losses_mean, label="Mean Training Loss")
    plt.fill_between(
        epochs,
        training_losses_mean - training_losses_se,
        training_losses_mean + training_losses_se,
        color="lightgrey",
        alpha=0.5,
        label="Standard Error",
    )
    plt.title("Mean Training Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "mean_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating mean training loss plot: {e}")
    plt.close()

try:
    # Ground truth vs Predictions
    ground_truth = all_experiment_data[0]["ground_truth"]
    predictions = [exp["predictions"] for exp in all_experiment_data]
    predictions_mean = np.mean(predictions, axis=0)
    predictions_se = np.std(predictions, axis=0)

    plt.figure()
    plt.scatter(ground_truth, predictions_mean, alpha=0.5)
    plt.errorbar(
        ground_truth,
        predictions_mean,
        yerr=predictions_se,
        fmt="o",
        label="Mean Predictions with SE",
        color="orange",
    )
    plt.title("Ground Truth vs Mean Predictions")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.plot([0, 10], [0, 10], "r--")  # Identity line
    plt.legend()
    plt.savefig(os.path.join(working_dir, "ground_truth_vs_mean_predictions.png"))
    plt.close()
except Exception as e:
    print(f"Error creating ground truth vs predictions plot: {e}")
    plt.close()
