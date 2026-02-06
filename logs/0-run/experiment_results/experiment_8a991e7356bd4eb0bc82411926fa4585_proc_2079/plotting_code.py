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

# Plot training losses for each learning rate
learning_rates = [0.01, 0.001, 0.0005, 0.0001]
for lr in learning_rates:
    try:
        losses = experiment_data["hyperparam_tuning_type_1"]["learning_rate_tuning"][
            "losses"
        ]["train"]
        plt.figure()
        plt.plot(losses, label=f"Learning Rate: {lr}")
        plt.title("Training Losses Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"training_loss_lr_{lr}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for learning rate {lr}: {e}")
        plt.close()

# Plot predictions vs ground truth for the last learning rate
try:
    predictions = experiment_data["hyperparam_tuning_type_1"]["learning_rate_tuning"][
        "predictions"
    ][-1]
    ground_truth = experiment_data["hyperparam_tuning_type_1"]["learning_rate_tuning"][
        "ground_truth"
    ][-1]

    plt.figure()
    plt.scatter(ground_truth, predictions, alpha=0.5)
    plt.plot(
        [ground_truth.min(), ground_truth.max()],
        [ground_truth.min(), ground_truth.max()],
        color="red",
        linestyle="--",
    )
    plt.title("Predictions vs Ground Truth")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.savefig(os.path.join(working_dir, "predictions_vs_ground_truth.png"))
    plt.close()
except Exception as e:
    print(f"Error creating predictions vs ground truth plot: {e}")
    plt.close()
