import matplotlib.pyplot as plt
import numpy as np
import os

# Create the figures directory
os.makedirs("figures", exist_ok=True)

# Load the experiment data from the provided .npy files
def load_experiment_data(file_path):
    try:
        return np.load(file_path, allow_pickle=True).item()
    except Exception as e:
        print(f"Error loading experiment data from {file_path}: {e}")
        return None

# Define file paths from JSON summaries
baseline_npy = "experiment_results/experiment_aaf3524e2b33491b982d1a6e21d1f12f_proc_2079/experiment_data.npy"
research_npy = "experiment_results/experiment_26c436781f254bd4b57bde764690a3e4_proc_2668/experiment_data.npy"
ablation_npy = "experiment_results/experiment_8bda5f677cba49b7a470fc1729758ec9_proc_2884/experiment_data.npy"

# Load data
baseline_data = load_experiment_data(baseline_npy)
research_data = load_experiment_data(research_npy)
ablation_data = load_experiment_data(ablation_npy)

# Plotting functions
def plot_weight_decay_losses(data):
    """Plot training and validation losses from weight decay tuning."""
    try:
        losses = data["weight_decay_tuning"]["hydrogen_bond_experiment"]["losses"]
        plt.figure(figsize=(12, 6))
        plt.plot(losses["train"], label="Training Loss")
        plt.plot(losses["val"], label="Validation Loss", color="orange")
        plt.title("Training and Validation Loss Over Weight Decay Tuning", fontsize=16)
        plt.xlabel("Epochs", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.legend()
        plt.grid()
        plt.savefig("figures/weight_decay_training_validation_loss.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error creating weight decay losses plot: {e}")

def plot_hbis(data):
    """Plot Hydrogen Bonding Interaction Score (HBIS)."""
    try:
        metrics = data["hydrogen_bond_experiment"]["metrics"]
        plt.figure(figsize=(12, 6))
        plt.plot(metrics["train"], label="Training HBIS", color="green")
        plt.plot(metrics["val"], label="Validation HBIS", color="red")
        plt.title("Hydrogen Bonding Interaction Score (HBIS) Over Epochs", fontsize=16)
        plt.xlabel("Epochs", fontsize=14)
        plt.ylabel("HBIS Score", fontsize=14)
        plt.legend()
        plt.grid()
        plt.savefig("figures/hbis_over_epochs.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error creating HBIS plot: {e}")

def plot_ablation(data):
    """Plot various ablation results."""
    try:
        datasets = data["MULTIPLE_SYNTHETIC_DATASETS"]
        plt.figure(figsize=(12, 12))
        noise_levels = datasets.keys()
        for i, noise_level in enumerate(noise_levels, 1):
            metrics = datasets[noise_level]
            plt.subplot(3, 2, i)
            plt.plot(metrics["losses"]["train"], label="Training Loss")
            plt.plot(metrics["losses"]["val"], label="Validation Loss")
            plt.title(f"Loss Curves for {noise_level} Noise Level", fontsize=12)
            plt.xlabel("Epochs", fontsize=10)
            plt.ylabel("Loss", fontsize=10)
            plt.legend()
            plt.grid()

        plt.tight_layout()
        plt.savefig("figures/ablation_noise_level_curves.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error creating ablation plots: {e}")

# Execute plotting functions
if baseline_data is not None:
    plot_weight_decay_losses(baseline_data)
    plot_hbis(baseline_data)

if research_data is not None:
    plot_hbis(research_data)

if ablation_data is not None:
    plot_ablation(ablation_data)

print("Plotting completed. Check the 'figures' directory for outputs.")