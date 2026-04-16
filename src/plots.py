from pathlib import Path
import matplotlib.pyplot as plt


def plot_adaptive_pr_auc(adaptive_df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(adaptive_df["batch_id"], adaptive_df["old_pr_auc"], marker="o", label="Before Retraining")
    plt.plot(adaptive_df["batch_id"], adaptive_df["new_pr_auc"], marker="o", label="After Retraining")

    retrained_batches = adaptive_df[adaptive_df["retrained"] == True]["batch_id"]
    retrained_scores = adaptive_df[adaptive_df["retrained"] == True]["new_pr_auc"]

    plt.scatter(retrained_batches, retrained_scores, marker="x", s=100, label="Retraining Triggered")

    plt.xlabel("Batch ID")
    plt.ylabel("PR-AUC")
    plt.title("Adaptive Retraining: PR-AUC Before vs After Retraining")
    plt.xticks(adaptive_df["batch_id"])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "phase4_adaptive_pr_auc.png")
    plt.close()


def plot_avg_psi(adaptive_df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(adaptive_df["batch_id"], adaptive_df["avg_psi"], marker="o")

    plt.axhline(y=0.1, linestyle="--", label="Drift Threshold (0.1)")

    plt.xlabel("Batch ID")
    plt.ylabel("Average PSI")
    plt.title("Average PSI by Batch")
    plt.xticks(adaptive_df["batch_id"])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "phase4_avg_psi.png")
    plt.close()


def plot_drift_vs_performance(adaptive_df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(adaptive_df["batch_id"], adaptive_df["avg_psi"], marker="o", label="Average PSI")
    plt.plot(adaptive_df["batch_id"], adaptive_df["old_pr_auc"], marker="o", label="Old PR-AUC")
    plt.plot(adaptive_df["batch_id"], adaptive_df["new_pr_auc"], marker="o", label="New PR-AUC")

    plt.xlabel("Batch ID")
    plt.ylabel("Score")
    plt.title("Drift vs Model Performance Across Batches")
    plt.xticks(adaptive_df["batch_id"])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "phase4_drift_vs_performance.png")
    plt.close()