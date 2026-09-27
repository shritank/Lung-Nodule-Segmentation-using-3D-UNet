"""
Plots mean +/- std train/val loss vs epoch for each architecture, using the
lightweight per-seed loss histories saved by compare_models.py
(loss_history/<variant>_seed<seed>.csv). Does not require re-running training.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_DIR    = Path(r"C:\Users\ramar\OneDrive\Desktop\Lung Nodule Analysis")
RESULTS_DIR    = PROJECT_DIR / "results"
HISTORY_DIR    = RESULTS_DIR / "loss_history"
MODEL_VARIANTS = ["base", "attention", "selective_attention", "se_residual"]
SEEDS          = [0, 1, 2, 3, 4]


def load_variant_histories(variant):
    """
    Load per-seed loss histories for a variant and align them on epoch.
    Returns (epochs, train_mean, train_std, val_mean, val_std) or None
    if no history files are found.
    """
    per_seed_train = []
    per_seed_val   = []
    epochs = None

    for seed in SEEDS:
        csv_path = HISTORY_DIR / f"{variant}_seed{seed}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if epochs is None:
            epochs = df['epoch'].to_numpy()
        per_seed_train.append(df['train_loss'].to_numpy())
        per_seed_val.append(df['val_loss'].to_numpy())

    if not per_seed_train:
        return None

    train_arr = np.stack(per_seed_train)  # (n_seeds, n_epochs)
    val_arr   = np.stack(per_seed_val)

    train_mean = train_arr.mean(axis=0)
    val_mean   = val_arr.mean(axis=0)

    if train_arr.shape[0] > 1:
        train_std = train_arr.std(axis=0, ddof=1)
        val_std   = val_arr.std(axis=0, ddof=1)
    else:
        train_std = np.zeros_like(train_mean)
        val_std   = np.zeros_like(val_mean)

    return epochs, train_mean, train_std, val_mean, val_std


def plot_per_variant(histories):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (variant, (epochs, train_mean, train_std, val_mean, val_std)) in zip(axes, histories.items()):
        ax.plot(epochs, train_mean, label="Train Loss (mean)")
        ax.fill_between(epochs, train_mean - train_std, train_mean + train_std, alpha=0.2)
        ax.plot(epochs, val_mean, label="Val Loss (mean)")
        ax.fill_between(epochs, val_mean - val_std, val_mean + val_std, alpha=0.2)
        ax.set_title(variant)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    for ax in axes[len(histories):]:
        ax.axis("off")

    fig.tight_layout()
    out_path = RESULTS_DIR / "loss_curves_per_variant.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


def plot_val_comparison(histories):
    fig, ax = plt.subplots(figsize=(8, 6))

    for variant, (epochs, _, _, val_mean, val_std) in histories.items():
        line, = ax.plot(epochs, val_mean, label=variant)
        ax.fill_between(epochs, val_mean - val_std, val_mean + val_std,
                        color=line.get_color(), alpha=0.15)

    ax.set_title(f"Validation Loss Comparison Across Architectures (mean ± std, n={len(SEEDS)} seeds)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = RESULTS_DIR / "val_loss_comparison.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    histories = {}
    for variant in MODEL_VARIANTS:
        result = load_variant_histories(variant)
        if result is None:
            print(f"  Warning: no loss history found for '{variant}' — skipping.")
            continue
        histories[variant] = result
        print(f"  {variant}: loaded loss history")

    if not histories:
        print("No loss history found for any variant.")
        return

    plot_per_variant(histories)
    plot_val_comparison(histories)
    plt.show()


if __name__ == "__main__":
    main()
