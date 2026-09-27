import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, random_split

from model_base import UNet3D

from train import (
    LunaDataset,
    CombinedLoss,
    VAL_SPLIT,
    BATCH_SIZE,
    POS_WEIGHT,
    SAVE_DIR
)

from main import (
    build_scan_index,
    build_samples,
    ANNOT_PATH
)


# =====================================================
# Metric Functions
# =====================================================

def get_confusion(predictions, targets, threshold=0.5):
    """
    Returns:
        TP, FP, TN, FN
    """

    predictions = torch.sigmoid(predictions)
    predictions = (predictions > threshold).float()

    tp = ((predictions == 1) & (targets == 1)).sum().item()
    fp = ((predictions == 1) & (targets == 0)).sum().item()
    tn = ((predictions == 0) & (targets == 0)).sum().item()
    fn = ((predictions == 0) & (targets == 1)).sum().item()

    return tp, fp, tn, fn


def compute_metrics(tp, fp, tn, fn):

    eps = 1e-8

    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    precision = tp / (tp + fp + eps)

    recall = tp / (tp + fn + eps)

    specificity = tn / (tn + fp + eps)

    iou = tp / (tp + fp + fn + eps)

    dice = (2 * tp) / (2 * tp + fp + fn + eps)

    f1 = dice

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "IoU": iou,
        "Dice": dice,
        "F1 Score": f1,
    }

# =====================================================
# Evaluation
# =====================================================

def evaluate():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Running on: {device}")

    # -----------------------------
    # Build validation dataset
    # -----------------------------
    mhd_files = build_scan_index()

    annotations = pd.read_csv(ANNOT_PATH)

    samples = build_samples(
        annotations,
        mhd_files
    )

    val_size = int(len(samples) * VAL_SPLIT)
    train_size = len(samples) - val_size

    train_subset, val_subset = random_split(
        samples,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    val_samples = [
        samples[i]
        for i in val_subset.indices
    ]

    val_dataset = LunaDataset(
        val_samples,
        mhd_files,
        augment=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # -----------------------------
    # Load Model
    # -----------------------------
    model = UNet3D().to(device)

    checkpoint = torch.load(
        SAVE_DIR / "best_model.pth",
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    criterion = CombinedLoss(
        pos_weight=POS_WEIGHT
    )

    total_loss = 0.0

    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0

    # -----------------------------
    # Run Evaluation
    # -----------------------------
    with torch.no_grad():

        for batch_idx, (patches, labels) in enumerate(val_loader):

            patches = patches.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            predictions = model(patches)

            loss = criterion(predictions, labels)

            total_loss += loss.item()

            tp, fp, tn, fn = get_confusion(
                predictions,
                labels
            )

            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn

            if (batch_idx + 1) % 20 == 0:
                print(
                    f"Processed {batch_idx + 1}/{len(val_loader)} batches..."
                )

    avg_loss = total_loss / len(val_loader)

    metrics = compute_metrics(
        total_tp,
        total_fp,
        total_tn,
        total_fn
    )
    # -----------------------------
    # Print Results
    # -----------------------------
    print("           BASELINE 3D U-NET EVALUATION RESULTS")

    print(f"\nValidation Loss : {avg_loss:.4f}\n")

    print("Performance Metrics")
    print("-" * 35)

    print(f"Dice Score      : {metrics['Dice']:.4f}")
    print(f"IoU             : {metrics['IoU']:.4f}")
    print(f"Precision       : {metrics['Precision']:.4f}")
    print(f"Recall          : {metrics['Recall']:.4f}")
    print(f"Specificity     : {metrics['Specificity']:.4f}")
    print(f"Accuracy        : {metrics['Accuracy']:.4f}")
    print(f"F1 Score        : {metrics['F1 Score']:.4f}")

    print("\nConfusion Matrix")
    print("-" * 35)
    print(f"True Positives  : {total_tp:,}")
    print(f"False Positives : {total_fp:,}")
    print(f"True Negatives  : {total_tn:,}")
    print(f"False Negatives : {total_fn:,}")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    evaluate()