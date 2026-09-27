import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import random
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, random_split

from main import build_scan_index, build_samples, ANNOT_PATH
from train import LunaDataset, CombinedLoss, PATCH_SIZE, BATCH_SIZE, LR, VAL_SPLIT, POS_WEIGHT, NUM_WORKERS
from evaluation import get_confusion, compute_metrics

from model_base import UNet3D as BaseUNet3D
from model_att import UNet3D as AttentionUNet3D
from model_selectiveatt import UNet3D as SelectiveAttentionUNet3D
from model_se_residual import UNet3D as SEResidualUNet3D

# ─── Config ───────────────────────────────────────────────────────────────────
NUM_EPOCHS          = 50    # matches baseline train.py — val loss plateaus well before this
DIAMETER_THRESHOLD  = 6.4   # median diameter — splits val set into small vs large
SEEDS               = [0, 1, 2, 3, 4]  # same seeds reused across all variants for paired comparison
PROJECT_DIR         = Path(r"C:\Users\ramar\OneDrive\Desktop\Lung Nodule Analysis")
RESULTS_DIR         = PROJECT_DIR / "results"
HISTORY_DIR         = RESULTS_DIR / "loss_history"
RAW_CSV_PATH        = RESULTS_DIR / "model_comparison_raw_results.csv"

MODEL_VARIANTS = {
    "base"                : BaseUNet3D,
    "attention"           : AttentionUNet3D,
    "selective_attention" : SelectiveAttentionUNet3D,
    "se_residual"         : SEResidualUNet3D,
}


# ─── Seeding ──────────────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─── Training ─────────────────────────────────────────────────────────────────
def train_variant(name, seed, model_class, train_loader, val_loader, device):
    print(f"\n{'='*60}", flush=True)
    print(f"Training variant: {name} (seed {seed})", flush=True)
    print(f"{'='*60}\n", flush=True)

    save_dir = PROJECT_DIR / f"checkpoints_{name}"
    save_dir.mkdir(parents=True, exist_ok=True)

    model = model_class().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}", flush=True)

    criterion = CombinedLoss(pos_weight=POS_WEIGHT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    scaler = torch.amp.GradScaler('cuda')

    best_val_loss = float('inf')
    history = []  # (epoch, train_loss, val_loss) — kept lightweight, used for plotting

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0

        for batch_idx, (patches, labels) in enumerate(train_loader):
            patches = patches.to(device, non_blocking=True)
            labels  = labels.to(device,  non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                predictions = model(patches)
                loss        = criterion(predictions, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"[{name} seed{seed}] Epoch {epoch+1}/{NUM_EPOCHS} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}", flush=True)

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for patches, labels in val_loader:
                patches = patches.to(device, non_blocking=True)
                labels  = labels.to(device,  non_blocking=True)

                with torch.amp.autocast('cuda'):
                    predictions = model(patches)
                    loss        = criterion(predictions, labels)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"\n-- [{name} seed{seed}] Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} --\n", flush=True)

        scheduler.step(avg_val_loss)
        history.append((epoch + 1, avg_train_loss, avg_val_loss))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                'epoch'          : epoch + 1,
                'variant'        : name,
                'seed'           : seed,
                'model_state'    : model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'train_loss'     : avg_train_loss,
                'val_loss'       : avg_val_loss
            }
            torch.save(checkpoint, save_dir / "best_model.pth")
            print(f"  [{name} seed{seed}] Best model saved (val loss: {best_val_loss:.4f})", flush=True)

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame(history, columns=["epoch", "train_loss", "val_loss"])
    history_df.to_csv(HISTORY_DIR / f"{name}_seed{seed}.csv", index=False)

    print(f"\n[{name} seed{seed}] Training complete. Best val loss: {best_val_loss:.4f}", flush=True)
    return save_dir / "best_model.pth", best_val_loss, total_params


# ─── Evaluation ───────────────────────────────────────────────────────────────
def load_variant(model_class, checkpoint_path, device):
    model      = model_class().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model


def evaluate_on_samples(model, samples, mhd_files, device, label):
    if not samples:
        return None

    dataset = LunaDataset(samples, mhd_files, augment=False)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)

    criterion = CombinedLoss(pos_weight=POS_WEIGHT)
    total_loss = 0.0
    total_tp = total_fp = total_tn = total_fn = 0

    with torch.no_grad():
        for patches, labels in loader:
            patches = patches.to(device, non_blocking=True)
            labels  = labels.to(device,  non_blocking=True)

            with torch.amp.autocast('cuda'):
                predictions = model(patches)
                loss        = criterion(predictions, labels)

            total_loss += loss.item()

            tp, fp, tn, fn = get_confusion(predictions, labels)
            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn

    avg_loss = total_loss / len(loader)
    metrics  = compute_metrics(total_tp, total_fp, total_tn, total_fn)
    metrics['val_loss'] = avg_loss

    print(f"\n  {label}", flush=True)
    print(f"  Samples   : {len(samples)}", flush=True)
    print(f"  Val Loss  : {avg_loss:.4f}", flush=True)
    print(f"  Dice      : {metrics['Dice']:.4f}", flush=True)
    print(f"  IoU       : {metrics['IoU']:.4f}", flush=True)
    print(f"  Precision : {metrics['Precision']:.4f}", flush=True)
    print(f"  Recall    : {metrics['Recall']:.4f}", flush=True)

    return metrics


# ─── Main ─────────────────────────────────────────────────────────────────────
def run():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available — this script requires a GPU to train on.")

    device = torch.device("cuda")
    print(f"Running on: {device}", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nBuilding dataset...", flush=True)
    mhd_files   = build_scan_index()
    annotations = pd.read_csv(ANNOT_PATH)
    samples     = build_samples(annotations, mhd_files)
    print(f"Total samples: {len(samples)}", flush=True)

    # Train/val split — fixed seed 42, independent of the experiment seeds below,
    # so every variant/seed combination trains and evaluates on the exact same split.
    val_size   = int(len(samples) * VAL_SPLIT)
    train_size = len(samples) - val_size
    train_subset, val_subset = random_split(
        samples,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_samples = [samples[i] for i in train_subset.indices]
    val_samples   = [samples[i] for i in val_subset.indices]
    print(f"Train samples: {len(train_samples)} | Val samples: {len(val_samples)}", flush=True)

    small_val = [s for s in val_samples if s['diameter'] <  DIAMETER_THRESHOLD]
    large_val = [s for s in val_samples if s['diameter'] >= DIAMETER_THRESHOLD]
    print(f"Val split by size — small (<{DIAMETER_THRESHOLD}mm): {len(small_val)} | large (>={DIAMETER_THRESHOLD}mm): {len(large_val)}", flush=True)

    train_dataset = LunaDataset(train_samples, mhd_files, augment=True)
    val_dataset   = LunaDataset(val_samples,   mhd_files, augment=False)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    # Reuse any (variant, seed) combinations already completed in a previous run —
    # only the missing ones get trained.
    existing_results = []
    completed = set()
    if RAW_CSV_PATH.exists():
        existing_df = pd.read_csv(RAW_CSV_PATH)
        existing_results = existing_df.to_dict('records')
        completed = {(r['variant'], r['seed']) for r in existing_results}
        print(f"\nFound {len(completed)} already-completed (variant, seed) runs in {RAW_CSV_PATH.name} — skipping those.", flush=True)

    raw_results = []  # one row per newly-trained (variant, seed)

    for name, model_class in MODEL_VARIANTS.items():
        for seed in SEEDS:
            if (name, seed) in completed:
                print(f"\nSkipping {name} seed {seed} — already completed.", flush=True)
                continue

            set_seed(seed)

            # Seeded generator — train batch order is identical across variants
            # for a given seed, only the architecture differs.
            train_generator = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                      num_workers=NUM_WORKERS, pin_memory=True,
                                      generator=train_generator)

            checkpoint_path, best_val_loss, total_params = train_variant(
                name, seed, model_class, train_loader, val_loader, device
            )

            model = load_variant(model_class, checkpoint_path, device)

            overall_metrics = evaluate_on_samples(
                model, val_samples, mhd_files, device,
                label=f"[{name} seed{seed}] Overall validation set"
            )
            small_metrics = evaluate_on_samples(
                model, small_val, mhd_files, device,
                label=f"[{name} seed{seed}] Small nodules (<{DIAMETER_THRESHOLD}mm)"
            )
            large_metrics = evaluate_on_samples(
                model, large_val, mhd_files, device,
                label=f"[{name} seed{seed}] Large nodules (>={DIAMETER_THRESHOLD}mm)"
            )

            raw_results.append({
                'variant'       : name,
                'seed'          : seed,
                'params'        : total_params,
                'best_val_loss' : best_val_loss,
                'dice'          : overall_metrics['Dice'],
                'iou'           : overall_metrics['IoU'],
                'precision'     : overall_metrics['Precision'],
                'recall'        : overall_metrics['Recall'],
                'small_dice'    : small_metrics['Dice'] if small_metrics else float('nan'),
                'large_dice'    : large_metrics['Dice'] if large_metrics else float('nan'),
            })

    all_results = existing_results + raw_results
    raw_df = pd.DataFrame(all_results)
    raw_df = raw_df.drop_duplicates(subset=['variant', 'seed'], keep='last')
    raw_df = raw_df.sort_values(['variant', 'seed']).reset_index(drop=True)
    raw_df.to_csv(RAW_CSV_PATH, index=False)
    print(f"\nRaw per-seed results ({len(raw_df)} rows) saved to: {RAW_CSV_PATH}", flush=True)

    # ─── Aggregate mean ± std across seeds ───────────────────────────────────
    metric_cols = ['best_val_loss', 'dice', 'iou', 'precision', 'recall', 'small_dice', 'large_dice']
    summary_rows = []
    for name in MODEL_VARIANTS:
        variant_rows = raw_df[raw_df['variant'] == name]
        row = {
            'variant' : name,
            'params'  : variant_rows['params'].iloc[0],
            'n_seeds' : len(variant_rows),
        }
        for col in metric_cols:
            row[f'{col}_mean'] = variant_rows[col].mean()
            row[f'{col}_std']  = variant_rows[col].std(ddof=1)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    csv_path = RESULTS_DIR / "model_comparison_results.csv"
    summary_df.to_csv(csv_path, index=False)

    # ─── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*100}", flush=True)
    print(f"Model Comparison Summary (mean ± std across {len(SEEDS)} seeds)", flush=True)
    print(f"{'='*100}", flush=True)
    header = f"{'Variant':<22}{'Dice':<20}{'IoU':<20}{'Small Dice':<20}{'Large Dice':<20}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for row in summary_rows:
        dice_str  = f"{row['dice_mean']:.4f} ± {row['dice_std']:.4f}"
        iou_str   = f"{row['iou_mean']:.4f} ± {row['iou_std']:.4f}"
        small_str = f"{row['small_dice_mean']:.4f} ± {row['small_dice_std']:.4f}"
        large_str = f"{row['large_dice_mean']:.4f} ± {row['large_dice_std']:.4f}"
        print(f"{row['variant']:<22}{dice_str:<20}{iou_str:<20}{small_str:<20}{large_str:<20}", flush=True)

    best_name = max(summary_rows, key=lambda r: r['dice_mean'])['variant']
    print(f"\nBest overall mean Dice: {best_name}", flush=True)
    print(f"\nAggregated results saved to: {csv_path}", flush=True)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    run()
