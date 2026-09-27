import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from model_base import UNet3D
from main import (
    build_scan_index, build_samples, load_ct_scan, load_lung_mask,
    normalize_hu, apply_lung_mask, world_to_voxel,
    extract_patch, create_nodule_mask, make_cache_key,
    ANNOT_PATH, SEG_DIR, CACHE_DIR
)

# ─── Config ───────────────────────────────────────────────────────────────────
PATCH_SIZE   = 64
BATCH_SIZE   = 2
NUM_EPOCHS   = 50
LR           = 1e-4
VAL_SPLIT    = 0.2
POS_WEIGHT   = 10.0
NUM_WORKERS  = 0
SAVE_DIR     = Path(r"C:\Users\ramar\OneDrive\Desktop\Lung Nodule Analysis\checkpoints")

# ─── Augmentation ─────────────────────────────────────────────────────────────
def augment_patch(patch, label):
    """
    Apply random 3D augmentations to patch and label simultaneously.
    All operations are numpy-based, applied before tensor conversion.
    patch and label are both (D, H, W) float32 numpy arrays.
    """
    # Random flips along each axis
    if np.random.random() > 0.5:
        patch = np.flip(patch, axis=0).copy()
        label = np.flip(label, axis=0).copy()
    if np.random.random() > 0.5:
        patch = np.flip(patch, axis=1).copy()
        label = np.flip(label, axis=1).copy()
    if np.random.random() > 0.5:
        patch = np.flip(patch, axis=2).copy()
        label = np.flip(label, axis=2).copy()

    # Random 90° rotations in the axial plane (z fixed, rotate in H-W plane)
    k = np.random.randint(0, 4)
    if k > 0:
        patch = np.rot90(patch, k=k, axes=(1, 2)).copy()
        label = np.rot90(label, k=k, axes=(1, 2)).copy()

    # Intensity jitter — patch only, never label
    if np.random.random() > 0.5:
        patch = patch + np.random.uniform(-0.05, 0.05)
        patch = np.clip(patch, 0.0, 1.0)

    if np.random.random() > 0.5:
        patch = patch * np.random.uniform(0.9, 1.1)
        patch = np.clip(patch, 0.0, 1.0)

    return patch.astype(np.float32), label.astype(np.float32)


# ─── Dataset ──────────────────────────────────────────────────────────────────
class LunaDataset(Dataset):
    def __init__(self, samples, mhd_files, patch_size=PATCH_SIZE, augment=False):
        self.samples    = samples
        self.mhd_files  = mhd_files
        self.patch_size = patch_size
        self.augment    = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample     = self.samples[idx]
        uid        = sample['uid']
        cache_name = make_cache_key(uid, sample['world_x'], sample['world_y'], sample['world_z'])
        patch_path = CACHE_DIR / f"{cache_name}_patch.npy"
        label_path = CACHE_DIR / f"{cache_name}_label.npy"

        if patch_path.exists() and label_path.exists():
            patch = np.load(patch_path)
            label = np.load(label_path)
        else:
            ct_array, origin, spacing = load_ct_scan(self.mhd_files[uid])
            mask_array                = load_lung_mask(uid)
            ct_array                  = normalize_hu(ct_array)
            ct_array                  = apply_lung_mask(ct_array, mask_array)

            world_coords = np.array([sample['world_x'], sample['world_y'], sample['world_z']])
            voxel_coords = world_to_voxel(world_coords, origin, spacing)
            voxel_zyx    = voxel_coords[::-1]

            patch = extract_patch(ct_array, voxel_zyx, self.patch_size)
            label = create_nodule_mask(self.patch_size, sample['diameter'], spacing)

        # Augmentation applied only when self.augment=True (training set only)
        if self.augment:
            patch, label = augment_patch(patch, label)

        patch = torch.tensor(patch).unsqueeze(0)
        label = torch.tensor(label).unsqueeze(0)
        return patch, label


# ─── Loss Functions ───────────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)
        pred_flat   = predictions.view(-1)
        target_flat = targets.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """Dice Loss + BCE Loss with positive voxel weighting."""
    def __init__(self, pos_weight=POS_WEIGHT):
        super().__init__()
        self.dice       = DiceLoss()
        self.pos_weight = pos_weight

    def forward(self, predictions, targets):
        dice_loss = self.dice(predictions, targets)

        weights  = torch.ones_like(targets)
        weights[targets > 0.5] = self.pos_weight
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            predictions, targets, weight=weights
        )

        return dice_loss + bce_loss


# ─── Training ─────────────────────────────────────────────────────────────────
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("\nBuilding dataset...")
    mhd_files   = build_scan_index()
    annotations = pd.read_csv(ANNOT_PATH)
    samples     = build_samples(annotations, mhd_files)
    print(f"Total samples: {len(samples)}")

    val_size   = int(len(samples) * VAL_SPLIT)
    train_size = len(samples) - val_size
    train_subset, val_subset = random_split(
        samples,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_samples = [samples[i] for i in train_subset.indices]
    val_samples   = [samples[i] for i in val_subset.indices]
    print(f"Train samples: {len(train_samples)} | Val samples: {len(val_samples)}")

    # Augmentation enabled for training set, disabled for validation
    train_dataset = LunaDataset(train_samples, mhd_files, augment=True)
    val_dataset   = LunaDataset(val_samples,   mhd_files, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model     = UNet3D().to(device)
    criterion = CombinedLoss(pos_weight=POS_WEIGHT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    scaler = torch.amp.GradScaler('cuda')

    best_val_loss = float('inf')

    print("\nStarting training...\n")

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
                print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

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

        print(f"\n-- Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} --\n")

        scheduler.step(avg_val_loss)

        checkpoint = {
            'epoch'          : epoch + 1,
            'model_state'    : model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'train_loss'     : avg_train_loss,
            'val_loss'       : avg_val_loss
        }
        torch.save(checkpoint, SAVE_DIR / f"checkpoint_epoch{epoch+1}.pth")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, SAVE_DIR / "best_model.pth")
            print(f"  Best model saved (val loss: {best_val_loss:.4f})")

    print("\nTraining complete.")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {SAVE_DIR}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    train()