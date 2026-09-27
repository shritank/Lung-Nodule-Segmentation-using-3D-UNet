"""
Generates two qualitative figures for the paper from real data and real
trained models — no synthetic/fabricated visuals:

  1. figures/preprocessing_example.png
     Raw HU-windowed CT slice vs. after normalize_hu + lung masking,
     centered on a real nodule.

  2. figures/segmentation_highlighting.png
     For the same nodule: ground-truth sphere vs. each of the four trained
     architectures' predicted probability map, on the central axial slice
     of the 64^3 patch. Uses each architecture's actual best_model.pth
     (seed 4, the checkpoint currently in checkpoints_<variant>/).

Reuses the exact sample-selection logic (seed 42 split) from train.py/main.py
so the chosen nodule is a genuine member of the paper's validation set.
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from main import (
    build_scan_index, build_samples, load_ct_scan, load_lung_mask,
    normalize_hu, apply_lung_mask, world_to_voxel, extract_patch,
    create_nodule_mask, ANNOT_PATH
)
from train import VAL_SPLIT, PATCH_SIZE

from model_base import UNet3D as BaseUNet3D
from model_att import UNet3D as AttentionUNet3D
from model_selectiveatt import UNet3D as SelectiveAttentionUNet3D
from model_se_residual import UNet3D as SEResidualUNet3D

PROJECT_DIR = Path(r"C:\Users\ramar\OneDrive\Desktop\Lung Nodule Analysis")
FIG_DIR     = PROJECT_DIR / "paper" / "figures"

MODEL_VARIANTS = {
    "base"                : BaseUNet3D,
    "attention"           : AttentionUNet3D,
    "selective_attention" : SelectiveAttentionUNet3D,
    "se_residual"         : SEResidualUNet3D,
}
DISPLAY_NAMES = {
    "base"                : "base",
    "attention"           : "attention",
    "selective_attention" : "selective_attention",
    "se_residual"         : "se_residual",
}


def select_reference_sample():
    """Rebuilds the exact seed-42 validation split and picks the
    median-diameter validation nodule as a representative example."""
    mhd_files   = build_scan_index()
    annotations = pd.read_csv(ANNOT_PATH)
    samples     = build_samples(annotations, mhd_files)

    val_size   = int(len(samples) * VAL_SPLIT)
    train_size = len(samples) - val_size
    _, val_subset = torch.utils.data.random_split(
        samples, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    val_samples = [samples[i] for i in val_subset.indices]

    val_sorted = sorted(val_samples, key=lambda s: s['diameter'])
    sample = val_sorted[len(val_sorted) // 2]
    print(f"Reference validation sample: uid={sample['uid'][-20:]}..., "
          f"diameter={sample['diameter']:.2f}mm", flush=True)
    return sample, mhd_files


def build_patch_and_label(sample, mhd_files):
    ct_array, origin, spacing = load_ct_scan(mhd_files[sample['uid']])
    mask_array = load_lung_mask(sample['uid'])

    raw_hu_array = ct_array.copy()  # keep pre-normalization HU values for Fig 1

    norm_array   = normalize_hu(ct_array)
    masked_array = apply_lung_mask(norm_array, mask_array)

    world_coords = np.array([sample['world_x'], sample['world_y'], sample['world_z']])
    voxel_coords = world_to_voxel(world_coords, origin, spacing)
    voxel_zyx    = voxel_coords[::-1]

    patch = extract_patch(masked_array, voxel_zyx, PATCH_SIZE)
    label = create_nodule_mask(PATCH_SIZE, sample['diameter'], spacing)

    raw_patch = extract_patch(raw_hu_array, voxel_zyx, PATCH_SIZE)

    return patch, label, raw_patch, voxel_zyx, ct_array.shape


def figure_1_preprocessing(sample, raw_patch, patch):
    """Before (raw HU) vs after (normalized + lung-masked) on the central
    axial slice of the extracted 64^3 patch centered on the nodule."""
    mid = PATCH_SIZE // 2

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    axes[0].imshow(raw_patch[mid], cmap='gray', vmin=-1000, vmax=400)
    axes[0].set_title("Before: raw HU (windowed for display)")
    axes[0].axis('off')

    axes[1].imshow(patch[mid], cmap='gray', vmin=0, vmax=1)
    axes[1].set_title("After: normalized + lung-masked")
    axes[1].axis('off')

    fig.suptitle(f"Preprocessing example (nodule diameter {sample['diameter']:.1f}mm)")
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / "preprocessing_example.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}", flush=True)


def load_best_model(variant_name, model_class, device):
    ckpt_path = PROJECT_DIR / f"checkpoints_{variant_name}" / "best_model.pth"
    model = model_class().to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f"  Loaded {variant_name}: seed={checkpoint.get('seed')}, "
          f"epoch={checkpoint.get('epoch')}, val_loss={checkpoint.get('val_loss'):.4f}", flush=True)
    return model


def figure_2_segmentation_highlighting(sample, patch, label, device):
    mid = PATCH_SIZE // 2
    patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))

    axes[0].imshow(patch[mid], cmap='gray', vmin=0, vmax=1)
    axes[0].contour(label[mid], levels=[0.5], colors='lime', linewidths=1.5)
    axes[0].set_title("Input + ground truth")
    axes[0].axis('off')

    print("Running inference for each architecture (best checkpoint, seed 4):", flush=True)
    for i, (variant_name, model_class) in enumerate(MODEL_VARIANTS.items(), start=1):
        model = load_best_model(variant_name, model_class, device)
        with torch.no_grad():
            logits = model(patch_tensor)
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

        ax = axes[i]
        ax.imshow(patch[mid], cmap='gray', vmin=0, vmax=1)
        ax.imshow(probs[mid], cmap='hot', alpha=0.5, vmin=0, vmax=1)
        ax.contour(label[mid], levels=[0.5], colors='lime', linewidths=1.2)
        ax.contour(probs[mid], levels=[0.5], colors='cyan', linewidths=1.2)
        ax.set_title(DISPLAY_NAMES[variant_name])
        ax.axis('off')

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    fig.suptitle(
        f"Predicted nodule probability by architecture (diameter {sample['diameter']:.1f}mm) — "
        f"green = ground truth, cyan = predicted 0.5 contour, heatmap = predicted probability"
    )
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / "segmentation_highlighting.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}", flush=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}", flush=True)

    sample, mhd_files = select_reference_sample()
    patch, label, raw_patch, voxel_zyx, scan_shape = build_patch_and_label(sample, mhd_files)

    figure_1_preprocessing(sample, raw_patch, patch)
    figure_2_segmentation_highlighting(sample, patch, label, device)

    print("\nDone. Both figures written to paper/figures/.", flush=True)
    print(f"Reference sample for reproducibility: uid={sample['uid']}, "
          f"world=({sample['world_x']:.3f},{sample['world_y']:.3f},{sample['world_z']:.3f}), "
          f"diameter={sample['diameter']:.4f}mm", flush=True)


if __name__ == "__main__":
    main()
