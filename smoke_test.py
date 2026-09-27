"""
Quick sanity check for compare_models.py — verifies CUDA, imports, dataset
wiring, and a single forward/backward step for every model variant.
Runs in seconds/minutes, not hours. Does not write any checkpoints.
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import pandas as pd

from main import build_scan_index, build_samples, ANNOT_PATH
from train import LunaDataset, CombinedLoss, POS_WEIGHT, NUM_WORKERS
from torch.utils.data import DataLoader

from model_base import UNet3D as BaseUNet3D
from model_att import UNet3D as AttentionUNet3D
from model_selectiveatt import UNet3D as SelectiveAttentionUNet3D
from model_se_residual import UNet3D as SEResidualUNet3D

MODEL_VARIANTS = {
    "base"                : BaseUNet3D,
    "attention"           : AttentionUNet3D,
    "selective_attention" : SelectiveAttentionUNet3D,
    "se_residual"         : SEResidualUNet3D,
}

NUM_CHECK_SAMPLES = 4  # just enough to run one real batch through each model


def check_cuda():
    print("=" * 50)
    print("1. CUDA check")
    print("=" * 50)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    print(f"  CUDA available — GPU: {torch.cuda.get_device_name(0)}")


def check_dataset():
    print("\n" + "=" * 50)
    print("2. Dataset wiring check")
    print("=" * 50)
    mhd_files   = build_scan_index()
    print(f"  Found {len(mhd_files)} scans")

    annotations = pd.read_csv(ANNOT_PATH)
    samples     = build_samples(annotations, mhd_files)
    print(f"  Built {len(samples)} samples")

    small_samples = samples[:NUM_CHECK_SAMPLES]
    dataset = LunaDataset(small_samples, mhd_files, augment=False)
    loader  = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=NUM_WORKERS)

    patches, labels = next(iter(loader))
    print(f"  Loaded one batch — patches: {tuple(patches.shape)}, labels: {tuple(labels.shape)}")
    return small_samples, mhd_files


def check_models(device, small_samples, mhd_files):
    print("\n" + "=" * 50)
    print("3. Model forward/backward check")
    print("=" * 50)

    dataset = LunaDataset(small_samples, mhd_files, augment=False)
    loader  = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=NUM_WORKERS)
    patches, labels = next(iter(loader))
    patches, labels = patches.to(device), labels.to(device)

    criterion = CombinedLoss(pos_weight=POS_WEIGHT)

    for name, model_class in MODEL_VARIANTS.items():
        model = model_class().to(device)
        total_params = sum(p.numel() for p in model.parameters())

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        optimizer.zero_grad()

        predictions = model(patches)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        print(f"  [{name}] params: {total_params:,} | output: {tuple(predictions.shape)} | loss: {loss.item():.4f}  OK")


def main():
    check_cuda()
    device = torch.device("cuda")
    small_samples, mhd_files = check_dataset()
    check_models(device, small_samples, mhd_files)

    print("\n" + "=" * 50)
    print("All checks passed — compare_models.py should be safe to run.")
    print("=" * 50)


if __name__ == "__main__":
    main()
