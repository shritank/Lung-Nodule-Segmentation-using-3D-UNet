import SimpleITK as sitk
import numpy as np
import pandas as pd
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(r"D:\LUNA16")
SUBSET_DIRS = [DATA_DIR / f"subset{i}" for i in range(10)]
SEG_DIR    = DATA_DIR / "seg-lungs-LUNA16"
ANNOT_PATH = DATA_DIR / "annotations.csv"
CACHE_DIR = Path(r"C:\Users\ramar\LUNA16_cache")

# ─── CT Loading ───────────────────────────────────────────────────────────────
def load_ct_scan(mhd_path):
    """Load a CT scan from .mhd file. Returns array (Z,Y,X), origin, spacing."""
    itk_image = sitk.ReadImage(str(mhd_path))
    ct_array  = sitk.GetArrayFromImage(itk_image)   # (Z, Y, X)
    origin    = np.array(itk_image.GetOrigin())      # (X, Y, Z) in mm
    spacing   = np.array(itk_image.GetSpacing())     # (X, Y, Z) mm/voxel
    return ct_array, origin, spacing

# ─── Lung Mask ────────────────────────────────────────────────────────────────
def load_lung_mask(series_uid):
    """Load pre-computed lung segmentation mask. Returns None if not found."""
    seg_path = SEG_DIR / f"{series_uid}.mhd"
    if not seg_path.exists():
        return None
    itk_mask   = sitk.ReadImage(str(seg_path))
    mask_array = sitk.GetArrayFromImage(itk_mask)
    return mask_array

# ─── Preprocessing ────────────────────────────────────────────────────────────
def normalize_hu(ct_array, min_hu=-1000, max_hu=400):
    """Clip to lung-relevant HU range and normalize to [0, 1]."""
    ct_array = ct_array.clip(min_hu, max_hu)
    ct_array = (ct_array - min_hu) / (max_hu - min_hu)
    return ct_array.astype(np.float32)

def apply_lung_mask(ct_array, mask_array):
    """Zero out everything outside the lungs."""
    if mask_array is None:
        return ct_array
    return ct_array * (mask_array > 0).astype(np.float32)

# ─── Coordinate Conversion ────────────────────────────────────────────────────
def world_to_voxel(world_coords, origin, spacing):
    """Convert world coordinates (mm) to voxel indices (X,Y,Z)."""
    return np.round((world_coords - origin) / spacing).astype(int)

# ─── Patch Extraction ─────────────────────────────────────────────────────────
def extract_patch(ct_array, voxel_zyx, patch_size=64):
    """
    Extract a 3D patch of shape (patch_size, patch_size, patch_size)
    centered on voxel_zyx = (z, y, x). Zero-pads at boundaries.
    """
    z, y, x = voxel_zyx
    half      = patch_size // 2
    ct_padded = np.pad(ct_array, half, mode='constant', constant_values=0)
    z, y, x   = z + half, y + half, x + half
    return ct_padded[z-half:z+half, y-half:y+half, x-half:x+half]

# ─── Nodule Mask Creation ─────────────────────────────────────────────────────
def create_nodule_mask(patch_size, nodule_diameter_mm, spacing):
    """
    Create a spherical binary mask centered in the patch.
    Vectorized — no nested loops.
    """
    half   = patch_size // 2
    radius = (nodule_diameter_mm / 2.0) / np.mean(spacing)

    zz, yy, xx = np.mgrid[0:patch_size, 0:patch_size, 0:patch_size]
    dist = np.sqrt((zz - half)**2 + (yy - half)**2 + (xx - half)**2)

    mask = (dist <= radius).astype(np.float32)
    return mask

# ─── Scan Index ───────────────────────────────────────────────────────────────
def build_scan_index():
    """
    Scan all subsets and return a dict: {series_uid: Path_to_mhd}.
    """
    mhd_files = {}
    for subset_dir in SUBSET_DIRS:
        for f in Path(subset_dir).glob("*.mhd"):
            mhd_files[f.stem] = f
    return mhd_files

# ─── Sample List ──────────────────────────────────────────────────────────────
def build_samples(annotations, mhd_files):
    """
    Build list of sample dicts from annotations matched to downloaded scans.
    THIS IS THE SINGLE SOURCE OF TRUTH — both build_cache() and train.py
    must use this exact function so the sample list is always identical.
    """
    samples = []
    for _, row in annotations.iterrows():
        uid = row['seriesuid']
        if uid in mhd_files:
            samples.append({
                'uid'     : uid,
                'world_x' : row['coordX'],
                'world_y' : row['coordY'],
                'world_z' : row['coordZ'],
                'diameter': row['diameter_mm']
            })
    return samples

# ─── Cache Key ────────────────────────────────────────────────────────────────
def make_cache_key(uid, world_x, world_y, world_z):
    """
    Deterministic cache key. Rounds coordinates to 2 decimal places to avoid
    floating-point formatting mismatches between different code paths.
    THIS EXACT FUNCTION must be used everywhere a cache filename is built —
    never reconstruct the f-string separately, always call this.
    """
    return f"{uid}_{round(float(world_x), 2)}_{round(float(world_y), 2)}_{round(float(world_z), 2)}"

# ─── Cache Building ───────────────────────────────────────────────────────────
def build_cache():
    """
    Preprocess all scans once and save patches + labels as .npy files.
    Subsequent training runs load from cache instead of raw CT.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    annotations = pd.read_csv(ANNOT_PATH)
    mhd_files   = build_scan_index()
    samples     = build_samples(annotations, mhd_files)

    print(f"Building cache for {len(samples)} samples...")
    print(f"Cache directory: {CACHE_DIR}")

    skipped = 0
    saved   = 0
    failed  = 0

    for i, sample in enumerate(samples):
        uid = sample['uid']
        cache_name = make_cache_key(uid, sample['world_x'], sample['world_y'], sample['world_z'])
        patch_path = CACHE_DIR / f"{cache_name}_patch.npy"
        label_path = CACHE_DIR / f"{cache_name}_label.npy"

        if patch_path.exists() and label_path.exists():
            skipped += 1
            if i % 50 == 0:
                print(f"  [{i}/{len(samples)}] Skipping already cached: {uid[:30]}...")
            continue

        try:
            ct_array, origin, spacing = load_ct_scan(mhd_files[uid])
            mask_array                = load_lung_mask(uid)
            ct_array                  = normalize_hu(ct_array)
            ct_array                  = apply_lung_mask(ct_array, mask_array)

            world_coords = np.array([sample['world_x'], sample['world_y'], sample['world_z']])
            voxel_coords = world_to_voxel(world_coords, origin, spacing)
            voxel_zyx    = voxel_coords[::-1]

            patch = extract_patch(ct_array, voxel_zyx, patch_size=64)
            label = create_nodule_mask(64, sample['diameter'], spacing)

            np.save(patch_path, patch)
            np.save(label_path, label)
            saved += 1

            if i % 10 == 0:
                print(f"  [{i}/{len(samples)}] Saved: {uid[:30]}...")

        except Exception as e:
            print(f"  [{i}/{len(samples)}] FAILED: {uid[:30]}... Error: {e}")
            failed += 1

    print(f"\nCache complete.")
    print(f"  Saved  : {saved}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed : {failed}")
    print(f"  Total  : {saved + skipped} / {len(samples)} samples cached")

# ─── Verification ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("LUNA16 Data Verification")
    print("=" * 50)

    mhd_files = build_scan_index()
    print(f"\nTotal CT scans found across all subsets: {len(mhd_files)}")
    for i, subset_dir in enumerate(SUBSET_DIRS):
        count = len(list(Path(subset_dir).glob("*.mhd")))
        print(f"  subset{i}: {count} scans")

    annotations = pd.read_csv(ANNOT_PATH)
    print(f"\nTotal nodules in annotations.csv: {len(annotations)}")

    matched = annotations[annotations['seriesuid'].isin(mhd_files.keys())]
    print(f"Nodules matched to downloaded scans: {len(matched)}")
    print(f"Unique scans with nodules: {matched['seriesuid'].nunique()}")
    print(f"\nNodule diameter stats (mm):")
    print(f"  Min:    {matched['diameter_mm'].min():.1f}")
    print(f"  Max:    {matched['diameter_mm'].max():.1f}")
    print(f"  Mean:   {matched['diameter_mm'].mean():.1f}")
    print(f"  Median: {matched['diameter_mm'].median():.1f}")

    print("\n" + "=" * 50)
    print("Testing load on first matched scan...")
    first_uid = matched['seriesuid'].iloc[0]
    ct_array, origin, spacing = load_ct_scan(mhd_files[first_uid])
    print(f"  Series UID : {first_uid}")
    print(f"  CT shape   : {ct_array.shape}")
    print(f"  Origin     : {origin}")
    print(f"  Spacing    : {spacing}")
    print(f"  HU range   : {ct_array.min()} to {ct_array.max()}")

    mask = load_lung_mask(first_uid)
    if mask is not None:
        print(f"  Lung mask  : found, shape {mask.shape}")
    else:
        print(f"  Lung mask  : NOT found for this scan")

    ct_norm    = normalize_hu(ct_array)
    ct_masked  = apply_lung_mask(ct_norm, mask)

    nodule     = matched[matched['seriesuid'] == first_uid].iloc[0]
    world_coords = np.array([nodule['coordX'], nodule['coordY'], nodule['coordZ']])
    voxel_coords = world_to_voxel(world_coords, origin, spacing)
    voxel_zyx    = voxel_coords[::-1]

    patch = extract_patch(ct_masked, voxel_zyx)
    label = create_nodule_mask(64, nodule['diameter_mm'], spacing)

    print(f"\n  Patch shape      : {patch.shape}")
    print(f"  Patch value range: {patch.min():.3f} to {patch.max():.3f}")
    print(f"  Label shape      : {label.shape}")
    print(f"  Label voxels > 0 : {int(label.sum())} (nodule sphere)")

    print("\nBuilding patch cache...")
    build_cache()

    print("\nmain.py verified - ready to train")