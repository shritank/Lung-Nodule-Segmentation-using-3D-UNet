# Lung Nodule Segmentation — Architecture Comparison

A seed-controlled, paired comparison of four 3D U-Net variants — plain, fully attention-gated,
selectively attention-gated, and squeeze-excitation residual — for pulmonary nodule segmentation
on [LUNA16](https://luna16.grand-challenge.org/). All four architectures are trained under an
identical protocol across the **same five random seeds**, paired by seed, so that any difference
between architectures can be separated from ordinary run-to-run variance.

**Finding:** seed-to-seed variance in Dice score exceeds the largest architectural difference by
**2.65×**. Across 42 pairwise statistical comparisons, one result reached nominal significance —
and it did not survive correction for multiple comparisons. The takeaway is not that these
attention/residual mechanisms are ineffective, but that reporting a single training run per
architecture — the norm in this literature — is not sufficient to tell architectures apart from
noise.

## Requirements

- Python 3.x
- PyTorch with CUDA (an NVIDIA GPU is required — the training script will not run on CPU)
- NumPy, pandas, SciPy
- SimpleITK (reading LUNA16's `.mhd`/`.raw` volumes)
- Matplotlib (figure generation only)

## Dataset setup

This project uses [LUNA16](https://luna16.grand-challenge.org/) (`subset0`–`subset9`,
`annotations.csv`, and the provided lung segmentation masks). The dataset is not included in this
repository.

A handful of paths are hardcoded at the top of `main.py` and `train.py` and **must be edited**
before running anything:

| Variable | File | Purpose |
|---|---|---|
| `DATA_DIR` | `main.py` | Root of the LUNA16 dataset (expects `subset0`–`subset9`, `annotations.csv`, `seg-lungs-LUNA16`) |
| `CACHE_DIR` | `main.py` | Where preprocessed `.npy` patches/labels are cached |
| `SAVE_DIR` | `train.py` | Checkpoint directory for the single-model legacy trainer |

`compare_models.py` derives its own per-architecture checkpoint directories
(`checkpoints_<variant>/`) relative to the project root and needs no separate edit.

## Project structure

```
main.py                          Preprocessing: HU normalization, lung masking, patch
                                  extraction, ground-truth mask synthesis, disk caching
train.py                         Dataset class, augmentation, loss functions, and a
                                  legacy single-model training loop (kept for reference)
compare_models.py                THE EXPERIMENT — trains all 4 variants x 5 seeds,
                                  evaluates each, writes results/ (resumable)
evaluation.py                    Confusion-matrix and metric computation (Dice, IoU,
                                  precision, recall)
significance_test.py             Paired t-tests across all architecture pairs and
                                  metrics (42 comparisons)
smoke_test.py                    Fast sanity check (CUDA, data wiring, one forward/
                                  backward pass per model) before a multi-hour run

model_base.py                    Plain 3D U-Net (baseline)
model_att.py                     Attention gates on all 3 skip connections
model_selectiveatt.py            Attention gates on the 2 deepest skips only
                                  (matches the original Attention U-Net's placement)
model_se_residual.py             Residual blocks with squeeze-excitation channel
                                  recalibration; no attention gates

plot_losses.py                   Renders loss-curve figures from saved per-seed
                                  loss histories (no retraining required)
generate_architecture_diagram.py Draws the shared U-Net skeleton, with every channel
                                  count/spatial dimension introspected live from the
                                  instantiated model (nothing hand-typed)
generate_variant_blocks_diagram.py  4-panel figure contrasting the block-level and
                                  skip-connection differences between all 4 variants
generate_qualitative_figures.py  Preprocessing example + per-architecture segmentation
                                  comparison, using real trained checkpoints

results/
  model_comparison_raw_results.csv    20 rows — one per (variant, seed): params,
                                       val loss, Dice, IoU, precision, recall,
                                       size-stratified Dice
  model_comparison_results.csv        Aggregated mean +/- std per variant
  pairwise_significance_results.csv   42 paired t-test results (all architecture
                                       pairs x all metrics)
  loss_history/<variant>_seed<N>.csv  Per-epoch train/val loss, one file per run
  loss_curves_per_variant.png
  val_loss_comparison.png

context.md                       Internal working notes / project history — not
                                  polished documentation, kept for reference
```

## What's not in this repository

- **The paper itself** (LaTeX source, figures, bibliography) — kept local, not published here.
- **Trained model checkpoints** (`checkpoints_<variant>/*.pth`) — excluded due to file size; the
  full numeric results of every run are preserved in `results/`, so nothing quantitative is lost,
  but the weights themselves are not distributed through this repo.

## Running the pipeline

```bash
python main.py              # build the preprocessing cache (one-time, slow)
python smoke_test.py        # quick sanity check before committing to a long run
python compare_models.py    # the actual experiment: 20 runs, resumable if interrupted
python significance_test.py # paired t-tests over the results
python plot_losses.py       # loss-curve figures
```

The three `generate_*_diagram.py` / `generate_qualitative_figures.py` scripts render the paper's
figures and are independent of each other; the qualitative-figures script additionally requires
the trained checkpoints from `compare_models.py` to exist.

## License

See [LICENSE](LICENSE).
