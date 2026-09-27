# Project Context — Lung Nodule Segmentation Architecture Comparison Paper

**Purpose of this file:** complete handoff context so a new chat can continue this work without
asking setup questions. Read this fully before doing anything.

**Last updated:** end of the session that ran a full claim+reference verification pass and fixed
4 factual errors + 7 internal contradictions (see §8, entries 11-21, and §5 ref23 status).

---

## 0. TL;DR — what this project is

A conference/journal paper comparing **four 3D U-Net variants** for pulmonary nodule segmentation
on **LUNA16**, under a **seed-controlled paired protocol** (5 seeds × 4 architectures = 20 runs).

**The finding:** no architecture is significantly better. 1 of 42 paired comparisons was nominally
significant and it does not survive multiple-comparison correction. Seed-to-seed variance is
**2.65×** larger than the largest between-architecture difference. The paper's contribution is
**methodological rigour**, not a new architecture.

**Do not let the paper drift into a "we propose a better architecture" narrative.** That is the
single most important editorial constraint. See §7.

---

## 1. Environment and paths

| Thing | Path / value |
|---|---|
| Project root | `C:\Users\ramar\OneDrive\Desktop\Lung Nodule Analysis` |
| Paper source (canonical) | `paper/paper.tex` |
| Overleaf copy | `paper/main.tex` — **must be kept byte-identical to `paper.tex`** |
| Overleaf bundle | `paper/paper_for_overleaf.zip` (contains `main.tex` + `figures/`) |
| Figures | `paper/figures/*.png` |
| Results CSVs | `results/` |
| Per-run loss histories | `results/loss_history/<variant>_seed<N>.csv` (20 files) |
| Raw LUNA16 data | `D:\LUNA16` — **often unmounted; check before running anything that needs it** |
| Preprocessed patch cache | `C:\Users\ramar\LUNA16_cache` (`.npy` patch/label pairs) |
| OS / shell | Windows 11, Git Bash + PowerShell both available |
| GPU | NVIDIA RTX 4050 Laptop (paper says "Single NVIDIA consumer GPU" — user asked to keep it generic) |

**No LaTeX toolchain is installed locally.** `pdflatex`/MiKTeX/TeX Live are all absent. The paper
cannot be compiled here. The user compiles in **Overleaf**. Verification is therefore done by
script (see §6), never by compiling.

---

## 2. Code files and what they do

| File | Role |
|---|---|
| `main.py` | Data loading, preprocessing, patch extraction, cache. **Single source of truth** for `build_samples()`, `make_cache_key()`. Paths `DATA_DIR`, `CACHE_DIR`, `ANNOT_PATH` live here. |
| `model_base.py` | Plain 3D U-Net (`UNet3D`). ConvBlock = Conv3d→IN→ReLU ×2. |
| `model_att.py` | Attention U-Net variant. Gates **all three** skips. |
| `model_selectiveatt.py` | Gates only the **two deepest** skips (att2, att3); no `att1` module exists. |
| `model_se_residual.py` | ResidualSEBlock everywhere; no attention gates. |
| `train.py` | Original single-model training. Also defines `LunaDataset`, `CombinedLoss`, `augment_patch`, and the constants other scripts import. |
| `compare_models.py` | **The main experiment.** Trains all 4 variants × seeds {0,1,2,3,4}, evaluates, writes both result CSVs. Has skip-logic: reads `results/model_comparison_raw_results.csv` and skips any `(variant, seed)` already present. |
| `plot_losses.py` | Reads `results/loss_history/` → `loss_curves_per_variant.png`, `val_loss_comparison.png` (mean ± std bands). |
| `significance_test.py` | Paired two-tailed t-tests, all pairs × all metrics → `pairwise_significance_results.csv`. |
| `evaluation.py` | `get_confusion()`, `compute_metrics()` — reused by `compare_models.py`. |
| `smoke_test.py` | Fast sanity check (CUDA, one batch, one fwd/bwd per model). |
| `generate_qualitative_figures.py` | Fig 1 (preprocessing) + Fig 6 (qualitative segmentation). **Needs D: drive.** |
| `generate_architecture_diagram.py` | Fig 2 (U-Net skeleton). Introspects the real model. |
| `generate_variant_blocks_diagram.py` | Fig 3 (4-panel block comparison). Introspects all four models. |

**Run order for a full redo:** `compare_models.py` → `plot_losses.py` → `significance_test.py`.

---

## 3. Experimental setup (all values verified from code)

- **Dataset:** LUNA16, 888 CT scans across 10 subsets, derived from LIDC-IDRI.
- **Samples:** 1186 annotated nodules matched to available scans.
- **Split:** 80/20 → **949 train / 237 val**. Split seed fixed at **42**, deliberately independent
  of experiment seeds, so every run sees the identical partition.
- **Size stratification:** threshold **6.4 mm** (actual dataset median = 6.4336 mm — the paper says
  "approximately the median"). Gives **115 small / 122 large** validation nodules.
- **Preprocessing:** HU clip [-1000, 400] → scale to [0,1]; lung masks applied (all 888 scans have
  masks); world→voxel; 64³ patches, zero-padded at boundaries.
- **GROUND TRUTH — critical:** LUNA16 annotations give only centroid + diameter, so the target is a
  **synthesized binary sphere**, radius `(diameter_mm/2)/mean(spacing)`. This is disclosed
  prominently in §III-A. It means absolute Dice is **not** comparable to contour-based work, but
  relative architecture comparison stays valid (identical target for all four).
- **Architecture skeleton (all four share):** enc 32→64→128, bottleneck 256, MaxPool3d(2,2,2) down,
  ConvTranspose3d(2,2,2) up, final 1×1×1 conv → 1 channel. InstanceNorm throughout (batch size 2
  makes BatchNorm unreliable).
- **Params:** base 5,599,713 · attention 5,643,284 · selective_attention 5,641,155 ·
  se_residual 5,700,781. (Within 1.8% of each other → paper says "within 2%".)
- **Training:** Adam, lr 1e-4, ReduceLROnPlateau(factor 0.5, patience 3), 50 epochs, batch size 2,
  AMP. Loss = soft Dice + weighted BCE (pos_weight 10).
- **Augmentation (train only):** flips on all 3 axes (p=0.5 each), random 90° axial rotation,
  intensity shift U(−0.05,0.05) p=0.5, intensity scale U(0.9,1.1) p=0.5, clipped to [0,1].
- **Seeding:** `random`, `numpy`, `torch`, `torch.cuda` + a seeded DataLoader generator. Seeds
  {0,1,2,3,4} reused across all four architectures → **paired** design.
- **Evaluation:** best-val-loss checkpoint per run; sigmoid + 0.5 threshold; metrics
  **micro-averaged** over voxels (TP/FP/TN/FN accumulated then combined).
- **Stats:** paired two-tailed t-tests, matched on seed. 6 pairs × 7 metrics = **42 comparisons**,
  α = 0.05, df = 4.

---

## 4. The results (all verified against the CSVs — do not restate from memory, re-verify)

### Table III — mean ± std across 5 seeds

| Variant | Dice | IoU | Precision | Recall | Small Dice | Large Dice |
|---|---|---|---|---|---|---|
| base | 0.8139±0.0112 | 0.6863±0.0160 | 0.7960±0.0167 | 0.8328±0.0069 | 0.5712±0.1459 | 0.8487±0.0094 |
| attention | 0.7962±0.0176 | 0.6616±0.0246 | 0.7667±0.0301 | 0.8290±0.0234 | 0.5152±0.0971 | 0.8389±0.0096 |
| selective_attention | 0.8086±0.0230 | 0.6792±0.0331 | 0.7843±0.0459 | 0.8360±0.0138 | 0.5416±0.1494 | 0.8487±0.0057 |
| se_residual | **0.8224±0.0122** | **0.6985±0.0176** | **0.8181±0.0467** | 0.8291±0.0230 | **0.6194±0.1290** | 0.8486±0.0071 |

### Headline statistics

- **1 of 42** comparisons significant: `attention` vs `base` on **precision**, p = 0.0416,
  diff = −0.0292 (i.e. the *addition* is **worse** than baseline).
- **Bonferroni-adjusted p = 1.7474** → does not survive. **BH rejects nothing at any rank.**
- 2nd smallest p in the family = 0.0563.
- **Largest effect anywhere: d_z = 1.3232** (base vs attention, precision).
- **Minimum detectable d_z (n=5, α=.05, power=.80) = 1.6820.** → **Zero** of 42 comparisons reach
  it. The study is underpowered for every effect it observed.
- Largest architectural Dice gap = **0.02622** (se_residual − attention).
- 20 individual runs span **0.7783–0.8478 = 0.06948** → **2.650×** the architectural gap.
- Concrete illustration: `attention` (lowest mean, 0.7962) had a run at **0.8254**, beating
  `se_residual`'s *mean* (0.8224). Best single run overall = `selective_attention` seed 0 = 0.8478
  (third by mean).
- Small/large SD ratios: base 15.5× · attention 10.1× · selective 26.2× · se_residual 18.1×
  → paper says "10–26 times".
- **Brown–Forsythe p = 0.9882** → *no evidence* small-nodule variance is architecture-specific.

### Per-seed small-nodule Dice (drives the big SDs — outliers, not uniform spread)

```
base                {0:0.5468, 1:0.5193, 2:0.4653, 3:0.8267, 4:0.4979}
attention           {0:0.5473, 1:0.6631, 2:0.4286, 3:0.5071, 4:0.4299}
selective_attention {0:0.8002, 1:0.5024, 2:0.4460, 3:0.4346, 4:0.5246}
se_residual         {0:0.7257, 1:0.5323, 2:0.5268, 3:0.7910, 4:0.5211}
```

### Table V — epoch-wise val loss, se_residual vs each other (post-hoc, uncorrected)

| vs | Ep10 | Ep20 | Ep30 | Ep40 | Ep50 |
|---|---|---|---|---|---|
| base | 0.003* | 0.025* | 0.083 | 0.759 | 0.337 |
| attention | 0.003* | 0.009* | 0.005* | 0.010* | 0.881 |
| selective_attention | 0.002* | 0.034* | 0.674 | 0.849 | 0.138 |

Mean val loss at epoch 10: base 0.817 · attention 0.778 · selective 0.709 · **se_residual 0.361**.
At epoch 50 all converge to 0.2205–0.2285 and **zero** pairs differ significantly.

### The 3-seed pilot reversal (a key narrative beat)

An earlier 3-seed run produced **two** "significant" large-nodule Dice results
(base vs se_residual p=0.024; se_residual vs selective p=0.048). At 5 seeds these became
**p=0.983 and p=0.996**. This is used in the paper as an internal demonstration of its own thesis.

---

## 5. The paper — current state

**Format:** `\documentclass[conference]{IEEEtran}`. User's mentor has said to target a **journal**
instead (conference page limit 6–7 pages is too tight). Switching to `[journal,10pt]` is a pending
decision, not yet done.

**Length:** ~11–12 pages; user is watching page count closely and has asked for things to be
trimmed/compacted more than once.

### Structure (verified)

```
I.   Introduction
II.  Related Work            A. Backbones  B. Attention/SE  C. Nodule-seg literature  D. Reproducibility
III. Methods                 A. Dataset  B. Partitioning  C. Architectures  D. Training
                             E. Seed control  F. Evaluation/stats  G. Implementation
IV.  Results                 A. Overall  B. Significance  C. Training dynamics
                             D. Per-seed variance  E. Qualitative example
V.   Discussion              A. Multiple-comparison  B. Detectable threshold  C. Seed variance
                             D. Training budget  E. Small-nodule instability  F. Prior work
                             G. Practical implications
VI.  Limitations
VII. Conclusion
```

### Figures (source order determines number — all referenced via `\ref`, never hardcoded)

| # | Label | File | Width |
|---|---|---|---|
| 1 | `fig:preprocessing` | preprocessing_example.png | `\columnwidth` |
| 2 | `fig:architecture` | unet_architecture.png | `\columnwidth` (single-column `figure`, changed 2026-08-23 from `figure*` at `0.62\textwidth`) |
| 3 | `fig:blocks` | variant_blocks.png | `\textwidth` (`figure*`) |
| 4 | `fig:loss_curves` | loss_curves_per_variant.png | `0.85\textwidth` (`figure*`) |
| 5 | `fig:val_comparison` | val_loss_comparison.png | `\columnwidth` |
| 6 | `fig:qualitative` | segmentation_highlighting.png | `0.95\textwidth` (`figure*`) |

**Fig. 2 single-column conversion (2026-08-23):** `generate_architecture_diagram.py` was retuned
because moving from `figure*` (0.62\textwidth, ~4.4in) to single-column (\columnwidth, ~3.5in)
shrinks the display width, and text legibility is governed by fontsize (points, fixed) relative to
figsize (inches) -- LaTeX just rescales the whole raster to the target width. Fixes applied, each
verified by actually running the script and inspecting the rendered PNG (not guessed):
figsize reduced 13.5x8.0 -> 10.5x7.0in and every fontsize increased (labels 14->17, sublabels
12->14, arrow/skip labels 11->13, title 15->16) to raise the text-to-canvas ratio; this initially
caused two overlap bugs, both caught by rendering + cropping the actual PNG rather than by
calculation: (1) the "bottleneck: NNN ch" label overflowed its box at the larger font -- fixed by
giving the bottleneck box its own wider width (3.8 vs the common 2.3, later 2.5), kept centered on
the same midpoint so connecting arrows still land correctly; (2) taller boxes (box_h 1.05->1.3) ate
into the vertical gap between rows, causing the "MaxPool3d"/"ConvTranspose3d" arrow labels to
overlap the boxes above/below -- fixed by widening row spacing 2.5->3.4 data units. If this script
is touched again: **render it and inspect the actual PNG** (crop suspicious regions with PIL, view
via Read) rather than reasoning about fontsize/figsize/box-size interactions from first principles
-- the interactions are non-obvious enough that the first attempt here missed two real overflows.

### Tables

I architectures · II training config · III results · IV significance (42 p-values) ·
V epoch-wise · VI literature comparison

### References: 31, all verified against primary sources

ref1 Armato LIDC-IDRI · ref2 Setio LUNA16 · ref3 Ronneberger U-Net · ref4 Çiçek 3D U-Net ·
ref5 Milletari V-Net · ref6 Oktay Attention U-Net · ref7 Hu SE-Net · ref8 He ResNet ·
ref9 Isensee nnU-Net · ref10 Bouthillier variance · ref11 Picard seeds · ref12 Alhajim SEDARU-Net ·
ref13 Kingma Adam · ref14 Ulyanov InstanceNorm · ref15 Maier-Hein Metrics Reloaded ·
ref16 Wang HEE-SegGAN · ref17 Reinke pitfalls · ref18 Prithvika UNet/FPN · ref19 Liu CSEA-Net ·
ref20 Ma Dig-CS-VNet

**Added in the "more recent references" session (all 2024–2025, each verified against the primary
source via Europe PMC / CrossRef / publisher full text — not from search snippets):**

| Key | Work | Venue | Verified via |
|---|---|---|---|
| ref21 | Hu et al., MCAT-Net | Sci. Rep. 14:31743, 2024 | CrossRef + Europe PMC + **publisher PDF parsed locally** |
| ref22 | Liu et al., SCA-VNet | Electron. Res. Arch. 32(5):3016-3037, 2024 | CrossRef + **publisher PDF parsed locally** |
| ref23 | Krishnan O P & Roy, radial-attention U-Net | Eng. Res. Express 7(4):0452f9, 2025 | CrossRef + OpenAlex + **publisher PDF parsed locally** (user-supplied, verified 2026-08-22) |
| ref24 | Xi et al., CoreFormer | npj Digit. Med. 9:48, 2025 | CrossRef + Europe PMC + OpenAlex + **publisher PDF parsed locally** |

**Evidence tiers (important -- don't downgrade these by re-checking from search snippets):**
- **All four -- Tier 1.** Publisher PDFs downloaded and text-extracted with `pdfminer`; every
  Table VI cell asserted against the raw text by `scratchpad/verify_pdfs_local.py`
  (64 assertions, all passing). Nature PDFs came straight from `nature.com/articles/<doi>.pdf`;
  the AIMS one from `aimspress.com/.../era-32-05-138.pdf`; ref23's came from the user directly
  (`C:\Users\ramar\Downloads\Krishnan_O_P_2025_Eng._Res._Express_7_0452f9.pdf` -- ref23 is
  genuinely closed access, OpenAlex confirms `is_oa: false` with no repository copy, so this PDF
  had to come from the user's own institutional/personal access, not an automated fetch).
  Key verbatim confirmations: MCAT-Net results row `Ours LIDC-IDRI 88.29 79.10 86.33 90.39`;
  SCA-VNet `Dice coefficient of 87.50%` + IoU 77.80 in Table 4; radial-attention U-Net's Table 2
  five per-fold DSC values `0.8601 0.8627 0.8598 0.8663 0.8584` -> mean 0.8614/std 0.0030, and IoU
  values `0.7511 0.7537 0.7508 0.7502 0.7486` -> mean 0.7509/std 0.0019 (exactly the 86.1+/-0.3,
  75.1+/-0.2 used in Table VI), plus confirmation that its headline 86.33/88.09/83.89 figures live
  in a *different* table (Table 3, single 80:20 split) than the Table-2 pair actually used in
  Table VI -- so the two metrics reported together in Table VI are from the same experimental
  setting, not stitched across tables; CoreFormer `88.7 +/- 2.6` with "Results are reported as
  mean +/- Std **over the test set**" (so its +/- is across test cases, **not** across runs --
  this is why Table VI says Runs = 1 for it).
- Seed/CV sweeps confirm all four contain **zero** occurrences of "seed" as a repeated-training
  signal; ref23 uses 5-fold CV (not seeds) for its variance estimate; CoreFormer's only two
  "cross-validation" mentions are LNDb and k-means cluster selection, never LIDC-IDRI. This is
  what backs the "six of nine are single-run" claim in §V-F.

**Deliberately rejected candidates (do not re-add without new evidence):**
- *Multi-phase Swin/DeepLabv3+ framework*, Sci. Rep. 16:1652 (doi 10.1038/s41598-025-31147-2).
  Reports **IoU 97.88% and Jaccard 89.62% as separate metrics** — these are the same quantity, so
  the results table is self-contradictory. Two fetches of the full text also **disagreed** on
  whether it repeats runs over 3 seeds. Protocol columns are therefore unverifiable. Excluded.
- *TransAtenNet*, SN Comput. Sci. (doi 10.1007/s42979-025-04558-1) — Springer paywall, GT/runs/
  variance not verifiable.
- *arXiv 2505.17602* (Abdullah & Shaukat) — preprint; nodule-specific Dice/IoU not extractable.

**Re-verified while here:** ref20 (Ma et al.) Dice on LUNA16 **is 94.9%** (a search snippet claimed
89% — the snippet was wrong; the abstract says 94.9% LUNA16 / 81.1% LNDb). Table VI cell correct.

### The "24 -> 30 references" session (2026-08-19)

User asked to push from 24 to ~30 references. Found and verified **6** (hit the target exactly):

**Three statistical-methodology citations -- a real gap, not padding.** The paper had used
Bonferroni correction, the Benjamini-Hochberg procedure, and a Brown-Forsythe test extensively
(Sections V-A, V-E) without citing any of the three original papers -- exactly the kind of gap a
reviewer flags. All three verified via CrossRef, cited inline at first use (not in Related Work,
matching how Adam/InstanceNorm are handled -- method citations, not literature-review citations):

| Key | Work | Verified via |
|---|---|---|
| ref25 | Dunn 1961, "Multiple Comparisons Among Means," JASA 56(293):52-64 | CrossRef |
| ref26 | Benjamini & Hochberg 1995, JRSS-B 57(1):289-300 | CrossRef |
| ref27 | Brown & Forsythe 1974, JASA 69(346):364-367 | CrossRef |

**Three new domain papers, all Tier 1 (publisher PDF parsed locally, same standard as ref21/22/24):**

| Key | Work | Venue | Verified via |
|---|---|---|---|
| ref28 | Zhang et al., CA-3DTransUNet | Sci. Rep. 16:16033, 2026 | CrossRef + EuropePMC + **publisher PDF parsed locally** |
| ref29 | Gu et al., ShapeField-lung | npj Digit. Med. 8:736, 2025 | CrossRef + EuropePMC + **publisher PDF parsed locally** |
| ref30 | Raza et al., Trans RCED-UNet3+ | Front. Oncol. 15:1654466, 2025 | EuropePMC full-text XML (PMID 41179656, PMC12571626; Frontiers is CC-BY OA) |

**ref28 is the single most important new find in this batch.** Its Table 1 lists LUNA16 as
1186 total / **949 train / 237 validation** -- not just the same 888-scan/1186-nodule totals as
ref21/ref22, but the *exact same partition sizes* as this study. It reports 89.65% +/- 0.54% Dice
(5-fold CV, std across folds) on that identical split, vs our 82.2%. This is the cleanest
"same data, same split, different pipeline, very different number" data point in the whole table.

**ref29 gotcha caught during verification, worth remembering:** ShapeField-lung reports numbers on
*three* datasets (LIDC-IDRI primary, LUNA16 and Tianchi as **zero-shot cross-dataset transfer
tests**, "we train on LIDC-IDRI and test directly on two unseen datasets"). Its LUNA16 number
(85.8%) is a domain-transfer score, not an in-domain trained-and-evaluated result like every other
Table VI entry -- using it would have silently changed what the table measures for one row. Used
the LIDC-IDRI in-domain primary number (87.3% +/- 1.3%, 5-fold CV) instead, consistent with every
other row. Also notable: ShapeField-lung's LUNA16 experiment (not the one we cite) synthesizes
targets via centroid+diameter spheres "following the protocol in ref. 4" -- i.e. a *third* paper
using our exact ground-truth-construction method -- but this lives in their transfer-test setting,
not their headline number, so it wasn't pulled into the table; worth knowing if this paper's LUNA16
row is ever revisited.

**Rejected (checked, not added):** LNTransformer (CVPR 2025 workshop, `openaccess.thecvf.com`) --
both `WebFetch` (403) and a direct `curl` download were blocked/corrupted; no verifiable copy
obtainable, so excluded rather than cited from a search snippet.

### Prior-art check: "has someone already done our study?" (2026-08-19) -- ANSWERED, ref31 added

User asked directly whether an equivalent study already exists. Searched for each component of our
contribution separately and together (pulmonary nodule segmentation + multi-architecture comparison
+ seed-controlled paired design + significance/power analysis). **Conclusion: no exact match --
we are not scooped** -- but one paper is close enough that omitting it would have been a real
reviewer risk, so it is now **ref31**.

**ref31 -- {\AA}kesson, T\"oger & Heiberg, "Random effects during training: Implications for deep
learning-based medical image segmentation," Comput. Biol. Med. 180:108944, 2024,
doi 10.1016/j.compbiomed.2024.108944.** Verified via CrossRef (metadata) + the **full abstract
verbatim from Lund University Publications**, the authors' own institution
(`lup.lub.lu.se/record/cf0176e7-ae75-466b-9380-08041058a0c3`). OpenAlex says `is_oa: true`,
`oa_status: hybrid`, but **no PDF URL is registered anywhere** and ScienceDirect 403s automated
fetches -- so this is Tier 2 (abstract-verified), not Tier 1 (local PDF). Every number we state
about it comes from that verbatim abstract; **do not add any claim about its internals without
first obtaining the full text.**

**What it does (and why it does NOT scoop us):** it runs **one** learning algorithm (nnU-Net) under
**50 random seeds** on **brain tumour, hippocampus, and cardiac** segmentation -- *not* lung
nodules -- and finds the best seed significantly outperformed **0-76%** of the other seeds under
hold-out validation and **10-38%** under 5-fold CV, concluding a significant difference is "a weak
and unreliable indicator of a true performance difference between two learning algorithms."
So: **within-algorithm seed variance, one architecture, three non-pulmonary tasks.** Ours:
**between-architecture comparison, four architectures, seeds paired across them, pulmonary nodules,
plus a power/minimum-detectable-effect analysis they do not perform.** The two are complementary --
theirs bounds how far one algorithm differs from *itself*; ours asks whether four *different*
architectures differ from *each other* once that variation is held constant.

**Where cited:** §II-D (Reproducibility and Evaluation Methodology), positioned right after Picard
as the medical-imaging bridge between the general-ML variance literature and our task, with the
complementarity spelled out explicitly so a reviewer sees we know the distinction; and a compact
corroboration sentence in §V-F next to Bouthillier/Picard. **It strengthens rather than threatens
the paper** -- it is independent, larger-scale, external evidence for exactly what §V-C and the
3-seed pilot reversal argue.

**LaTeX note:** the accented names need escapes matching ref17's existing style --
`J.~{\AA}kesson` and `J.~T\"oger` (the preamble loads no `inputenc`/`fontenc`).

Two others checked and ruled out as not close enough: *"Accounting for Underspecification in
Statistical Claims of Model Superiority"* (arXiv 2511.02453) -- theoretical/methodological, not an
empirical architecture study, and an unpublished preprint; and *"Designing image segmentation
studies: Statistical power, sample size and reference standard quality"* (Med. Image Anal.) --
general power/sample-size methodology, not an architecture comparison.

### Table VI -- literature comparison (13 rows now; every cell verified from primary source)

| Method | Data | GT construction | Runs | Variance | Dice | IoU |
|---|---|---|---|---|---|---|
| UNet/FPN [18] | LIDC-IDRI (1018) | **Radiologist XML contours** | 1 | No | 71.6 | 58.6 |
| **This work** | LUNA16 (949/237) | Synthesized sphere (stated) | 5, paired | Yes ±std | **82.2±1.2** | **69.8±1.8** |
| HEE-SegGAN [16] | LUNA16 (6747/736) | Not stated | 10 | Boxplots only | 85.3 | 74.3 |
| Radial-attn U-Net [23] | LIDC-IDRI (~2048 2D patches) | pylidc consensus mask (≤4 radiologists) | 5-fold CV | Yes ±std across folds | 86.1±0.3 | 75.1±0.2 |
| ShapeField-lung [29] | LIDC-IDRI (count not stated; 60/20/20 per fold) | Consensus of ≥3 radiologists (majority vote) | 5-fold CV | Yes ±std across folds | 87.3±1.3 | — |
| SCA-VNet [22] | LIDC-IDRI (**888 scans / 1186 nodules**) | Radiologist annot.; rule not stated | 1 | Box plots across test cases | 87.5 | 77.8 |
| MCAT-Net [21] | LIDC-IDRI (**888 / 1186**; 9797 2D images) | Radiologist annot.; rule not stated | 1 | No | 88.3 | 79.1 |
| CoreFormer [24] | LIDC-IDRI (1018 scans, 2687 nodules) | Voxels with ≥3 annotator consensus | 1 | ±std across test cases | 88.7 | — |
| CA-3DTransUNet [28] | LUNA16 (**1186; 949/237 -- identical split to this work**) | Consensus mask via centroid clustering | 5-fold CV | Yes ±std across folds | 89.7±0.5 | — |
| CSEA-Net [19] | LUNA16 (808, 2D) | Not stated | 1 | No | 92.7 | 86.6 |
| Dig-CS-VNet [20] | LUNA16 (8746 cubes) | Annotation-file coords | 10-fold CV | No | 94.9 | 90.3 |
| SEDARU-Net [12] | LUNA16 | Not stated | 1 | No | 97.9 | — |
| Trans RCED-UNet3+ [30] | LIDC-IDRI (1018 scans, 1010 patients; 62,200 augmented) | Radiologist annot.; rule not stated | 1 | No | **99.0** | 98.1 |

Range 71.6–99.0 = **27.4 points** (was 26.3), ~10.5× our largest architectural gap (2.6 points).

**Three within-table statistics the paper now makes (all script-verified, recomputed from scratch,
not incrementally patched):**
1. **Target construction can't explain the spread -- now demonstrated more strongly than before.**
   The radiologist-derived group grew from 5 to **8 entries** ([18],[21],[22],[23],[24],[28],[29],
   [30]) and now **spans the entire 27.4-point range by itself** -- both the table's lowest value
   (71.6, [18]) and its highest (99.0, [30]) are radiologist-derived. The centroid+diameter pair
   still differs by 12.7; the "Not stated" trio still differs by 12.6. Since the radiologist-derived
   category alone reproduces the *full* range, target construction is demonstrably not the
   dominant factor -- a cleaner argument than the original "17.1 of 26.3 points" framing.
2. **A shared sample base doesn't make figures comparable -- now a 4-way comparison.** This work,
   [21], [22], and **[28]** all apply the LUNA16 screening criteria to land on the identical 888
   scans / 1186 nodules; **[28] additionally matches the exact 949/237 partition**, not just the
   totals. Four works, same data (one pair same split down to the nodule): 82.2 / 88.3 / 87.5 /
   89.7 -- a **7.5-point** spread = **2.9×** our largest architectural gap (was 6.1pt/2.3× with 3
   works).
3. **Variance is reported inconsistently.** Seven of the twelve prior works are single-run,
   *including the highest figure in the table* ([30], 99.0%, no variance reported at all). [20]
   does 10-fold CV with no spread; [16] runs 10× but shows boxplots only; [23]/[28]/[29] give ±std
   across **folds** (partition variance, 3 papers now, up from 1); [24] gives ±std across
   **test cases** (sampling variance). These are three different quantities, and **none of the
   twelve** reports numeric variance across repeated runs differing only in seed -- the quantity we
   measure. **This claim has now survived two expansions (9→12 works); re-verify again if more
   refs are added, since it's the load-bearing claim the whole table exists to support.**


### Abstract

**Written** (254 words). Summarises: 4 variants, 5 seeds/20 runs, 1-of-42 with no correction
surviving, d_z 1.32 < 1.68, 2.65× seed variance, pilot reversal, epoch-budget dependence,
10–26× small-nodule instability with no architecture-specificity, and the closing "these results do
not show the mechanisms to be ineffective" framing. Contains no citations (IEEE convention) and no
winner/SOTA language.

---

## 6. Verification workflow — **THIS IS THE MOST IMPORTANT PART**

The user has repeatedly demanded exhaustive verification and has caught me being overconfident.
**Never state a number from memory. Always recompute it from the CSVs/code before writing it.**

Verification scripts live in the scratchpad dir:
`C:\Users\ramar\AppData\Local\Temp\claude\C--Users-ramar-OneDrive-Desktop-Lung-Nodule-Analysis\<session-id>\scratchpad\`

**These will not exist in a new session — recreate them as needed.** They are:

| Script | Checks |
|---|---|
| `lint_check.py` | brace balance, `\begin`/`\end` balance, cite↔bibitem, raw `<`/`>` in text mode |
| `ref_check.py` | `\label` ↔ `\ref` resolution, figure files exist on disk |
| `round_audit.py` | every stated value equals the correctly-rounded actual value |
| `verify_new_sections.py` | Discussion/Limitations/Conclusion numbers |
| `verify_littable.py` | Table VI cells vs sources |
| `verify_blocks_fig.py` / `verify_arch_fig.py` | figure content vs live model introspection |
| `verify_abstract.py` | every abstract claim + no-winner-language check |
| `audit_full2.py` | hardcoded Fig/Table numbers, section-ref validity, structure |

### Two recurring false-alarm traps (I hit both; don't repeat them)

1. **LaTeX line wrapping.** A phrase split across source lines will fail a naive `in text` check.
   Always build `flat = ' '.join(text.split())` and test prose phrases against that.
2. **Stale assertions.** After intentionally changing wording, the old check still asserts the old
   state and "fails". Update the check to the new intended state — don't revert the paper.

Also: `re.findall(r'>{...')` flags the `>{\raggedright\arraybackslash}` column specs in tabular
preambles as "raw > in text mode". Those **10 hits are expected and fine**.

### After every edit

```bash
# 1. run lint + relevant verifiers
# 2. sync:
cd paper && cp paper.tex main.tex && diff -q paper.tex main.tex
# 3. rebuild zip (PowerShell):
Compress-Archive -Path "$src\main.tex","$src\figures" -DestinationPath $zip
```

---

## 7. Editorial rules — claims discipline (non-negotiable)

**Never write:**
- That any architecture "outperforms", "is superior", "achieves the best". A single-run version of
  this exact experiment produced a confident winner that vanished under repetition — that's the
  paper's cautionary point, not its result.
- That attention/SE "improve" small-nodule segmentation (not significant).
- Dice comparisons against published LUNA16 papers as if commensurable (different ground truth).
- That non-significance proves equivalence (it doesn't — the study is underpowered).

**Do write:** "did not yield a statistically significant improvement", "we did not observe",
"within the resolution of this experiment".

---

## 8. Discrepancies found and fixed (history — so they aren't reintroduced)

Major ones, all corrected:

1. **Attention U-Net mislabelling.** Oktay et al. explicitly do **not** gate the shallowest skip
   ("low-level feature-maps, i.e. the first skip connections, are not used in the gating function").
   So our `attention` variant (all 3 skips) **departs** from the published design, and
   `selective_attention` (2 deepest) is the one **matching** it. Originally the paper had this
   backwards. Also: Oktay's gate output is a bare element-wise product — our extra `W` output
   projection is **not** in the published formulation. Both now disclosed in §II-B and §III-C, and
   in the model docstrings.
2. **V-Net Dice attribution.** We claimed the ε=1e-6 smoothing was "following [ref5]". V-Net has
   **zero** occurrences of smooth/epsilon, uses a **squared** denominator, and **softmax** over two
   channels. Now stated as three explicit deviations, with the full equation given.
3. **Wrong Maier-Hein paper.** "Catalogue pitfalls" describes Reinke et al. (Nat. Methods 21:182–194),
   not the cited Metrics Reloaded (195–212). Added ref17.
4. **Unsupported contour claim.** Removed "including some of the related work cited in II-C" —
   neither ref12 nor ref16 states its GT construction. Verified.
5. **nnU-Net scope.** It configures 2D + 3D full-res + 3D cascade, not just "3D U-Net". Corrected.
6. **"Dominant factor" overclaim** in V-F — replaced with the quantified within-group spreads.
7. **Refs 18/19/20 first cited in Discussion** — moved introduction to §II-C.
8. **Contribution 3** presupposed variance *is* architecture-dependent while §V-E found no evidence.
   Reworded to "An assessment of **whether** run-to-run variance is itself architecture-dependent".
9. Numerous rounding/boundary fixes (e.g. IoU 69.9→69.8; "four of the five"→"three of the five").
10. **§V-F claim falsified by the four new refs and rewritten.** The paper had said the lowest
    figure (71.6%, [18]) "is the only one obtained against radiologist-drawn contour annotations
    from LIDC-IDRI; every higher figure either synthesizes the target from a centroid and diameter
    ... or does not state" its construction. Refs 21–24 are all radiologist-derived and all score
    **higher** than 71.6, so that sentence was no longer true. Replaced with the quantified
    17.1-point within-group spread, which makes the same point more strongly. Also updated every
    "five works" count to "nine works" (there were 5 such phrases, one of which — in §II-C
    line ~214 — was missed by the first verification pass and only caught by a follow-up grep).

**One I retracted:** I flagged "dominate" as overstating Bouthillier — it isn't. Their paper says
uncontrolled factors "dominate the meaningful difference between the methods". Left as-is.

### Full verification pass (2026-08-22) — 4 factual errors + 7 internal contradictions found and fixed

User asked for every claim and reference verified twice, then a full contradiction sweep. Method:
Table III/IV/V and every headline stat recomputed via two independent code paths (pandas/scipy,
and a from-scratch pure-python implementation including a hand-rolled incomplete-beta p-value
function) — **zero mismatches** across all 24 Table III values, all 42 Table IV p-values, all 15
Table V p-values. All 31 references checked against CrossRef metadata (21 DOIs) or CrossRef/arXiv
bibliographic search (the rest); content claims checked against full text for every reference where
OA full text was obtainable (10 via Europe PMC + ref22/ref23 via publisher/user-supplied PDF).
`D:\LUNA16`-dependent claims (median diameter, 115/122 split, Fig. 6 per-nodule values) were **not**
re-checked this session — user has verified those directly against the dataset many times and
asked to leave them alone.

**Factual errors found and fixed:**
11. **§III-C attention-gate description was wrong.** Said "all convolutions are followed by
    instance normalization," but ψ (the 1×1×1 conv feeding the sigmoid) has no InstanceNorm —
    confirmed by introspecting the live `AttentionGate` module and by the existing code comment
    in `model_selectiveatt.py` ("No InstanceNorm before sigmoid — matches original paper"). Fixed
    to name $W_g$, $W_x$, $W$ as normalized and state ψ feeds the sigmoid directly.
12. **§II-B unsupported claim about Oktay et al.** Said their visualized attention coefficients
    "are correspondingly drawn only from the coarser scales." Checked the arXiv full text twice
    (WebFetch + direct grep): Oktay et al. discuss AG behavior "at each scale" and say coarser-scale
    coefficients are "refined at finer resolutions" — the opposite of "only coarser scales." The
    other two claims in that sentence (shallowest skip ungated; gate output = plain elementwise
    product, no projection) verified correct and were left alone. Removed the false clause.
13. **Table VI, SEDARU-Net (ref12) IoU cell said `---`.** ref12's own text and results table report
    IoU 96.40% right alongside its 97.86% Dice. Filled in.
14. **Table VI + §V-F, Dig-CS-VNet (ref20) misclassified as centroid-and-diameter GT.** ref20's own
    text says labels were "drawn and generated according to the nodule **contour coordinates**
    stored in the annotation file" — centroid+diameter are used only to *crop* the volume, not to
    synthesize the mask. This meant the sentence "the two entries sharing this study's
    centroid-and-diameter construction differ by 12.7 points" was very likely wrong: this study is
    probably the *only* centroid-and-diameter entry in the table. Table VI's GT cell reworded to
    "Nodule contour coordinates from the annotation file"; the false pairing sentence in §V-F
    replaced with "This study is the only entry that synthesizes its target from a centroid and a
    diameter." Does not affect the paper's main §V-F argument (the radiologist-derived group alone
    still spans the full 27.4-point range).

**Internal contradictions found and fixed (none were factual errors on their own, but each
sentence conflicted with another sentence elsewhere in the paper):**
15. **§II-C, self-contradictory sentence about ref30.** Said it was "from a single training run
    with **no reported variance** … and, together with [23] and [29], one of the few entries … to
    **attach a numeric spread** to a headline figure at all" — internally contradictory, and also
    wrong about which three refs form the across-folds variance group (that's [23],[28],[29] per
    §V-F, not [23]/[29]+[30]). Removed the erroneous clause; §V-F already covers this completely.
16. **§II-C, "Seven further works … report intermediate values."** One of those seven is ref30 at
    99.0%, called "the highest figure among the twelve works" four lines later — not intermediate.
    Reworded to "also evaluate against radiologist-derived targets."
17. **§III-A, broken sentence.** "Third, and importantly for the target geometry is constructed
    identically…" was missing a word (should read "importantly, **because** the target geometry…").
    Fixed.
18. **§III-D vs §IV-C, apparent boundary contradiction.** Methods said the final-20-epoch val-loss
    change was "**less than** 0.032" while Results reports the range topping out at exactly
    "0.0320" (true value 0.0319966, correctly rounds to that). Both were technically true but read
    as contradictory side by side. Changed Methods to "**at most** 0.032".
19. **§III-D vs §V-D/§VI, budget-sensitivity contradiction.** Methods asserted "additional epochs
    would therefore be unlikely to alter the relative comparison," directly undercutting §V-D's
    entire point (the same runs give the opposite conclusion at 10-20 epochs) and §VI's Limitations
    entry ("a substantially longer budget … could in principle separate the architectures
    further"). Reworded to say the comparison is evaluated at a converged point and to point
    forward to §V-D/§VI for the budget-dependence caveat.
20. **Fig. 6 caption vs §IV-E text, ordering-inversion mismatch.** Caption said the per-nodule
    ordering "inverts" the five-seed ordering; §IV-E correctly says "close to the reverse" (a true
    inversion would swap `base` and `selective_attention`, which it doesn't). Caption corrected to
    match the (accurate) body text.
21. **§IV-D, unsupported availability claim.** "(available in the released raw results)" — nothing
    in the paper states anything is released, and the GitHub/repository sentence was deliberately
    removed from §III-G earlier (user won't reference a repo). Removed; the per-seed values are
    already enumerated inline in that same paragraph, so nothing is lost.

**Systematic contradiction sweep (beyond the 7 above) came back clean:** every repeated quantity —
2.65, d_z 1.32/1.68, 42 comparisons, 20 runs, S=5, Bonferroni 1.75, run span 0.7783–0.8478,
10.1–26.2×, 27.4-point literature range, 949/237/1186/888, params 5.60M/5.64M/5.70M, all epoch-wise
values — agrees across every section it appears in. No "outperform"/"superior"/"best" language
describes this study's own architectures (all such hits describe Picard, Åkesson, or the
early-stopping counterfactual). All three "equivalence" mentions correctly *deny* that
non-significance proves equivalence, matching the §7 editorial rule.

**ref23 verification (Krishnan O P & Roy, radial-attention U-Net) — now fully done.** User supplied
the PDF this session. Table 2 gives the five per-fold values used in Table VI: DSC
`0.8601 0.8627 0.8598 0.8663 0.8584` → mean **0.8614 ± 0.0030**; IoU
`0.7511 0.7537 0.7508 0.7502 0.7486` → mean **0.7509 ± 0.0019** — exactly the 86.1±0.3 / 75.1±0.2 in
Table VI. Also confirmed: "approximately 2,048 image and mask pairs," "up to four radiologists …
Pylidc … the consensus mask was extracted," 64×64 2D patches, zero random-seed mentions. Confirms
(again, independently of the prior session's check) that Table VI correctly uses the Table-2
5-fold pair rather than the abstract's 86.33%/88.09%/83.89% headline, which comes from a separate
80:20 single-split experiment in the same paper.

**Left alone deliberately (checked, judged fine or out of scope):**
- ref16's Table VI GT cell says "Not stated." ref16 does say it "generated the ground truth based
  on the point set described in the annotation file" — states the *source* but not the derivation
  *rule*, so "Not stated" is still defensible. Changing it would cascade into the "three entries
  whose construction is unstated differ by 12.6 points" claim in §V-F.
- §V-B's "approximately 0.041" detectable Dice difference reproduces exactly from
  MDE(1.682) × sd_diff(attention, se_residual) = 0.0408, i.e. the specific pair that produces the
  0.026 architectural gap — coherent but not spelled out in the text. The pair-averaged sd would
  give ~0.036 instead. Conclusion (0.041 > 0.026, or 0.036 > 0.026) holds either way; not changed.
- ref15 omits an issue number that ref17 (same journal, same year) includes. Cosmetic only.

---

## 9. Outstanding / next steps

**Blocked on the user:**
- **Title** — line 26 still has `[Draft title -- to be finalized]`. Working title:
  "An Empirical Comparison of Attention and Residual Mechanisms in 3D U-Net for Pulmonary Nodule
  Segmentation: A Seed-Controlled, Paired Analysis". User was offered alternatives and chose to
  keep it for now.
- **Author block -- DONE, and compile-error-free (confirmed via two real Overleaf error logs,
  not guessed).** Placeholder replaced with three authors: Shritan Kalidindi (1st, Dept. of
  Computing Technologies), Nalini Pusarla (2nd, Dept. of Computing Technologies), Avinash Vujji
  (3rd, Dept. of Networking and Communications) -- all SRM Institute of Science and Technology,
  Kattankulathur, Chennai, India. No email addresses given (none were provided; don't invent any).

  **Two failed attempts before landing on the working pattern -- read this before touching the
  author block again:**

  1. First attempt: wrapped two `minipage`s in a `\begin{tabular}{@{}c@{\hspace{2.2cm}}c@{}} ...
     & ... \end{tabular}` inside `\author{}`, to get author 3 under author 1 instead of centered.
     Failed on Overleaf: `Missing $ inserted` / `Missing \endgroup` / `Missing \cr inserted` /
     `Misplaced \cr`, reported several lines downstream (TeX error recovery lags the real cause).

  2. Second attempt: removed the `tabular`, kept `\IEEEauthorblockN`/`\IEEEauthorblockA` nested
     inside two side-by-side `minipage`s separated by `\hspace`. Looked safer (no raw `&`/`\` at
     the top level) but **also failed**, with a much longer cascade: `Missing \endgroup`,
     `Undefined control sequence` on `\endminipage`, `Misplaced \cr`/`\noalign`/`\crcr` inside
     macro traces reading `\@IEEEauthorblockNtopspaceline ->\cr ...` and
     `\@IEEEauthorblockAtopspaceline ->\cr ...`, eventually corrupting `\@maketitle` itself
     (`Extra }, or forgotten \endgroup` inside `\@maketitle`/`\@topnewpage`) and visibly garbling
     the rendered author names.

  **Root cause (confirmed from attempt 2's macro trace, not attempt 1's guess):**
  `\IEEEauthorblockN` and `\IEEEauthorblockA` are THEMSELVES implemented with `\halign`-based
  alignment internally (that's what `...topspaceline ->\cr` is). This means they can never be
  safely nested inside ANY custom box -- `minipage`, `tabular`, `parbox`, all equally unsafe --
  not just `tabular` specifically as the first fix assumed. Attempt 1's diagnosis was half right
  (identified `\halign` conflict) but mislocated it as an `\author`-level / `tabular`-specific
  problem, when it's actually inside the block macros themselves.

  **Working fix:** use IEEEtran's native, fully-supported pattern -- `\IEEEauthorblockN`/`A` pairs
  joined by `\and` at the top level of `\author{}`, nothing else wrapping them. To get author 3
  positioned under author 1 (left column) instead of centered as a lone leftover in row 2 (which is
  IEEEtran's default when an odd one wraps), add a 4th, **empty** `\IEEEauthorblockN{}`
  `\IEEEauthorblockA{}` pair as an invisible filler after author 3. With 4 blocks, IEEEtran's own
  auto-flow naturally lays out 2 per row (confirmed by the original screenshot itself, which showed
  4 real authors in exactly this 2x2 grid using the same institution/department text length) --
  landing author 3 under author 1 and the blank filler under author 2, using zero custom boxes and
  therefore zero conflict with the `\halign` internals. This is the current, working state of
  `paper.tex`/`main.tex`. **Never nest `\IEEEauthorblockN`/`\IEEEauthorblockA` inside a
  `minipage`/`tabular`/`parbox` again** -- if a future layout needs something `\and` + blank-filler
  can't express, don't use IEEEtran's block macros at all; build the block manually with plain
  centered text/`\textit` instead (bypassing `\IEEEauthorblockN/A` entirely avoids the `\halign`
  problem, at the cost of matching their exact font/spacing by hand).

  Still open: the page-13 concern from a longer author block hasn't been re-checked now that real
  names are in (short names, likely fine, but verify page count once compiled in Overleaf).


**Decisions pending:**
- Switch `\documentclass[conference]` → `[journal,10pt]` for the journal target.
- Page-count trimming if needed. Cheapest cuts: Table V (post-hoc, explicitly "not a competing
  result") or condensing Limitations.

**Recommended but not done:**
- **Bump to 10 seeds.** Highest-value remaining experiment — roughly halves the detectable-effect
  threshold (currently d_z=1.68), which is the limitation a reviewer is most likely to attack.
  `compare_models.py`'s skip-logic means only the new seeds retrain (~8h GPU). Change
  `SEEDS = [0,1,2,3,4]` to `[0,...,9]`, rerun the three scripts, then re-verify everything.
- Held-out test set / k-fold — better methodology but invalidates every current number and requires
  rewriting Results + Discussion. Currently disclosed honestly in §VI instead.

**Deliberately removed:** the GitHub/repository sentence in §III-G (user won't reference a repo).

**Target venues discussed:** Scientific Reports (direct precedent — ref12 is published there),
IEEE Access, Computers in Biology and Medicine. Advised *against* Medical Image Analysis / IEEE TMI
as first targets given the honest limitations.

---

## 10. Working style the user expects

- **Verify, don't assert.** Recompute every number. The user has explicitly asked "are you sure?"
  and been right to. When a check fails, work out whether the *paper* or the *check* is wrong.
- **Report discrepancies honestly**, including ones I introduced, and including retractions when I
  flag something that turns out to be fine.
- **Don't edit when asked only to review.** The user sometimes says "just check, don't change".
- **Figures must be generated from real data/models**, never hand-drawn numbers. All current
  figures introspect the live models or use real inference on the real cached patch.
- **Answer questions in chat** when asked, without editing files.
- Ask before large/expensive actions (e.g. installing MiKTeX, launching multi-hour training).
- The user pastes screenshots of the compiled PDF to report layout problems — expect to debug
  LaTeX layout blind, since we can't compile locally.

### Known LaTeX gotchas hit in this project

- `matplotlib` titles are **not** LaTeX — don't escape underscores as `\_` there (renders literally).
- IEEEtran uses `\flushbottom`; stretchy `\titlespacing` glue gets exploited to pad short columns,
  producing large gaps. Use rigid skips.
- `\includegraphics[width=\columnwidth]` locks *physical* size to ~3.5in regardless of PNG DPI —
  to make a figure visibly bigger you must switch to `figure*` and use `\textwidth` fractions.
- `\texttt{selective\_attention}` overflows narrow table columns; fix with
  `>{\raggedright\arraybackslash}p{...}` plus `\allowbreak` after the underscore.
- Overleaf: upload the **zip via New Project → Upload Project**; uploading it into an existing
  project just stores the zip unextracted. Entry file must be `main.tex`.
