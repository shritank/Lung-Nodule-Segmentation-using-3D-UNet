"""
Paired significance testing across the seeded model comparison runs.

Uses model_comparison_raw_results.csv (one row per variant x seed). Since the
same seeds were reused across all four architectures (identical data order /
augmentation stream per seed, only the architecture differs), a paired test
is appropriate — each seed acts as its own control across architectures.

Runs a paired t-test for every pair of architectures, on every metric,
across the 3 matched seeds. Does not require retraining.
"""
import pandas as pd
from itertools import combinations
from pathlib import Path
from scipy.stats import ttest_rel

PROJECT_DIR = Path(r"C:\Users\ramar\OneDrive\Desktop\Lung Nodule Analysis")
RESULTS_DIR = PROJECT_DIR / "results"
RAW_CSV     = RESULTS_DIR / "model_comparison_raw_results.csv"
OUT_CSV     = RESULTS_DIR / "pairwise_significance_results.csv"

METRICS = ['best_val_loss', 'dice', 'iou', 'precision', 'recall', 'small_dice', 'large_dice']
ALPHA   = 0.05


def main():
    df = pd.read_csv(RAW_CSV)
    variants = sorted(df['variant'].unique())

    rows = []
    for var_a, var_b in combinations(variants, 2):
        # Align on seed so the paired t-test compares matched runs
        a = df[df['variant'] == var_a].set_index('seed').sort_index()
        b = df[df['variant'] == var_b].set_index('seed').sort_index()

        for metric in METRICS:
            values_a = a[metric].to_numpy()
            values_b = b[metric].to_numpy()

            mean_diff = values_a.mean() - values_b.mean()
            t_stat, p_value = ttest_rel(values_a, values_b)

            rows.append({
                'variant_a'      : var_a,
                'variant_b'      : var_b,
                'metric'         : metric,
                f'{var_a}_mean'  : values_a.mean(),
                f'{var_b}_mean'  : values_b.mean(),
                'mean_diff'      : mean_diff,
                't_stat'         : t_stat,
                'p_value'        : p_value,
                'significant_0.05': p_value < ALPHA,
            })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUT_CSV, index=False)

    print(f"Paired t-tests across {df['seed'].nunique()} matched seeds, {len(variants)} architectures")
    print(f"Degrees of freedom per test: {df['seed'].nunique() - 1}")
    print(f"{'='*100}")

    for var_a, var_b in combinations(variants, 2):
        print(f"\n{var_a}  vs  {var_b}")
        print("-" * 80)
        subset = results_df[(results_df['variant_a'] == var_a) & (results_df['variant_b'] == var_b)]
        for _, row in subset.iterrows():
            flag = " *SIGNIFICANT*" if row['significant_0.05'] else ""
            print(f"  {row['metric']:<15} mean_diff={row['mean_diff']:+.4f}  t={row['t_stat']:+.3f}  p={row['p_value']:.4f}{flag}")

    n_sig = results_df['significant_0.05'].sum()
    n_total = len(results_df)
    print(f"\n{'='*100}")
    print(f"Significant results (p < {ALPHA}): {n_sig}/{n_total}")
    print(f"Results saved to: {OUT_CSV}")


if __name__ == "__main__":
    main()
