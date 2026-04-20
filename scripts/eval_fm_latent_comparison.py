#!/usr/bin/env python
"""evaluate FM-in-latent comparison across 4 AE variants at matched dim

loads predictions from FM models trained in each latent space (linear AE
narrow MSE AE wide MSE AE PHATE AE at d=128) and runs run_benchmark_suite
with identical metrics and grouping so the four are directly comparable

prints a summary table of the four downstream perturbation prediction
metrics averaged over all held-out cell_type x product_dose groups the
metric that moves is the signal

usage
  uv run python scripts/eval_fm_latent_comparison.py
"""

from __future__ import annotations

import pandas as pd

from flatcfm.analysis import run_benchmark_suite

EXPERIMENTS = [
    "sciplex_fm_deg_latent_linear_d128",
    "sciplex_fm_deg_latent_mse_narrow_d128",
    "sciplex_fm_deg_latent_mse_wide_d128",
    "sciplex_fm_deg_latent_phate_d128",
]

METRICS = ["mean_gene_w1", "w2_squared", "cosine_log_fc"]
GROUP_COLUMNS = ["cell_type", "product_dose"]
REDUCTIONS = ["unweighted_mean"]


def main() -> None:
    result = run_benchmark_suite(
        {
            "experiments": EXPERIMENTS,
            "metrics": METRICS,
            "group_columns": GROUP_COLUMNS,
            "reductions": REDUCTIONS,
        }
    )
    per_group = result["per_group_metrics"]

    mmd_result = run_benchmark_suite(
        {
            "experiments": EXPERIMENTS,
            "metrics": ["mmd"],
            "group_columns": GROUP_COLUMNS,
            "reductions": REDUCTIONS,
            "metric_spaces": [
                {"name": "train_pca_50", "kind": "train_pca", "pca_n_components": 50},
            ],
        }
    )
    per_group = pd.concat(
        [per_group, mmd_result["per_group_metrics"]], ignore_index=True
    )

    rows = []
    for exp in EXPERIMENTS:
        exp_rows = per_group[per_group["model_name"] == exp]
        if exp_rows.empty:
            continue
        row = {"experiment": exp}
        for metric in METRICS + ["mmd"]:
            metric_rows = exp_rows[
                (exp_rows.get("metric_base", exp_rows.get("metric")) == metric)
            ]
            if metric_rows.empty:
                metric_rows = exp_rows[exp_rows["metric"] == metric]
            if metric_rows.empty:
                row[metric] = float("nan")
                continue
            row[metric] = float(metric_rows["value"].mean())
        rows.append(row)

    df = pd.DataFrame(rows)
    print("\nFM-in-latent comparison at d=128 (unweighted mean over held-out groups):\n")
    print(df.round(4).to_string(index=False))
    print()
    print("direction reminder:")
    print("  mean_gene_w1    - lower is better (wasserstein per gene)")
    print("  w2_squared      - lower is better (distributional distance)")
    print("  cosine_log_fc   - higher is better (perturbation direction)")
    print("  mmd             - lower is better (distributional distance in PCA-50)")


if __name__ == "__main__":
    main()
