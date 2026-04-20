#!/usr/bin/env python
"""evaluate FM training-budget sweep: cos_lfc vs epoch count

loads predictions from FM models trained at four epoch budgets (50 100 200 500)
all in the same narrow MSE AE d=128 latent space and evaluates downstream
perturbation prediction metrics to diagnose FM overfitting

if held-out cos_lfc peaks before 500 epochs the FM is overfitting to training
perturbations (the AE achieves cos_lfc~0.34 but FM-in-latent gives ~0.14 at
500 epochs so overfitting is the prime suspect)

usage
  uv run python scripts/eval_fm_budget_sweep.py
"""

from __future__ import annotations

import pandas as pd

from flatcfm.analysis import run_benchmark_suite

EPOCH_BUDGETS = [50, 100, 200, 500]
EXPERIMENTS = [f"sciplex_fm_deg_budget_e{e}" for e in EPOCH_BUDGETS]

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

    all_metrics = METRICS + ["mmd"]
    rows = []
    for epochs, exp in zip(EPOCH_BUDGETS, EXPERIMENTS):
        exp_rows = per_group[per_group["model_name"] == exp]
        if exp_rows.empty:
            continue
        row = {"epochs": epochs, "experiment": exp}
        for metric in all_metrics:
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
    print("\nFM budget sweep in narrow MSE AE d=128 (unweighted mean over held-out groups):\n")
    print(df.round(4).to_string(index=False))
    print()
    print("direction reminder:")
    print("  mean_gene_w1    - lower is better (wasserstein per gene)")
    print("  w2_squared      - lower is better (distributional distance)")
    print("  cosine_log_fc   - higher is better (perturbation direction)")
    print("  mmd             - lower is better (distributional distance in PCA-50)")
    print()
    print("AE direct reconstruction baseline: cos_lfc ~ 0.34")
    print("FM at 500 epochs (prior run):       cos_lfc ~ 0.14")
    print("if cos_lfc peaks well before 500 epochs -> FM is overfitting")


if __name__ == "__main__":
    main()
