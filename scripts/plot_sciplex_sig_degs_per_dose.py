"""violin plot of # fdr significant degs per dose for the full sci-plex3 dataset

for each (cell_type product_name dose) group with at least 5 cells runs welch's
t-test on log1p normalized expression vs the vehicle cells of that cell type
applies bh correction across all genes and counts genes with adj p < 0.05
plots one violin per dose with cell type as hue
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy import stats as _stats
from scipy.stats import false_discovery_control


def bh(pvals: np.ndarray) -> np.ndarray:
    """bh fdr correction"""
    pvals = np.asarray(pvals, dtype=np.float64)
    pvals = np.nan_to_num(pvals, nan=1.0)
    return false_discovery_control(pvals, method="bh")


def group_stats(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """mean variance and count across rows"""
    if sp.issparse(mat):
        mean = np.asarray(mat.mean(axis=0)).ravel()
        # E[X^2] - E[X]^2
        sq = mat.multiply(mat) if hasattr(mat, "multiply") else mat ** 2
        mean_sq = np.asarray(sq.mean(axis=0)).ravel()
        var = np.maximum(mean_sq - mean ** 2, 0.0)
        n = int(mat.shape[0])
        # unbiased variance
        if n > 1:
            var = var * n / (n - 1)
        return mean, var, n
    m = np.asarray(mat)
    return m.mean(axis=0), m.var(axis=0, ddof=1), int(m.shape[0])


def welch_t(mean_a, var_a, n_a, mean_b, var_b, n_b):
    """welch's t-statistic and df per gene"""
    se2 = var_a / n_a + var_b / n_b
    se2 = np.where(se2 <= 0, np.finfo(np.float64).tiny, se2)
    t = (mean_a - mean_b) / np.sqrt(se2)
    df_num = se2 ** 2
    df_den = (var_a / n_a) ** 2 / max(n_a - 1, 1) + (var_b / n_b) ** 2 / max(n_b - 1, 1)
    df_den = np.where(df_den <= 0, np.finfo(np.float64).tiny, df_den)
    df = df_num / df_den
    return t, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", type=Path, default=Path("/nfs/roberts/scratch/pi_sk2433/dac227/FlatCFM/artifacts/data/sciplex_subsample_underrep_k562_seed42_subseed0_n600000_ae100000_strat-cell_type-vehicle.h5ad"))
    ap.add_argument("--min-cells", type=int, default=5)
    ap.add_argument("--target-sum", type=float, default=10000.0)
    ap.add_argument("--fdr-alpha", type=float, default=0.05)
    ap.add_argument("--out-png", type=Path, default=Path("figures/sciplex3_sig_degs_per_dose_violin.png"))
    ap.add_argument("--out-tsv", type=Path, default=Path("figures/sciplex3_sig_degs_per_dose.tsv"))
    args = ap.parse_args()

    t0 = time.time()
    print(f"loading {args.h5ad}")
    adata = ad.read_h5ad(args.h5ad)
    print(f"  shape {adata.shape}  loaded in {time.time()-t0:.1f}s")

    print("normalizing target_sum + log1p")
    sc.pp.normalize_total(adata, target_sum=args.target_sum)
    sc.pp.log1p(adata)

    X = adata.X
    obs = adata.obs
    ctrl_mask = obs["vehicle"].astype(float).to_numpy() == 1.0
    print(f"  n_ctrl {int(ctrl_mask.sum())}  n_pert {int((~ctrl_mask).sum())}")

    rows = []
    t_groups = time.time()
    cell_types = list(obs["cell_type"].cat.categories) if hasattr(obs["cell_type"], "cat") else sorted(obs["cell_type"].unique())
    for ct in cell_types:
        ct_mask = obs["cell_type"].to_numpy() == ct
        ctrl_idx = np.where(ct_mask & ctrl_mask)[0]
        if ctrl_idx.size < args.min_cells:
            continue
        ctrl_mat = X[ctrl_idx]
        ctrl_mean, ctrl_var, ctrl_n = group_stats(ctrl_mat)
        ct_frame = obs[ct_mask & ~ctrl_mask].copy()
        ct_frame["__row"] = np.where(ct_mask & ~ctrl_mask)[0]
        for (product, dose), frame in ct_frame.groupby(["product_name", "dose"], observed=True):
            idx = frame["__row"].to_numpy()
            if idx.size < args.min_cells:
                continue
            ref_mean, ref_var, ref_n = group_stats(X[idx])
            t, df = welch_t(ref_mean, ref_var, ref_n, ctrl_mean, ctrl_var, ctrl_n)
            p = 2.0 * _stats.t.sf(np.abs(t), df)
            p_adj = bh(p)
            n_sig = int((p_adj < args.fdr_alpha).sum())
            rows.append({"cell_type": ct, "product_name": str(product), "dose": float(dose),
                         "n_cells": int(idx.size), "n_sig_deg": n_sig})
    print(f"  per-group welch's t + bh took {time.time()-t_groups:.1f}s over {len(rows)} groups")

    df = pd.DataFrame(rows)
    df.to_csv(args.out_tsv, sep="\t", index=False)
    print(f"saved table {args.out_tsv}")

    # violin per dose split by cell type
    doses = sorted(df["dose"].unique())
    cts = sorted(df["cell_type"].unique())
    colors = {"K562": "#4477AA", "A549": "#EE6677", "MCF7": "#228833"}

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    positions = np.arange(len(doses))
    width = 0.26
    offsets = {ct: (i - (len(cts) - 1) / 2) * width for i, ct in enumerate(cts)}
    legend_handles = []
    for ct in cts:
        sub = df[df["cell_type"] == ct]
        data = [sub.loc[sub["dose"] == d, "n_sig_deg"].to_numpy() for d in doses]
        pos = [p + offsets[ct] for p in positions]
        parts = ax.violinplot(data, positions=pos, widths=width * 0.95, showmeans=False, showmedians=True, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(colors.get(ct, "gray"))
            body.set_edgecolor(colors.get(ct, "gray"))
            body.set_alpha(0.6)
        parts["cmedians"].set_color("black")
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=colors.get(ct, "gray"), alpha=0.6, label=f"{ct} (n drugs={sub['product_name'].nunique()})"))
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{int(d)} nM" for d in doses])
    ax.set_xlabel("dose")
    ax.set_ylabel("# FDR-significant DEGs (BH adj p < 0.05)")
    ax.set_title(f"sci-Plex3 — # significant DEGs per (cell_type, product_name) at each dose\nwelch's t-test vs vehicle, BH-corrected over {X.shape[1]} genes, n={len(df)} groups")
    ax.grid(alpha=0.2, axis="y")
    ax.set_yscale("symlog", linthresh=10)
    ax.legend(handles=legend_handles, loc="upper left")
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=150)
    print(f"saved plot {args.out_png}")

    # text summary
    print("\n=== median # sig DEGs per (cell_type, dose) ===")
    pivot = df.pivot_table(index="cell_type", columns="dose", values="n_sig_deg", aggfunc="median")
    print(pivot.round(1))


if __name__ == "__main__":
    main()
