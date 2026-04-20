#!/usr/bin/env python
"""quick evaluation of varnorm AE vs baseline wide AE on DEG-restricted metrics

loads baseline wide, varnorm v1 broken floor, varnorm v2 fixed floor
evaluates each on
  - total var_explained (same as notebook aggregate metric)
  - DEG-restricted var_explained on top 500 perturbation-responsive genes
  - cosine_logfc on held-out perturbation groups

usage
  uv run python scripts/eval_varnorm_debug.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from flatcfm.data.space import load_ae_projection

RUNS_ROOT = pathlib.Path("artifacts/runs")
SPACES_ROOT = pathlib.Path("artifacts/spaces")


def _latest_run_cfg(experiment_name: str) -> OmegaConf | None:
    run_root = RUNS_ROOT / experiment_name
    if not run_root.is_dir():
        return None
    subdirs = sorted(d for d in run_root.iterdir() if d.is_dir())
    if not subdirs:
        return None
    return OmegaConf.load(subdirs[-1] / "run_config.yaml")


def _load_projection(tag: str):
    pkls = list(SPACES_ROOT.glob(f"sciplex_ae_projection_*_{tag}.pkl"))
    if not pkls:
        return None
    return load_ae_projection(pkls[0])


def main() -> None:
    # use the baseline wide d=512 config to instantiate a datamodule matching
    # the space all the AEs in this experiment were trained on
    cfg = _latest_run_cfg("sciplex_ae_deg_mse_wide_d512")
    if cfg is None:
        raise SystemExit("no baseline wide d=512 run found to pull datamodule config from")

    datamodule = instantiate(
        cfg.data,
        data=OmegaConf.to_container(cfg.data, resolve=True),
        splitter=OmegaConf.to_container(cfg.splitter, resolve=True),
        space=OmegaConf.to_container(cfg.space, resolve=True),
        evaluation_space=OmegaConf.to_container(cfg.evaluation_space, resolve=True),
        condition=OmegaConf.to_container(cfg.condition, resolve=True),
        paths=OmegaConf.to_container(cfg.paths, resolve=True),
        ae_geometry=OmegaConf.to_container(cfg.ae_geometry, resolve=True),
        task=OmegaConf.to_container(cfg.task, resolve=True),
        predict=OmegaConf.to_container(cfg.predict, resolve=True),
        trainer=OmegaConf.to_container(cfg.trainer, resolve=True),
        _recursive_=False,
    )
    datamodule.setup("fit")

    held_out_mask = np.asarray(datamodule.masks["is_held_out"], dtype=bool)
    test_adata = datamodule.adata_full[held_out_mask].copy()
    train_adata = datamodule.adata_full[~held_out_mask].copy()

    x_train, lib_train, _ = datamodule.train_pipeline.transform(train_adata, device="cpu")
    x_test, lib_test, _ = datamodule.train_pipeline.transform(test_adata, device="cpu")
    x_train = np.asarray(x_train, dtype=np.float32)
    x_test = np.asarray(x_test, dtype=np.float32)
    lib_test = np.asarray(lib_test, dtype=np.float32)

    schema = datamodule.schema
    control_col = str(schema.control_column)
    control_val = str(schema.control_value)
    pert_col = str(schema.perturbation_source)
    covariate_keys = tuple(str(cov.source_column) for cov in schema.sample_covariates)

    test_obs = test_adata.obs.copy().reset_index(drop=True)
    pert_mask = test_obs[control_col].astype(str).to_numpy() != control_val
    pert_obs = test_obs.loc[pert_mask].reset_index(drop=True)
    x_test_pert = x_test[pert_mask]
    lib_test_pert = lib_test[pert_mask]

    # control means per cell type in training pipeline space (for cosine logfc)
    train_obs = train_adata.obs.copy().reset_index(drop=True)
    train_ctrl_mask = train_obs[control_col].astype(str).to_numpy() == control_val
    x_train_ctrl = x_train[train_ctrl_mask]
    ctrl_obs_df = train_obs.loc[train_ctrl_mask].reset_index(drop=True)
    ctrl_means: dict[str, np.ndarray] = {}
    for _ct in ctrl_obs_df["cell_type"].astype(str).unique():
        _mask = ctrl_obs_df["cell_type"].astype(str).to_numpy() == _ct
        ctrl_means[_ct] = x_train_ctrl[_mask].mean(axis=0)
    fallback_ctrl = x_train_ctrl.mean(axis=0)

    # training perturbation groups for perturbation responsiveness ranking
    train_pert_mask = train_obs[control_col].astype(str).to_numpy() != control_val
    train_pert_obs = train_obs.loc[train_pert_mask].reset_index(drop=True)
    x_train_pert = x_train[train_pert_mask]
    group_cols_train = [*covariate_keys, pert_col, "dose"]
    group_means_train = []
    for _, grp in train_pert_obs.groupby(list(group_cols_train), observed=True):
        if len(grp) < 2:
            continue
        group_means_train.append(x_train_pert[grp.index.to_numpy()].mean(axis=0))
    group_means_train = np.stack(group_means_train)
    gene_pert_responsiveness = group_means_train.std(axis=0)
    top_deg_idx = np.argsort(-gene_pert_responsiveness)[:500]
    print(f"top 500 perturbation-responsive gene std range: [{gene_pert_responsiveness[top_deg_idx].min():.4f}, {gene_pert_responsiveness[top_deg_idx].max():.4f}]")

    # held-out perturbation groups (for cosine logfc)
    dose_key = "dose"
    group_columns = [*covariate_keys, pert_col, dose_key]
    group_keys = []
    group_slices = []
    for _, grp in pert_obs.groupby(list(group_columns), observed=True):
        if len(grp) < 2:
            continue
        group_keys.append({col: str(grp.iloc[0][col]) for col in group_columns})
        group_slices.append(grp.index.to_numpy())

    orig_logfcs = np.array(
        [
            x_test_pert[idx].mean(axis=0) - ctrl_means.get(gk.get("cell_type", ""), fallback_ctrl)
            for gk, idx in zip(group_keys, group_slices)
        ]
    )
    valid_mask = np.array([np.linalg.norm(lfc) > 1e-8 for lfc in orig_logfcs])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def _score(tag: str) -> dict | None:
        proj = _load_projection(tag)
        if proj is None:
            return None
        ae = proj.ae_model.to(device).eval()
        with torch.no_grad():
            z = ae.encode(torch.as_tensor(x_test_pert, dtype=torch.float32, device=device))
            lib_t = torch.as_tensor(lib_test_pert, dtype=torch.float32, device=device)
            recon = ae.reconstruct_input(z, lib_t).cpu().numpy().astype(np.float32)

        var_total = float(np.var(x_test_pert))
        mse_total = float(np.mean((x_test_pert - recon) ** 2))
        ve_total = 1.0 - mse_total / var_total

        var_deg = float(np.var(x_test_pert[:, top_deg_idx]))
        mse_deg = float(np.mean((x_test_pert[:, top_deg_idx] - recon[:, top_deg_idx]) ** 2))
        ve_deg = 1.0 - mse_deg / var_deg

        recon_logfcs = np.array(
            [
                recon[idx].mean(axis=0) - ctrl_means.get(gk.get("cell_type", ""), fallback_ctrl)
                for gk, idx in zip(group_keys, group_slices)
            ]
        )
        dots = np.sum(orig_logfcs * recon_logfcs, axis=1)
        nu = np.linalg.norm(orig_logfcs, axis=1)
        nv = np.linalg.norm(recon_logfcs, axis=1)
        cos_lfc = (dots / (nu * nv + 1e-12))[valid_mask].mean()

        del ae
        torch.cuda.empty_cache()
        return {
            "tag": tag,
            "ve_total": ve_total,
            "ve_deg500": ve_deg,
            "cos_lfc": float(cos_lfc),
        }

    tags = [
        "sciplex_ae_deg_mse_d512",  # narrow baseline
        "sciplex_ae_deg_mse_wide_d512",  # wide baseline (100 ep uniform mse)
        "debug_mse_wide_varnorm_d512",  # v1 broken floor (1e-6)
        "debug_mse_wide_varnorm_v2_d512",  # v2 fixed floor (0.1 * mean)
        "debug_mse_wide_pertw_d512",  # pert-weighted top 500 * 10x
        "debug_mse_linear_d512",  # linear AE (should match PCA d=512)
    ]
    print()
    print(f"{'tag':40s} {'ve_total':>10s} {'ve_deg500':>12s} {'cos_lfc':>10s}")
    for tag in tags:
        r = _score(tag)
        if r is None:
            print(f"{tag:40s}   (projection pickle not found)")
            continue
        print(f"{r['tag']:40s} {r['ve_total']:10.4f} {r['ve_deg500']:12.4f} {r['cos_lfc']:10.4f}")


if __name__ == "__main__":
    main()
