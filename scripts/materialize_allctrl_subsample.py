#!/usr/bin/env python
"""materialize the strict_k562_allcontrols subsample artifacts on a cpu
node so subsequent sbatch jobs skip the sciplex3_raw download and union
step the logic here mirrors SciplexDataModule._prepare_canonical_subsample
with include_all_controls=True
"""

from __future__ import annotations

from pathlib import Path

import typer


def main(
    subsample_seed: int = typer.Option(0, help="seed for the base perturbed subsample"),
    subsample_n_cells: int = typer.Option(100_000, help="size of the base subsample"),
    force: bool = typer.Option(False, "--force", help="rebuild even if artifacts exist"),
) -> None:
    import sys
    sys.path.insert(0, "src")

    import anndata as ad
    import pertpy

    from flatcfm.data.accessors import resolve_sciplex_artifacts
    from flatcfm.data.splitters import (
        SplitConfig,
        build_holdout_manifest,
        save_cell_names_csv,
        save_manifest_json,
        select_subsample_cell_names,
    )

    splitter_cfg = {
        "seed": 42,
        "test_cell_type": "K562",
        "holdout_fraction": 0.5,
        "subsample_seed": subsample_seed,
        "subsample_n_cells": subsample_n_cells,
        "split_policy": "underrepresented_context",
        "ae_subsample_seed": 42,
        "ae_subsample_n_cells": 50_000,
        "ae_subsample_group_cols": ["cell_type", "vehicle"],
        "include_all_controls": True,
    }
    paths_cfg = {
        "split_artifact_dir": "artifacts/splits",
        "data_dir": "artifacts/data",
        "model_dir": "artifacts/models",
        "space_dir": "artifacts/spaces",
    }
    artifacts = resolve_sciplex_artifacts(splitter_cfg, paths_cfg)

    typer.echo(f"tag = {artifacts.tag}")
    typer.echo(f"h5ad = {artifacts.subsample_h5ad_path}")
    typer.echo(f"csv  = {artifacts.subsample_cells_csv_path}")
    typer.echo(f"json = {artifacts.holdout_json_path}")

    if artifacts.subsample_h5ad_path.exists() and not force:
        typer.echo("already materialized (use --force to rebuild)")
        return

    artifacts.split_artifact_dir.mkdir(parents=True, exist_ok=True)
    artifacts.subsample_h5ad_path.parent.mkdir(parents=True, exist_ok=True)

    typer.echo("loading sciplex3_raw from pertpy (first time is slow)...")
    adata_full = pertpy.data.sciplex3_raw()
    typer.echo(f"full dataset: {adata_full.n_obs} cells")

    base_names = select_subsample_cell_names(
        adata_full,
        n_cells=subsample_n_cells,
        seed=subsample_seed,
    )
    vehicle_col = "vehicle"
    is_ctrl = adata_full.obs[vehicle_col].astype(int) == 1
    all_control_names = adata_full.obs_names[is_ctrl.to_numpy(dtype=bool)].tolist()
    control_set = set(all_control_names)
    pert_from_base = [c for c in base_names if c not in control_set]

    merged: list[str] = []
    seen: set[str] = set()
    for c in pert_from_base + all_control_names:
        if c in seen:
            continue
        seen.add(c)
        merged.append(c)
    typer.echo(
        f"merged: {len(pert_from_base)} perts (from seeded subsample) + "
        f"{len(all_control_names)} controls = {len(merged)} cells"
    )

    save_cell_names_csv(merged, artifacts.subsample_cells_csv_path)
    adata_sub = adata_full[merged].copy()
    adata_sub.write_h5ad(artifacts.subsample_h5ad_path)

    manifest = build_holdout_manifest(
        adata_sub,
        SplitConfig(
            seed=int(splitter_cfg["seed"]),
            test_cell_type=str(splitter_cfg["test_cell_type"]),
            holdout_fraction=float(splitter_cfg["holdout_fraction"]),
            subsample_seed=subsample_seed,
            subsample_n_cells=subsample_n_cells,
            split_policy=str(splitter_cfg["split_policy"]),
            ae_subsample_seed=int(splitter_cfg["ae_subsample_seed"]),
            ae_subsample_n_cells=int(splitter_cfg["ae_subsample_n_cells"]),
            ae_subsample_group_cols=tuple(splitter_cfg["ae_subsample_group_cols"]),
            include_all_controls=True,
        ),
    )
    save_manifest_json(manifest, artifacts.holdout_json_path)
    typer.echo(f"done. wrote {artifacts.subsample_h5ad_path}")


if __name__ == "__main__":
    typer.run(main)
