"""deterministic split for sciplex dataset"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import math
from typing import Any

import anndata as ad
import numpy as np


@dataclass(frozen=True)
class SplitConfig:
    """configuration for split

    Attributes:
        seed: seed used for held-out product selection
        test_cell_type: cell type used to define candidate held-out products
        holdout_fraction: fraction of unique products in `test_cell_type` to hold out
        subsample_seed: seed used for the global dataset subsample
        subsample_n_cells: number of cells in the canonical sciplex subsample
        strict_no_leakage: if true held-out products are removed from perturbed training cells globally
        ae_subsample_seed: seed used for ae training cell subsample
        ae_subsample_n_cells: number of ae training cells to sample
        ae_subsample_group_cols: columns used to stratify ae subsampling
    """

    seed: int = 42
    test_cell_type: str = "K562"
    holdout_fraction: float = 0.5
    subsample_seed: int = 0
    subsample_n_cells: int = 100_000
    strict_no_leakage: bool = True
    ae_subsample_seed: int = 42
    ae_subsample_n_cells: int = 50_000
    ae_subsample_group_cols: tuple[str, ...] = ("cell_type", "vehicle")


@dataclass(frozen=True)
class SplitArtifacts:
    """split artifact paths for a specific split tag

    Attributes:
        tag: deterministic split identifier
        holdout_json_path: path to held-out product manifest json
        subsample_cells_csv_path: path to canonical subsample cell ids csv
        ae_train_cells_csv_path: path to ae-train cell ids csv
    """

    tag: str
    holdout_json_path: Path
    subsample_cells_csv_path: Path
    ae_train_cells_csv_path: Path


def make_split_tag(config: SplitConfig) -> str:
    """build a deterministic split tag from configuration

    Args:
        config: split configuration

    Returns:
        deterministic split tag string
    """

    policy = "strict" if config.strict_no_leakage else "nonstrict"
    cell_type = config.test_cell_type.lower().replace(" ", "-")
    strat_cols = "-".join(config.ae_subsample_group_cols)
    return (
        f"{policy}_{cell_type}_seed{config.seed}_subseed{config.subsample_seed}_"
        f"n{config.subsample_n_cells}_ae{config.ae_subsample_n_cells}_strat-{strat_cols}"
    )


def make_split_artifacts(
    config: SplitConfig,
    artifact_dir: Path | str = Path("artifacts/splits"),
    dataset_name: str = "sciplex",
) -> SplitArtifacts:
    """construct artifact paths for a split configuration

    Args:
        config: split configuration
        artifact_dir: directory where split artifacts live
        dataset_name: dataset prefix used for file names

    Returns:
        dataclass with all split artifact paths
    """

    tag = make_split_tag(config)
    artifact_root = Path(artifact_dir)
    return SplitArtifacts(
        tag=tag,
        holdout_json_path=artifact_root / f"{dataset_name}_{tag}_holdout.json",
        subsample_cells_csv_path=artifact_root / f"{dataset_name}_{tag}_subsample_cells.csv",
        ae_train_cells_csv_path=artifact_root / f"{dataset_name}_{tag}_ae_train_cells.csv",
    )


def build_holdout_manifest(adata: ad.AnnData, config: SplitConfig) -> dict[str, Any]:
    """build a held-out product manifest

    Args:
        adata: anndata containing at least vehicle cell_type and product_name columns
        config: split configuration

    Returns:
        dictionary manifest with selected held-out product names and metadata
    """

    if not 0.0 <= config.holdout_fraction <= 1.0:
        raise ValueError(f"holdout_fraction must be in [0, 1], got {config.holdout_fraction}.")

    required_cols = {"vehicle", "cell_type", "product_name"}
    missing = sorted(required_cols - set(adata.obs.columns))
    if missing:
        raise ValueError(f"Missing required columns in adata.obs: {missing}")

    is_pert = adata.obs["vehicle"] == 0
    k562_mask = is_pert & (adata.obs["cell_type"] == config.test_cell_type)
    unique_products = sorted(adata.obs.loc[k562_mask, "product_name"].unique().tolist())

    n_holdout = int(math.floor(len(unique_products) * config.holdout_fraction))
    rng = np.random.default_rng(config.seed)
    if n_holdout == 0:
        selected = []
    else:
        selected = sorted(rng.choice(unique_products, size=n_holdout, replace=False).tolist())

    split_policy = "strict_no_leakage" if config.strict_no_leakage else "legacy_non_strict"

    return {
        "version": 1,
        "dataset_name": "sciplex",
        "split_policy": split_policy,
        "seed": config.seed,
        "test_cell_type": config.test_cell_type,
        "holdout_fraction": config.holdout_fraction,
        "selected_holdout_product_names": selected,
        "columns": {
            "product_name": "product_name",
            "cell_type": "cell_type",
            "vehicle": "vehicle",
            "vehicle_control_value": 1,
            "vehicle_perturbed_value": 0,
        },
        "n_obs_total": int(adata.n_obs),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def apply_holdout_masks(adata: ad.AnnData, manifest: dict[str, Any]) -> dict[str, np.ndarray]:
    """apply held-out manifest to produce train/eval masks

    Args:
        adata: anndata to mask
        manifest: holdout manifest dictionary

    Returns:
        dictionary of boolean numpy masks
    """

    columns = manifest.get("columns", {})
    product_col = columns.get("product_name", "product_name")
    vehicle_col = columns.get("vehicle", "vehicle")
    control_value = columns.get("vehicle_control_value", 1)
    pert_value = columns.get("vehicle_perturbed_value", 0)

    required_cols = {product_col, vehicle_col}
    missing = sorted(required_cols - set(adata.obs.columns))
    if missing:
        raise ValueError(f"Missing required columns in adata.obs: {missing}")

    selected_products = set(manifest.get("selected_holdout_product_names", []))
    is_pert_any = (adata.obs[vehicle_col] == pert_value).to_numpy(dtype=bool)
    is_ctrl = (adata.obs[vehicle_col] == control_value).to_numpy(dtype=bool)

    strict = manifest.get("split_policy", "strict_no_leakage") == "strict_no_leakage"
    if strict:
        is_held_out = is_pert_any & adata.obs[product_col].isin(selected_products).to_numpy(dtype=bool)
    else:
        is_held_out = adata.obs[product_col].isin(selected_products).to_numpy(dtype=bool)

    is_train = ~is_held_out
    is_pert_train = is_pert_any & is_train

    return {
        "is_ctrl": is_ctrl,
        "is_pert_any": is_pert_any,
        "is_held_out": is_held_out,
        "is_train": is_train,
        "is_pert_train": is_pert_train,
    }


def select_subsample_cell_names(adata: ad.AnnData, n_cells: int, seed: int) -> list[str]:
    """select deterministic random cell ids without replacement

    Args:
        adata: anndata whose obs_names are sampled
        n_cells: number of cells requested
        seed: random seed

    Returns:
        list of selected obs_names
    """

    if n_cells <= 0:
        raise ValueError(f"n_cells must be positive, got {n_cells}.")

    n_take = min(n_cells, adata.n_obs)
    rng = np.random.default_rng(seed)
    idx = rng.choice(adata.n_obs, size=n_take, replace=False)
    return adata.obs_names[idx].tolist()


def select_stratified_cell_names(
    adata: ad.AnnData,
    n_cells: int,
    seed: int,
    group_cols: tuple[str, ...],
) -> list[str]:
    """select deterministic stratified cell ids across obs groups

    Args:
        adata: anndata whose obs_names are sampled
        n_cells: number of cells requested
        seed: random seed
        group_cols: tuple of obs columns used to define strata

    Returns:
        list of selected obs_names
    """

    if n_cells <= 0:
        raise ValueError(f"n_cells must be positive, got {n_cells}.")
    if not group_cols:
        raise ValueError("group_cols must be non-empty for stratified sampling.")

    missing = sorted(set(group_cols) - set(adata.obs.columns))
    if missing:
        raise ValueError(f"Missing group columns in adata.obs: {missing}")

    n_take = min(n_cells, adata.n_obs)
    rng = np.random.default_rng(seed)

    key_series = adata.obs.loc[:, list(group_cols)].astype(str).agg("|".join, axis=1)
    groups: dict[str, np.ndarray] = {}
    for key in sorted(key_series.unique()):
        groups[key] = np.where(key_series.to_numpy() == key)[0]

    group_keys = sorted(groups.keys())
    group_sizes = np.array([len(groups[k]) for k in group_keys], dtype=float)
    expected = group_sizes / group_sizes.sum() * n_take
    base = np.floor(expected).astype(int)
    remainder = expected - base
    allocation = {k: int(v) for k, v in zip(group_keys, base)}

    remaining = int(n_take - base.sum())
    if remaining > 0:
        order = np.argsort(-remainder)
        for idx in order:
            key = group_keys[int(idx)]
            if allocation[key] < len(groups[key]):
                allocation[key] += 1
                remaining -= 1
                if remaining == 0:
                    break

    selected_idx_parts: list[np.ndarray] = []
    for key in group_keys:
        pool = groups[key]
        k = min(allocation[key], len(pool))
        if k > 0:
            selected_idx_parts.append(rng.choice(pool, size=k, replace=False))

    selected_idx = np.concatenate(selected_idx_parts) if selected_idx_parts else np.array([], dtype=int)

    if selected_idx.size < n_take:
        remaining_pool = np.setdiff1d(np.arange(adata.n_obs), selected_idx, assume_unique=False)
        extra = rng.choice(remaining_pool, size=n_take - selected_idx.size, replace=False)
        selected_idx = np.concatenate([selected_idx, extra])

    selected_idx = rng.permutation(selected_idx)
    return adata.obs_names[selected_idx].tolist()


def save_manifest_json(manifest: dict[str, Any], path: Path | str) -> None:
    """save a holdout manifest as json

    Args:
        manifest: manifest dictionary
        path: output json path
    """

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def load_manifest_json(path: Path | str) -> dict[str, Any]:
    """load a holdout manifest json

    Args:
        path: input json path

    Returns:
        parsed manifest dictionary
    """

    input_path = Path(path)
    return json.loads(input_path.read_text())


def save_cell_names_csv(cell_names: list[str], path: Path | str, column: str = "obs_name") -> None:
    """save cell ids to a single-column csv

    Args:
        cell_names: list of obs names
        path: output csv path
        column: header name for the single column
    """

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([column])
        for name in cell_names:
            writer.writerow([name])


def load_cell_names_csv(path: Path | str, column: str = "obs_name") -> list[str]:
    """load cell ids from a single-column csv

    Args:
        path: input csv path
        column: expected column header

    Returns:
        list of obs names
    """

    input_path = Path(path)
    with input_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(f"CSV {input_path} must contain column '{column}'.")
        return [row[column] for row in reader]


def validate_no_leakage(
    adata: ad.AnnData,
    masks: dict[str, np.ndarray],
    product_name_col: str = "product_name",
) -> None:
    """validate that train and held-out products do not overlap

    Args:
        adata: anndata corresponding to masks
        masks: boolean mask dictionary from `apply_holdout_masks`
        product_name_col: product-name column in `adata.obs`

    Raises:
        ValueError: if masks are malformed or leakage is detected
    """

    required_mask_keys = {"is_held_out", "is_train"}
    missing = sorted(required_mask_keys - set(masks))
    if missing:
        raise ValueError(f"Missing required masks: {missing}")

    is_held_out = np.asarray(masks["is_held_out"], dtype=bool)
    is_train = np.asarray(masks["is_train"], dtype=bool)

    if is_held_out.shape[0] != adata.n_obs or is_train.shape[0] != adata.n_obs:
        raise ValueError("Mask lengths must match adata.n_obs.")
    if np.any(is_held_out & is_train):
        raise ValueError("A cell cannot be both held out and in train.")
    if not np.array_equal(~is_held_out, is_train):
        raise ValueError("is_train must be the exact complement of is_held_out.")

    held_products = set(adata.obs.loc[is_held_out, product_name_col].astype(str).unique().tolist())
    train_products = set(adata.obs.loc[is_train, product_name_col].astype(str).unique().tolist())
    overlap = sorted(held_products & train_products)
    if overlap:
        head = overlap[:5]
        raise ValueError(f"Detected product leakage between train and held-out sets: {head}")
