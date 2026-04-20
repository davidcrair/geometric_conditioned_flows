"""artifact access helpers"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

from flatcfm.data.splitters import SplitConfig, make_split_artifacts


@dataclass(frozen=True)
class SciplexArtifacts:
    """sciplex artifact paths"""

    tag: str
    data_root: Path
    model_root: Path
    space_root: Path
    split_artifact_dir: Path
    holdout_json_path: Path
    subsample_cells_csv_path: Path
    ae_train_cells_csv_path: Path
    subsample_h5ad_path: Path
    space_path: Path
    ae_space_path: Path
    ae_metadata_path: Path
    ae_phate_train_embedding_path: Path
    ae_phate_val_embedding_path: Path
    fm_model_path: Path
    fm_metadata_path: Path
    fm_nbae_model_path: Path
    fm_nbae_metadata_path: Path
    ode_model_path: Path
    ode_metadata_path: Path
    ode_nbae_model_path: Path
    ode_nbae_metadata_path: Path
    ae_model_path: Path

    def space_path_for(self, space_tag: str) -> Path:
        """path for generic space artifact"""

        return self.space_root / f"sciplex_space_{self.tag}_{space_tag}.pkl"

    def ae_space_path_for(self, space_tag: str) -> Path:
        """path for ae space artifact"""

        return self.space_root / f"sciplex_ae_space_{self.tag}_{space_tag}.pkl"

    def ae_metadata_path_for(self, space_tag: str) -> Path:
        """path for ae metadata artifact"""

        return self.model_root / f"sciplex_ae_metadata_{self.tag}_{space_tag}.json"

    def ae_model_path_for(self, space_tag: str) -> Path:
        """path for ae model artifact"""

        return self.model_root / f"sciplex_ae_model_{self.tag}_{space_tag}.ckpt"

    def ae_phate_train_embedding_path_for(self, space_tag: str) -> Path:
        """path for ae phate train artifact"""

        return self.space_root / f"sciplex_ae_phate_train_{self.tag}_{space_tag}.npy"

    def ae_phate_val_embedding_path_for(self, space_tag: str) -> Path:
        """path for ae phate val artifact"""

        return self.space_root / f"sciplex_ae_phate_val_{self.tag}_{space_tag}.npy"


def resolve_sciplex_artifacts(splitter_cfg: dict, paths_cfg: dict) -> SciplexArtifacts:
    """resolve sciplex artifacts"""

    split_config = SplitConfig(
        seed=int(splitter_cfg.get("seed", 42)),
        test_cell_type=str(splitter_cfg.get("test_cell_type", "K562")),
        holdout_fraction=float(splitter_cfg.get("holdout_fraction", 0.5)),
        subsample_seed=int(splitter_cfg.get("subsample_seed", 0)),
        subsample_n_cells=int(splitter_cfg.get("subsample_n_cells", 100_000)),
        split_policy=str(splitter_cfg.get("split_policy", "strict_no_leakage")),
        ae_subsample_seed=int(splitter_cfg.get("ae_subsample_seed", 42)),
        ae_subsample_n_cells=int(splitter_cfg.get("ae_subsample_n_cells", 50_000)),
        ae_subsample_group_cols=tuple(splitter_cfg.get("ae_subsample_group_cols", ["cell_type", "vehicle"])),
        include_all_controls=bool(splitter_cfg.get("include_all_controls", False)),
    )
    split_root = Path(paths_cfg.get("split_artifact_dir", "artifacts/splits"))
    data_root = Path(paths_cfg.get("data_dir", "artifacts/data"))
    model_root = Path(paths_cfg.get("model_dir", "artifacts/models"))
    space_root = Path(paths_cfg.get("space_dir", "artifacts/spaces"))
    artifacts = make_split_artifacts(split_config, artifact_dir=split_root, dataset_name="sciplex")
    tag = artifacts.tag
    return SciplexArtifacts(
        tag=tag,
        data_root=data_root,
        model_root=model_root,
        space_root=space_root,
        split_artifact_dir=artifacts.holdout_json_path.parent,
        holdout_json_path=artifacts.holdout_json_path,
        subsample_cells_csv_path=artifacts.subsample_cells_csv_path,
        ae_train_cells_csv_path=artifacts.ae_train_cells_csv_path,
        subsample_h5ad_path=data_root / f"sciplex_subsample_{tag}.h5ad",
        space_path=space_root / f"sciplex_space_{tag}.pkl",
        ae_space_path=space_root / f"sciplex_ae_space_{tag}.pkl",
        ae_metadata_path=model_root / f"sciplex_ae_metadata_{tag}.json",
        ae_phate_train_embedding_path=space_root / f"sciplex_ae_phate_train_{tag}.npy",
        ae_phate_val_embedding_path=space_root / f"sciplex_ae_phate_val_{tag}.npy",
        fm_model_path=model_root / f"sciplex_fm_model_{tag}.pt",
        fm_metadata_path=model_root / f"sciplex_fm_metadata_{tag}.pkl",
        fm_nbae_model_path=model_root / f"sciplex_fm_nbae_model_{tag}.pt",
        fm_nbae_metadata_path=model_root / f"sciplex_fm_nbae_metadata_{tag}.pkl",
        ode_model_path=model_root / f"sciplex_ode_model_{tag}.pt",
        ode_metadata_path=model_root / f"sciplex_ode_metadata_{tag}.pkl",
        ode_nbae_model_path=model_root / f"sciplex_ode_nbae_model_{tag}.pt",
        ode_nbae_metadata_path=model_root / f"sciplex_ode_nbae_metadata_{tag}.pkl",
        ae_model_path=model_root / f"sciplex_ae_model_{tag}.pt",
    )


def load_pickle(path: Path):
    """load pickle"""

    with path.open("rb") as handle:
        return pickle.load(handle)
