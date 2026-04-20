"""toy analysis helpers"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
from omegaconf import OmegaConf

from flatcfm.modelcore.utils import instantiate_datamodule


def _load_run_config(run_dir: Path):
    """load run config"""

    return OmegaConf.load(run_dir / "run_config.yaml")


def load_toy_prediction_adata(run_dir: str | Path, prediction_name: str = "held_out") -> ad.AnnData:
    """load saved toy predictions"""

    run_path = Path(run_dir).expanduser().resolve()
    return ad.read_h5ad(run_path / "predictions" / prediction_name / "predictions.h5ad")


def build_toy_scatter_payload(run_dir: str | Path, prediction_name: str = "held_out") -> dict[str, np.ndarray]:
    """build canonical toy scatter payload"""

    run_path = Path(run_dir).expanduser().resolve()
    cfg = _load_run_config(run_path)
    datamodule = instantiate_datamodule(cfg)
    datamodule.setup("predict")

    pred_adata = load_toy_prediction_adata(run_path, prediction_name=prediction_name)
    control_mask = datamodule.adata_full.obs[datamodule.schema.control_column].to_numpy() == datamodule.schema.control_value
    _, val_mask = datamodule._build_train_val_masks()
    target_mask = np.asarray(val_mask, dtype=bool) & ~np.asarray(control_mask, dtype=bool)

    control_points = np.asarray(datamodule.adata_full[control_mask].X, dtype=np.float32)
    target_points = np.asarray(datamodule.adata_full[target_mask].X, dtype=np.float32)
    prediction_points = np.asarray(pred_adata.X, dtype=np.float32)

    return {
        "control": control_points,
        "target": target_points,
        "prediction": prediction_points,
    }
