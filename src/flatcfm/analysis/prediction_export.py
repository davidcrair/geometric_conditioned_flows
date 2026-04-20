"""prediction export helpers"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import anndata as ad
import numpy as np
import pandas as pd
import yaml


@dataclass
class PredictionArtifacts:
    """prediction artifact bundle"""

    predictions_path: Path
    run_config_path: Path
    prediction_request_path: Path
    prediction_metadata_path: Path


def _is_h5ad_list_scalar(value) -> bool:
    """check whether a list item can be stored as an h5ad array element"""

    return isinstance(value, (str, bytes, int, float, bool, np.generic)) or value is None


def _sanitize_uns_for_h5ad(value):
    """sanitize nested uns metadata for reliable h5ad serialization"""

    if isinstance(value, dict):
        return {str(key): _sanitize_uns_for_h5ad(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        items = [_sanitize_uns_for_h5ad(item) for item in value]
        if all(_is_h5ad_list_scalar(item) for item in items):
            return items
        return {str(idx): item for idx, item in enumerate(items)}
    if isinstance(value, Path):
        return str(value)
    return value


def build_prediction_adata(matrix, obs: pd.DataFrame, var_names: list[str], uns: dict | None = None) -> ad.AnnData:
    """build prediction adata"""

    pred_adata = ad.AnnData(X=matrix, obs=obs.copy())
    pred_adata.var_names = pd.Index(var_names)
    if uns:
        pred_adata.uns.update(uns)
    return pred_adata


def save_prediction_artifacts(
    output_dir: Path,
    predictions: ad.AnnData,
    run_config: dict,
    prediction_request: pd.DataFrame,
    prediction_metadata: dict,
) -> PredictionArtifacts:
    """save prediction artifacts"""

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.h5ad"
    run_config_path = output_dir / "run_config.yaml"
    prediction_request_path = output_dir / "prediction_request.csv"
    prediction_metadata_path = output_dir / "prediction_metadata.json"

    predictions_to_save = predictions.copy()
    predictions_to_save.uns.clear()
    predictions_to_save.uns.update(_sanitize_uns_for_h5ad(dict(predictions.uns)))
    predictions_to_save.write_h5ad(predictions_path)
    with run_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(run_config, handle, sort_keys=False)
    prediction_request.to_csv(prediction_request_path, index=False)
    with prediction_metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(prediction_metadata, handle, indent=2, sort_keys=True)

    return PredictionArtifacts(
        predictions_path=predictions_path,
        run_config_path=run_config_path,
        prediction_request_path=prediction_request_path,
        prediction_metadata_path=prediction_metadata_path,
    )
