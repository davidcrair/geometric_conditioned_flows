"""benchmark predictor interfaces"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import anndata as ad


@dataclass(frozen=True)
class PredictorDataBundle:
    """predictor data bundle"""

    flow_bundle: Any
    prediction_name: str = "held_out"
    prediction_overrides: dict | None = None


class BenchmarkPredictor:
    """benchmark predictor base"""

    requires_training: bool = False

    def __init__(self, name: str):
        self.name = str(name)

    def fit(self, train_bundle: PredictorDataBundle) -> None:
        """fit predictor"""

        del train_bundle

    def predict(self, prediction_bundle: PredictorDataBundle) -> ad.AnnData:
        """predict one benchmark dataset"""

        raise NotImplementedError("predictor subclasses must implement predict")
