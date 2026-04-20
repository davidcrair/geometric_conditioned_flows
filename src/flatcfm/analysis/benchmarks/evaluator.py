"""benchmark evaluator"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import anndata as ad
import pandas as pd

from .evaluation import Evaluation


@dataclass(frozen=True)
class Evaluator:
    """task aware benchmark evaluator"""

    reference_adata: ad.AnnData
    perturbation_key: str
    covariate_keys: tuple[str, ...] = ()
    control_value: str | None = None
    group_columns: tuple[str, ...] | None = None
    control_adata: ad.AnnData | None = None
    reference_filters: dict[str, Any] | None = None

    def evaluate(
        self,
        model_predictions: dict[str, ad.AnnData],
        metrics: Sequence[str] | None = None,
        aggregation: str | None = None,
        return_metrics_dataframe: bool = True,
        **kwargs,
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """evaluate one or more model prediction sets"""

        outputs = {}
        for model_name, prediction_adata in model_predictions.items():
            evaluation = Evaluation(
                prediction_adata=prediction_adata,
                reference_adata=self.reference_adata,
                perturbation_key=self.perturbation_key,
                covariate_keys=tuple(self.covariate_keys),
                control_value=self.control_value,
                model_name=str(model_name),
                group_columns=self.group_columns,
                control_adata=self.control_adata,
                reference_filters=self.reference_filters,
            )
            outputs[str(model_name)] = evaluation.evaluate(
                metrics=metrics,
                aggregation=aggregation,
                **kwargs,
            )
        if not return_metrics_dataframe:
            return outputs
        if not outputs:
            return pd.DataFrame()
        return pd.concat(outputs.values(), axis=0, ignore_index=True)
