"""run backed benchmark predictors"""

from __future__ import annotations

from pathlib import Path

import anndata as ad

from flatcfm.analysis.flow_results import load_flow_predictions, load_flow_run

from .base import BenchmarkPredictor, PredictorDataBundle


class _RunPredictor(BenchmarkPredictor):
    """run predictor"""

    expected_task: str | None = None

    def __init__(self, name: str, run_dir: str | Path):
        super().__init__(name=name)
        self.run_dir = Path(run_dir).expanduser().resolve()
        self._bundle = None

    def _load_bundle(self):
        """load flow bundle lazily"""

        if self._bundle is None:
            self._bundle = load_flow_run(run_dir=self.run_dir)
            if self.expected_task is not None:
                task_name = str(self._bundle.cfg.task.name)
                if task_name != self.expected_task:
                    raise ValueError(f"run task {task_name!r} does not match predictor task {self.expected_task!r}")
        return self._bundle

    def predict(self, prediction_bundle: PredictorDataBundle) -> ad.AnnData:
        """predict by loading saved run"""

        bundle = self._load_bundle()
        return load_flow_predictions(
            bundle,
            prediction_name=prediction_bundle.prediction_name,
            prediction_overrides=prediction_bundle.prediction_overrides,
        )


class FMRunPredictor(_RunPredictor):
    """fm run predictor"""

    expected_task = "fm"


class ODERunPredictor(_RunPredictor):
    """ode run predictor"""

    expected_task = "ode"


class BaselineRunPredictor(_RunPredictor):
    """baseline run predictor"""

    expected_task = None
