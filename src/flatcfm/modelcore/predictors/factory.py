"""predictor factory"""

from __future__ import annotations

from flatcfm._utils import to_plain_dict

from .base import BenchmarkPredictor
from .run import BaselineRunPredictor, FMRunPredictor, ODERunPredictor

_PREDICTOR_REGISTRY: dict[str, type[BenchmarkPredictor]] = {
    "run_fm": FMRunPredictor,
    "fm_run": FMRunPredictor,
    "run_ode": ODERunPredictor,
    "ode_run": ODERunPredictor,
    "run_baseline": BaselineRunPredictor,
    "baseline_run": BaselineRunPredictor,
}


def load_predictor_from_config(cfg) -> BenchmarkPredictor:
    """load predictor from config"""

    data = to_plain_dict(cfg)
    kind = str(data.get("kind", "")).lower()
    name = str(data.get("name", kind))

    predictor_cls = _PREDICTOR_REGISTRY.get(kind)
    if predictor_cls is None:
        raise ValueError(f"unsupported predictor kind {kind!r}")
    return predictor_cls(name=name, run_dir=data["run_dir"])
