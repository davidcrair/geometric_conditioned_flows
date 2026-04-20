"""runtime helpers for hydra and lightning"""

from __future__ import annotations

from pathlib import Path
import json
import random

from hydra.utils import instantiate
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from flatcfm._utils import to_plain_dict  # noqa: F401 (re-exported)


def instantiate_datamodule(cfg):
    """instantiate datamodule from hydra config"""

    return instantiate(
        cfg.data,
        data=to_plain_dict(cfg.data),
        splitter=to_plain_dict(cfg.splitter),
        space=to_plain_dict(cfg.space),
        evaluation_space=to_plain_dict(cfg.evaluation_space),
        condition=to_plain_dict(cfg.condition),
        paths=to_plain_dict(cfg.paths),
        ae_geometry=to_plain_dict(cfg.ae_geometry),
        task=to_plain_dict(cfg.task),
        predict=to_plain_dict(cfg.predict),
        trainer=to_plain_dict(cfg.trainer),
        _recursive_=False,
    )


def instantiate_model(cfg, datamodule):
    """instantiate model from hydra config and datamodule"""

    return instantiate(
        cfg.model,
        model_cfg=to_plain_dict(cfg.model),
        task_cfg=to_plain_dict(cfg.task),
        loss_cfg=to_plain_dict(cfg.loss),
        predict_cfg=to_plain_dict(cfg.predict),
        _recursive_=False,
        **datamodule.get_model_init_kwargs(),
    )


def set_random_seed(seed: int) -> None:
    """set random seed"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def instantiate_collection(cfg) -> list:
    """instantiate config collection"""

    if cfg is None:
        return []
    nodes = []
    for _, node in cfg.items():
        if node is None:
            continue
        nodes.append(instantiate(node))
    return nodes


def instantiate_loggers(cfg):
    """instantiate loggers"""

    loggers = instantiate_collection(cfg)
    if not loggers:
        return False
    if len(loggers) == 1:
        return loggers[0]
    return loggers


def save_json(path: Path, payload: dict) -> Path:
    """save json"""

    def make_json_safe(value):
        """convert nested values to json safe types"""

        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): make_json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [make_json_safe(item) for item in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return value.item()
            return value.detach().cpu().tolist()
        if OmegaConf.is_config(value):
            return make_json_safe(OmegaConf.to_container(value, resolve=True))
        return value

    with path.open("w", encoding="utf-8") as handle:
        json.dump(make_json_safe(payload), handle, indent=2, sort_keys=True)
    return path


def save_run_config(path: Path, cfg) -> Path:
    """save run config"""

    OmegaConf.save(cfg, path, resolve=True)
    return path


def get_metrics_csv_path(logger) -> Path | None:
    """get metrics csv path"""

    if logger is False or logger is None:
        return None
    loggers = logger if isinstance(logger, list) else [logger]
    for item in loggers:
        log_dir = getattr(item, "log_dir", None)
        if log_dir is None:
            continue
        metrics_path = Path(log_dir) / "metrics.csv"
        if metrics_path.exists():
            return metrics_path
    return None


def extract_history_from_metrics(metrics_path: Path | None) -> dict:
    """extract history from lightning metrics csv"""

    history = {
        "train_loss": [],
        "val_loss": [],
        "individual_train_losses": {},
        "individual_val_losses": {},
    }
    if metrics_path is None or not metrics_path.exists():
        return history

    metrics = pd.read_csv(metrics_path)
    skip_cols = {"epoch", "step"}
    for column in metrics.columns:
        if column in skip_cols:
            continue
        series = metrics[column].dropna().tolist()
        if not series:
            continue
        if column == "train_loss":
            history["train_loss"] = [float(value) for value in series]
        elif column == "val_loss":
            history["val_loss"] = [float(value) for value in series]
        elif column.startswith("train_"):
            history["individual_train_losses"][column.removeprefix("train_")] = [float(value) for value in series]
        elif column.startswith("val_"):
            history["individual_val_losses"][column.removeprefix("val_")] = [float(value) for value in series]
    return history


def select_history_metric(history: dict, metric_name: str) -> list[float]:
    """select metric series from history"""

    if metric_name == "train_loss":
        return [float(value) for value in history.get("train_loss", [])]
    if metric_name == "val_loss":
        return [float(value) for value in history.get("val_loss", [])]
    if metric_name.startswith("train_"):
        key = metric_name.removeprefix("train_")
        return [float(value) for value in history.get("individual_train_losses", {}).get(key, [])]
    if metric_name.startswith("val_"):
        key = metric_name.removeprefix("val_")
        return [float(value) for value in history.get("individual_val_losses", {}).get(key, [])]
    return []


def resolve_hpo_objective(history: dict, hpo_cfg: dict | None) -> tuple[str, float]:
    """resolve hpo objective from history"""

    cfg = dict(hpo_cfg or {})
    objective_cfg = dict(cfg.get("objective", {}))
    metric_name = str(objective_cfg.get("name", "val_loss"))
    source = str(objective_cfg.get("source", "best"))
    mode = str(objective_cfg.get("mode", "min"))

    values = select_history_metric(history, metric_name)
    if not values:
        raise ValueError(f"cannot resolve hpo objective for metric {metric_name!r} from history")

    if source == "last":
        objective_value = float(values[-1])
    elif source == "best":
        objective_value = float(min(values) if mode == "min" else max(values))
    else:
        raise ValueError(f"unsupported hpo objective source {source!r}")

    return metric_name, objective_value
