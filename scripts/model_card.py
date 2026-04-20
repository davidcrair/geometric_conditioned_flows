#!/usr/bin/env python
"""display a model card for a trained run"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from omegaconf import OmegaConf
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(add_completion=False)
console = Console(width=80)

RUNS_ROOT = Path("artifacts/runs")

# keys to skip when displaying model architecture
_MODEL_SKIP_KEYS = {"_target_", "name"}


def _resolve_run_path(raw: str) -> Path:
    """resolve a run path from full path or bare experiment name"""

    p = Path(raw)
    if p.is_dir() and (p / "run_config.yaml").exists():
        return p.resolve()
    if "/" not in raw:
        experiment_dir = RUNS_ROOT / raw
        if not experiment_dir.is_dir():
            console.print(f"[red]experiment directory not found: {experiment_dir}[/]")
            raise typer.Exit(1)
        subdirs = sorted(d for d in experiment_dir.iterdir() if d.is_dir())
        if not subdirs:
            console.print(f"[red]no run subdirectories in {experiment_dir}[/]")
            raise typer.Exit(1)
        latest = subdirs[-1]
        if not (latest / "run_config.yaml").exists():
            console.print(f"[red]no run_config.yaml in {latest}[/]")
            raise typer.Exit(1)
        return latest.resolve()
    console.print(f"[red]invalid run path: {raw}[/]")
    raise typer.Exit(1)


def _kv_table() -> Table:
    """create a key-value table with no header"""

    return Table(show_header=False, box=None, padding=(0, 2), expand=True)


def _sel(cfg, path: str, default=None):
    """safe nested config access"""

    return OmegaConf.select(cfg, path, default=default)


def _fmt(val) -> str:
    """format a config value for display"""

    if val is None:
        return "[dim]-[/]"
    if isinstance(val, bool):
        return "[green]true[/]" if val else "[dim]false[/]"
    if isinstance(val, float):
        if val == int(val) and abs(val) < 1e6:
            return str(int(val))
        if abs(val) < 0.01 or abs(val) >= 1e4:
            return f"{val:.1e}"
        return f"{val:g}"
    return str(val)


def _show_identity(cfg, run_dir: Path, history: dict, metadata: dict) -> None:
    """display model identity panel"""

    t = _kv_table()
    t.add_column("key", style="bold cyan", no_wrap=True, width=14)
    t.add_column("val")

    t.add_row("experiment", str(_sel(cfg, "experiment_name", "?")))

    task_name = str(_sel(cfg, "task.name", "?"))
    model_target = str(_sel(cfg, "model._target_", ""))
    model_class = model_target.rsplit(".", 1)[-1] if model_target else ""
    t.add_row("task", f"{task_name} ({model_class})" if model_class else task_name)

    t.add_row("run", run_dir.name)
    t.add_row("seed", _fmt(_sel(cfg, "seed")))

    has_ckpt = (run_dir / "checkpoints" / "best.ckpt").exists()
    has_last = (run_dir / "checkpoints" / "last.ckpt").exists()
    has_pred = (run_dir / "predictions" / "held_out" / "predictions.h5ad").exists()
    ckpt_str = "[green]yes[/] best" if has_ckpt else ("[yellow]yes[/] last" if has_last else "[red]no[/]")
    pred_str = "[green]yes[/]" if has_pred else "[red]no[/]"
    t.add_row("checkpoint", ckpt_str)
    t.add_row("predictions", pred_str)

    val_losses = history.get("val_loss", [])
    n_epochs = len(val_losses)
    if val_losses:
        best = min(val_losses)
        t.add_row("epochs", f"{n_epochs} (best val_loss: {best:.4f})")
    elif n_epochs:
        t.add_row("epochs", str(n_epochs))

    n_features = len(metadata.get("feature_names", []))
    if n_features:
        t.add_row("n_features", str(n_features))

    console.print(Panel(t, title="[bold]Model Identity[/]", border_style="blue"))


def _show_space(cfg) -> None:
    """display space configuration panel"""

    t = _kv_table()
    t.add_column("key", style="bold cyan", no_wrap=True, width=14)
    t.add_column("val")

    t.add_row("base", _fmt(_sel(cfg, "space.base.kind")))
    feature_set = _sel(cfg, "space.base.feature_set")
    n_hvgs = _sel(cfg, "space.base.n_hvgs")
    deg_n = _sel(cfg, "space.base.deg_n_top_genes")
    feature_str = str(feature_set or "?")
    if n_hvgs:
        feature_str += f" ({n_hvgs} HVGs"
        if deg_n:
            feature_str += f" + {deg_n} DEG/pert"
        feature_str += ")"
    t.add_row("features", feature_str)
    t.add_row("target_sum", _fmt(_sel(cfg, "space.base.target_sum")))

    projections = _sel(cfg, "space.projections", [])
    if projections:
        proj_list = OmegaConf.to_container(projections, resolve=True) if OmegaConf.is_config(projections) else projections
        parts = []
        for p in proj_list:
            if not isinstance(p, dict):
                parts.append(str(p))
                continue
            kind = p.get("kind", "?")
            if kind == "pca":
                parts.append(f"pca({p.get('n_components', '?')})")
            elif kind == "ae_latent":
                tag = p.get("artifact_tag", "?")
                parts.append(f"ae_latent({tag})")
            else:
                parts.append(kind)
        t.add_row("projections", " -> ".join(parts))
    else:
        t.add_row("projections", "[dim]none[/]")

    t.add_row("fit_scope", _fmt(_sel(cfg, "space.fit_scope")))

    ae_tag = _sel(cfg, "space.ae_export_artifact_tag")
    if ae_tag:
        t.add_row("ae_export_tag", str(ae_tag))

    console.print(Panel(t, title="[bold]Space[/]", border_style="green"))


def _show_architecture(cfg) -> None:
    """display model architecture panel"""

    model_cfg = _sel(cfg, "model")
    if model_cfg is None:
        return

    t = _kv_table()
    t.add_column("key", style="bold cyan", no_wrap=True, width=18)
    t.add_column("val")

    model_dict = OmegaConf.to_container(model_cfg, resolve=True) if OmegaConf.is_config(model_cfg) else {}

    for key, val in model_dict.items():
        if key in _MODEL_SKIP_KEYS:
            continue
        if isinstance(val, dict):
            # nested config (e.g. decoder)
            t.add_row(f"[bold]{key}[/]", "")
            for sub_key, sub_val in val.items():
                t.add_row(f"  {sub_key}", _fmt(sub_val))
        else:
            t.add_row(key, _fmt(val))

    console.print(Panel(t, title="[bold]Architecture[/]", border_style="magenta"))


def _show_training(cfg) -> None:
    """display training configuration panel"""

    t = _kv_table()
    t.add_column("key", style="bold cyan", no_wrap=True, width=14)
    t.add_column("val")

    t.add_row("epochs", _fmt(_sel(cfg, "task.epochs")))
    t.add_row("batch_size", _fmt(_sel(cfg, "task.batch_size")))
    t.add_row("lr", _fmt(_sel(cfg, "task.lr")))
    t.add_row("weight_decay", _fmt(_sel(cfg, "task.weight_decay")))

    ot = _sel(cfg, "task.use_ot_coupling")
    if ot is not None:
        t.add_row("ot_coupling", _fmt(ot))

    steps = _sel(cfg, "task.steps_per_epoch")
    if steps:
        t.add_row("steps/epoch", _fmt(steps))

    precision = _sel(cfg, "trainer.precision")
    if precision:
        t.add_row("precision", str(precision))

    console.print(Panel(t, title="[bold]Training[/]", border_style="yellow"))


def _show_loss(cfg) -> None:
    """display loss configuration panel"""

    t = _kv_table()
    t.add_column("key", style="bold cyan", no_wrap=True, width=20)
    t.add_column("val")

    weights = _sel(cfg, "loss.weights")
    if weights:
        weights_dict = OmegaConf.to_container(weights, resolve=True) if OmegaConf.is_config(weights) else weights
        for key, val in weights_dict.items():
            if isinstance(val, (int, float)) and val == 0:
                continue
            t.add_row(key, _fmt(val))

    # extra loss params
    zeta = _sel(cfg, "loss.distance_zeta")
    if zeta and float(zeta) > 0:
        t.add_row("distance_zeta", _fmt(zeta))
    alpha_min = _sel(cfg, "loss.distance_alpha_min")
    if alpha_min and float(alpha_min) > 0:
        t.add_row("distance_alpha_min", _fmt(alpha_min))
    loss_type = _sel(cfg, "loss.distance_loss_type")
    if loss_type and loss_type != "mse":
        t.add_row("distance_loss_type", str(loss_type))

    console.print(Panel(t, title="[bold]Loss[/]", border_style="red"))


def _show_data_split(cfg) -> None:
    """display data split panel"""

    t = _kv_table()
    t.add_column("key", style="bold cyan", no_wrap=True, width=16)
    t.add_column("val")

    data_name = _sel(cfg, "data.name", "?")
    data_source = _sel(cfg, "data.source", "?")
    t.add_row("data", f"{data_name} ({data_source})")

    data_path = _sel(cfg, "data.data_path")
    if data_path:
        t.add_row("data_path", str(data_path))

    max_dose = _sel(cfg, "data.max_dose_only")
    if max_dose:
        t.add_row("max_dose_only", _fmt(max_dose))

    t.add_row("test_cell_type", _fmt(_sel(cfg, "splitter.test_cell_type")))
    t.add_row("holdout_frac", _fmt(_sel(cfg, "splitter.holdout_fraction")))
    t.add_row("split_policy", _fmt(_sel(cfg, "splitter.split_policy")))

    subsample = _sel(cfg, "splitter.subsample_n_cells")
    if subsample:
        t.add_row("subsample", f"{int(subsample):,} cells")

    val_frac = _sel(cfg, "splitter.val_fraction")
    if val_frac:
        t.add_row("val_fraction", _fmt(val_frac))

    console.print(Panel(t, title="[bold]Data Split[/]", border_style="cyan"))


def _show_ae_schedule(cfg) -> None:
    """display AE schedule panel if applicable"""

    schedule_name = _sel(cfg, "ae_schedule.name")
    if not schedule_name or schedule_name == "single_phase":
        p1 = _sel(cfg, "ae_schedule.phase1_epochs", 0)
        p2 = _sel(cfg, "ae_schedule.phase2_epochs", 0)
        if not p1 and not p2:
            return

    t = _kv_table()
    t.add_column("key", style="bold cyan", no_wrap=True, width=24)
    t.add_column("val")

    t.add_row("schedule", str(schedule_name))
    t.add_row("phase1_epochs", _fmt(_sel(cfg, "ae_schedule.phase1_epochs")))
    t.add_row("phase2_epochs", _fmt(_sel(cfg, "ae_schedule.phase2_epochs")))

    for phase in ("phase1", "phase2"):
        freeze_enc = _sel(cfg, f"ae_schedule.freeze_encoder_{phase}")
        freeze_dec = _sel(cfg, f"ae_schedule.freeze_decoder_{phase}")
        if freeze_enc or freeze_dec:
            parts = []
            if freeze_enc:
                parts.append("encoder")
            if freeze_dec:
                parts.append("decoder")
            t.add_row(f"{phase}_frozen", ", ".join(parts))

        loss_weights = _sel(cfg, f"ae_schedule.{phase}_loss_weights")
        if loss_weights:
            lw = OmegaConf.to_container(loss_weights, resolve=True) if OmegaConf.is_config(loss_weights) else loss_weights
            active = {k: v for k, v in lw.items() if v and float(v) > 0}
            if active:
                t.add_row(f"{phase}_losses", ", ".join(f"{k}={v}" for k, v in active.items()))

    console.print(Panel(t, title="[bold]AE Schedule[/]", border_style="bright_magenta"))


def _show_ae_geometry(cfg) -> None:
    """display AE geometry panel if applicable"""

    mode = _sel(cfg, "ae_geometry.mode", "none")
    if mode == "none":
        return

    t = _kv_table()
    t.add_column("key", style="bold cyan", no_wrap=True, width=14)
    t.add_column("val")

    t.add_row("mode", str(mode))

    if mode == "phate_potential":
        phate = _sel(cfg, "ae_geometry.phate")
        if phate:
            phate_dict = OmegaConf.to_container(phate, resolve=True) if OmegaConf.is_config(phate) else phate
            for key, val in phate_dict.items():
                t.add_row(key, _fmt(val))

        per_ct = _sel(cfg, "ae_geometry.per_cell_type")
        if per_ct is not None:
            t.add_row("per_cell_type", _fmt(per_ct))

    console.print(Panel(t, title="[bold]AE Geometry[/]", border_style="bright_cyan"))


@app.command()
def card(
    run_path: Annotated[str, typer.Argument(help="run directory or experiment name")],
) -> None:
    """display model card for a trained run"""

    run_dir = _resolve_run_path(run_path)
    cfg = OmegaConf.load(run_dir / "run_config.yaml")

    metadata = {}
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

    history = {}
    history_path = run_dir / "history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

    console.print()
    _show_identity(cfg, run_dir, history, metadata)
    _show_space(cfg)
    _show_architecture(cfg)
    _show_training(cfg)
    _show_loss(cfg)
    _show_data_split(cfg)
    _show_ae_schedule(cfg)
    _show_ae_geometry(cfg)
    console.print()


if __name__ == "__main__":
    app()
