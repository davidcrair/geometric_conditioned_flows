#!/usr/bin/env python
"""show status of all trained runs grouped by space compatibility"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Annotated

import typer
from omegaconf import OmegaConf
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from flatcfm.data.space import normalize_space_config

app = typer.Typer(add_completion=False)
console = Console()


# -- helpers ------------------------------------------------------------------


def _space_signature(space_cfg: dict) -> tuple:
    """extract space signature tuple for compatibility comparison"""
    cfg = normalize_space_config(space_cfg, default_fit_scope="train")
    base = cfg["base"]
    return (
        str(base.get("kind")),
        str(base.get("feature_set")),
        None if base.get("n_hvgs") is None else int(base["n_hvgs"]),
        float(base.get("target_sum", 1e4)),
        str(base["hvg_batch_key"]) if base.get("hvg_batch_key") is not None else None,
        str(cfg.get("fit_scope", "train")),
    )


def _signature_label(sig: tuple) -> str:
    """human readable label for a space signature"""
    kind, fset, nhvg, tsum, batch_key, fit_scope = sig
    parts = [kind]
    if fset != "all_genes":
        parts.append(fset)
        if nhvg is not None:
            parts.append(f"n={nhvg}")
    parts.append(f"tsum={int(tsum)}")
    if batch_key and batch_key != "None":
        parts.append(f"batch={batch_key}")
    parts.append(f"scope={fit_scope}")
    return " ".join(parts)


def _projection_summary(space_cfg: dict) -> str:
    """short projection summary string"""
    cfg = normalize_space_config(space_cfg, default_fit_scope="train")
    projections = cfg.get("projections", [])
    if not projections:
        return "none"
    parts = []
    for p in projections:
        kind = p["kind"]
        if kind == "pca":
            parts.append(f"pca({p['n_components']})")
        elif kind == "ae_latent":
            tag = p.get("artifact_tag")
            parts.append(f"ae({tag or 'auto'})")
        elif kind == "identity":
            parts.append("id")
        else:
            parts.append(kind)
    return "->".join(parts)


def _load_run_info(run_dir: Path) -> dict | None:
    """load run info from a single run timestamp directory"""
    config_path = run_dir / "run_config.yaml"
    if not config_path.exists():
        # might be a partial/failed run with only .hydra
        return None

    cfg = OmegaConf.load(config_path)
    space_cfg = OmegaConf.to_container(cfg.space, resolve=True)

    has_checkpoint = (run_dir / "checkpoints" / "best.ckpt").exists()
    has_last_ckpt = (run_dir / "checkpoints" / "last.ckpt").exists()
    has_predictions = (run_dir / "predictions" / "held_out" / "predictions.h5ad").exists()

    n_features = None
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)
            feature_names = meta.get("feature_names", [])
            n_features = len(feature_names) if feature_names else None

    # training history
    best_val_loss = None
    n_epochs = 0
    history_path = run_dir / "history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        val_losses = history.get("val_loss", [])
        n_epochs = len(val_losses)
        if val_losses:
            best_val_loss = min(val_losses)

    # hydra overrides for reproducing the run
    overrides_path = run_dir / ".hydra" / "overrides.yaml"
    overrides = []
    if overrides_path.exists():
        overrides = list(OmegaConf.load(overrides_path) or [])

    task_name = str(OmegaConf.select(cfg, "task.name", default="?"))
    model_name = str(OmegaConf.select(cfg, "model.name", default="?"))

    return {
        "experiment_name": str(cfg.get("experiment_name", "?")),
        "task": task_name,
        "model": model_name,
        "space_cfg": space_cfg,
        "signature": _space_signature(space_cfg),
        "projections": _projection_summary(space_cfg),
        "has_checkpoint": has_checkpoint or has_last_ckpt,
        "has_best_ckpt": has_checkpoint,
        "has_predictions": has_predictions,
        "n_features": n_features,
        "n_epochs": n_epochs,
        "best_val_loss": best_val_loss,
        "run_dir": run_dir,
        "timestamp": run_dir.name,
        "overrides": overrides,
    }


def _collect_runs(runs_root: Path, prefix: str | None) -> list[dict]:
    """collect all run info from artifacts/runs"""
    runs = []
    if not runs_root.exists():
        return runs
    for experiment_dir in sorted(runs_root.iterdir()):
        if not experiment_dir.is_dir():
            continue
        if prefix and not experiment_dir.name.startswith(prefix):
            continue
        for run_dir in sorted(experiment_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            info = _load_run_info(run_dir)
            if info is not None:
                runs.append(info)
    return runs


def _load_current_space_signatures() -> dict[str, tuple]:
    """load space signatures from current config yaml files"""
    configs_dir = Path("src/flatcfm/configs/space")
    sigs = {}
    for yaml_path in sorted(configs_dir.glob("*.yaml")):
        cfg = OmegaConf.to_container(OmegaConf.load(yaml_path), resolve=True)
        sigs[yaml_path.stem] = _space_signature(cfg)
    return sigs


def _load_experiment_configs(prefix: str | None) -> dict[str, dict]:
    """load experiment config names and their space overrides"""
    configs_dir = Path("src/flatcfm/configs/experiment")
    experiments = {}
    for yaml_path in sorted(configs_dir.rglob("*.yaml")):
        cfg = OmegaConf.load(yaml_path)
        name = str(cfg.get("experiment_name", yaml_path.stem))
        if prefix and not name.startswith(prefix):
            continue
        experiments[name] = {
            "config_path": str(yaml_path.relative_to(Path("src/flatcfm/configs/experiment"))),
        }
    return experiments


def _pick_latest_per_experiment(runs: list[dict]) -> list[dict]:
    """keep only the latest run per experiment name"""
    latest: dict[str, dict] = {}
    for run in runs:
        name = run["experiment_name"]
        if name not in latest or run["timestamp"] > latest[name]["timestamp"]:
            latest[name] = run
    return list(latest.values())


# -- display ------------------------------------------------------------------


def _show_current_configs(current_sigs: dict[str, tuple]) -> None:
    """show panel with current space configs"""
    lines = []
    for name, sig in current_sigs.items():
        label = _signature_label(sig)
        lines.append(f"  [bold cyan]{name:20s}[/]  {label}")
    panel = Panel(
        "\n".join(lines),
        title="Current Space Configs",
        border_style="blue",
    )
    console.print(panel)
    console.print()


def _make_run_table(runs: list[dict], dim: bool = False) -> Table:
    """build a rich table for a group of runs"""
    table = Table(show_header=True, header_style="bold", expand=False, padding=(0, 1))
    table.add_column("experiment", no_wrap=True)
    table.add_column("task", width=6)
    table.add_column("proj", width=10)
    table.add_column("ckpt", width=4, justify="center")
    table.add_column("pred", width=4, justify="center")
    table.add_column("ep", width=5, justify="right")
    table.add_column("val", width=8, justify="right")

    style = "dim" if dim else None

    for run in sorted(runs, key=lambda r: r["experiment_name"]):
        ckpt_text = Text("✓", style="green") if run["has_checkpoint"] else Text("✗", style="red")
        pred_text = Text("✓", style="green") if run["has_predictions"] else Text("✗", style="red")
        if dim:
            ckpt_text = Text("✓" if run["has_checkpoint"] else "✗", style="dim")
            pred_text = Text("✓" if run["has_predictions"] else "✗", style="dim")

        val_str = f"{run['best_val_loss']:.4f}" if run["best_val_loss"] is not None else "-"
        epoch_str = str(run["n_epochs"]) if run["n_epochs"] else "-"

        table.add_row(
            run["experiment_name"],
            run["task"],
            run["projections"],
            ckpt_text,
            pred_text,
            epoch_str,
            val_str,
            style=style,
        )
    return table


# -- main command -------------------------------------------------------------


@app.command()
def status(
    prefix: Annotated[str | None, typer.Option(help="filter experiments by name prefix")] = "sciplex",
    current_only: Annotated[bool, typer.Option("--current-only", help="show only current runs")] = False,
    actionable: Annotated[bool, typer.Option("--actionable", help="show only runs needing action")] = False,
) -> None:
    """show status of trained runs grouped by space compatibility"""
    runs_root = Path("artifacts/runs")
    all_runs = _collect_runs(runs_root, prefix)
    if not all_runs:
        console.print("[yellow]no runs found[/]")
        raise typer.Exit()

    # deduplicate to latest per experiment
    runs = _pick_latest_per_experiment(all_runs)

    # current space config signatures
    current_sigs = _load_current_space_signatures()
    current_sig_set = set(current_sigs.values())

    if not actionable:
        _show_current_configs(current_sigs)

    # group runs by signature
    sig_groups: dict[tuple, list[dict]] = defaultdict(list)
    for run in runs:
        sig_groups[run["signature"]].append(run)

    # separate current vs stale
    current_groups = {s: g for s, g in sig_groups.items() if s in current_sig_set}
    stale_groups = {s: g for s, g in sig_groups.items() if s not in current_sig_set}

    # display current groups
    group_num = 0
    if current_groups and not actionable:
        for sig, group in sorted(current_groups.items(), key=lambda x: _signature_label(x[0])):
            group_num += 1
            label = _signature_label(sig)
            console.rule(f"[bold green]Group {group_num}: CURRENT[/]", style="green")
            console.print(f"  {label}\n")
            console.print(_make_run_table(group))
            console.print()

    # display stale groups
    if stale_groups and not current_only and not actionable:
        for sig, group in sorted(stale_groups.items(), key=lambda x: _signature_label(x[0])):
            group_num += 1
            label = _signature_label(sig)
            console.rule(f"[dim]Group {group_num}: STALE[/]", style="dim")
            console.print(f"  [dim]{label}[/]\n")
            console.print(_make_run_table(group, dim=True))
            console.print()

    # actionable: missing predictions
    needs_predict = [
        r for r in runs
        if r["signature"] in current_sig_set
        and r["has_checkpoint"]
        and not r["has_predictions"]
    ]
    if needs_predict:
        console.rule("[bold yellow]Missing Predictions[/]", style="yellow")
        for run in sorted(needs_predict, key=lambda r: r["experiment_name"]):
            run_dir = run["run_dir"]
            console.print(
                f"  PYTHONPATH=src .venv/bin/python -m flatcfm.modelcore.predict "
                f"predict.run_dir={run_dir}"
            )
        console.print()

    # actionable: needs (re)training
    # 1) stale runs that need retraining with current config
    stale_runs = [r for r in runs if r["signature"] not in current_sig_set]
    # 2) experiment configs with no current run at all
    experiment_configs = _load_experiment_configs(prefix)
    current_experiment_names = {r["experiment_name"] for r in runs if r["signature"] in current_sig_set}
    stale_experiment_names = {r["experiment_name"] for r in stale_runs}

    needs_training: list[tuple[str, str]] = []

    # stale runs — use their saved hydra overrides
    for run in sorted(stale_runs, key=lambda r: r["experiment_name"]):
        if run["experiment_name"] in current_experiment_names:
            continue
        overrides_str = " ".join(run["overrides"]) if run["overrides"] else f"experiment_name={run['experiment_name']}"
        needs_training.append((run["experiment_name"], overrides_str))

    # experiment configs with no run at all (neither current nor stale)
    for exp_name, exp_info in sorted(experiment_configs.items()):
        if exp_name in current_experiment_names or exp_name in stale_experiment_names:
            continue
        config_rel = exp_info["config_path"].replace(".yaml", "")
        needs_training.append((exp_name, f"experiment={config_rel}"))

    if needs_training:
        console.rule("[bold red]Needs Training[/]", style="red")
        for exp_name, cmd_args in needs_training:
            console.print(
                f"  PYTHONPATH=src .venv/bin/python -m flatcfm.modelcore.train "
                f"{cmd_args}"
            )
        console.print()

    # summary
    n_current = sum(len(g) for g in current_groups.values())
    n_stale = sum(len(g) for g in stale_groups.values())
    console.print(
        f"[bold]{n_current}[/] current, "
        f"[dim]{n_stale}[/] stale, "
        f"[yellow]{len(needs_predict)}[/] need predictions, "
        f"[red]{len(needs_training)}[/] need training"
    )


if __name__ == "__main__":
    app()
