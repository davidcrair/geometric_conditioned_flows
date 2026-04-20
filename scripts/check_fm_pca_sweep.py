#!/usr/bin/env python
"""check status of FM in PCA dimension sweep and generate train+predict commands"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False)
console = Console()

RUNS_ROOT = Path("artifacts/runs")
DIMS = [16, 32, 64, 128, 256, 512, 1024]
BASE_EXPERIMENT = "sciplex/fm_pca_deg"


def _experiment_name(dim: int) -> str:
    return f"sciplex_fm_deg_pca_d{dim}"


def _latest_run_dir(experiment_name: str) -> Path | None:
    run_root = RUNS_ROOT / experiment_name
    if not run_root.is_dir():
        return None
    subdirs = sorted(d for d in run_root.iterdir() if d.is_dir())
    return subdirs[-1] if subdirs else None


def _has_run(experiment_name: str) -> bool:
    return _latest_run_dir(experiment_name) is not None


def _has_checkpoint(experiment_name: str) -> bool:
    run_dir = _latest_run_dir(experiment_name)
    if run_dir is None:
        return False
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return False
    return (ckpt_dir / "best.ckpt").exists() or (ckpt_dir / "last.ckpt").exists()


def _has_predictions(experiment_name: str) -> bool:
    run_dir = _latest_run_dir(experiment_name)
    if run_dir is None:
        return False
    return (run_dir / "predictions" / "held_out" / "predictions.h5ad").exists()


def _train_command(dim: int) -> str:
    exp_name = _experiment_name(dim)
    return (
        f".venv/bin/python -m flatcfm.modelcore.train "
        f"experiment={BASE_EXPERIMENT} "
        f"experiment_name={exp_name} "
        f"space.projections.0.n_components={dim} "
        f"trainer.num_workers=12"
    )


def _predict_command(run_dir: Path) -> str:
    return f".venv/bin/python -m flatcfm.modelcore.predict predict.run_dir={run_dir}"


def _collect_missing() -> list[tuple[int, str, str | None]]:
    """collect missing (dim, train_cmd, predict_cmd_or_none) tuples"""

    missing = []
    for dim in DIMS:
        exp_name = _experiment_name(dim)
        if not _has_predictions(exp_name):
            train_cmd = _train_command(dim) if not _has_checkpoint(exp_name) else None
            run_dir = _latest_run_dir(exp_name)
            predict_cmd = _predict_command(run_dir) if run_dir and _has_checkpoint(exp_name) else None
            missing.append((dim, train_cmd, predict_cmd))
    return missing


@app.command()
def check(
    generate: Annotated[
        bool, typer.Option("--generate", "-g", help="print commands for missing models")
    ] = False,
    train: Annotated[
        bool, typer.Option("--train", "-t", help="train and predict all missing models sequentially")
    ] = False,
) -> None:
    """check which FM PCA dimension sweep models are trained and predicted"""

    table = Table(
        title="FM PCA Dimension Sweep Status",
        show_header=True,
        header_style="bold",
    )
    table.add_column("dim", justify="right")
    table.add_column("run", justify="center")
    table.add_column("checkpoint", justify="center")
    table.add_column("predictions", justify="center")

    total = 0
    ready = 0

    for dim in DIMS:
        total += 1
        exp_name = _experiment_name(dim)
        has_run = _has_run(exp_name)
        has_ckpt = _has_checkpoint(exp_name)
        has_pred = _has_predictions(exp_name)

        run_str = "[green]yes[/]" if has_run else "[red]no[/]"
        ckpt_str = "[green]yes[/]" if has_ckpt else "[red]no[/]"
        pred_str = "[green]yes[/]" if has_pred else "[red]no[/]"

        if has_pred:
            ready += 1

        table.add_row(str(dim), run_str, ckpt_str, pred_str)

    console.print(table)
    console.print(f"\n[bold]{ready}/{total}[/] models ready (have predictions)")

    missing = _collect_missing()

    if not missing:
        console.print("[green]all models trained and predicted[/]")
        return

    console.print(f"[yellow]{len(missing)} models need work[/]")

    if generate:
        console.print("\n[bold]commands:[/]\n")
        for dim, train_cmd, predict_cmd in missing:
            if train_cmd:
                console.print(f"# train d={dim}")
                console.print(train_cmd)
            if predict_cmd:
                console.print(f"# predict d={dim}")
                console.print(predict_cmd)
            elif train_cmd:
                console.print(f"# predict d={dim}: run train first")
            console.print()

    if train:
        import subprocess
        import sys

        for i, (dim, train_cmd, _) in enumerate(missing):
            exp_name = _experiment_name(dim)

            # train if needed
            if not _has_checkpoint(exp_name):
                console.rule(f"[bold][{i + 1}/{len(missing)}] train FM PCA d={dim}[/]")
                args = _train_command(dim).split()
                args[0] = sys.executable
                result = subprocess.run(args)
                if result.returncode != 0:
                    console.print(f"[red]FAILED train d={dim}[/]")
                    console.print("[yellow]continuing...[/]")
                    continue
                console.print(f"[green]DONE train d={dim}[/]")

            # predict
            run_dir = _latest_run_dir(exp_name)
            if run_dir and _has_checkpoint(exp_name) and not _has_predictions(exp_name):
                console.rule(f"[bold][{i + 1}/{len(missing)}] predict FM PCA d={dim}[/]")
                pred_args = _predict_command(run_dir).split()
                pred_args[0] = sys.executable
                result = subprocess.run(pred_args)
                if result.returncode != 0:
                    console.print(f"[red]FAILED predict d={dim}[/]")
                else:
                    console.print(f"[green]DONE predict d={dim}[/]")
            console.print()

        done = sum(1 for d in DIMS if _has_predictions(_experiment_name(d)))
        console.print(f"\n[bold]{done}/{total}[/] models ready after training")


if __name__ == "__main__":
    app()
