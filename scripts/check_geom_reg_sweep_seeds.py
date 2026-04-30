#!/usr/bin/env python
"""check status of geometric regularization sweep across 3 seeds and a wider weight grid

trains hybrid AE at each (seed, distance loss weight) then trains FM in each AE
latent space to compare downstream perturbation prediction performance vs
geometric regularization strength with seed level error bars

weight grid is 0, 1e-4, 1e-2, 1, 1e2 to span 6 orders of magnitude
seed grid is 42, 43, 44 (seed 42 has no _s42 suffix to match the existing
naming convention used by train_geom_reg_sweep.sbatch and friends)
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False)
console = Console()

RUNS_ROOT = Path("artifacts/runs")
SPACES_ROOT = Path("artifacts/spaces")

# distance loss weights spanning 6 orders of magnitude plus pure recon
WEIGHTS = [0.0, 1e-4, 1e-2, 1.0, 1e2]
SEEDS = [42, 43, 44]

AE_BASE_EXPERIMENT = "sciplex/ae_log1p_hybrid"
FM_BASE_EXPERIMENT = "sciplex/fm_ae_latent_hybrid"


def _weight_tag(weight: float) -> str:
    """build weight tag like w0p01 w1e-4 etc"""

    if weight == 0:
        return "w0"
    return "w" + f"{weight:g}".replace(".", "p")


def _seed_suffix(seed: int) -> str:
    """seed 42 gets no suffix to match the original sweep naming"""

    return "" if seed == 42 else f"_s{seed}"


def _ae_experiment_name(weight: float, seed: int) -> str:
    return f"sciplex_ae_deg_geom_reg_{_weight_tag(weight)}{_seed_suffix(seed)}"


def _fm_experiment_name(weight: float, seed: int) -> str:
    return f"sciplex_fm_deg_geom_reg_{_weight_tag(weight)}{_seed_suffix(seed)}"


def _has_ae_projection(weight: float, seed: int) -> bool:
    tag = _ae_experiment_name(weight, seed)
    return any(SPACES_ROOT.glob(f"sciplex_ae_projection_*_{tag}.pkl"))


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


def _ae_train_command(weight: float, seed: int) -> str:
    exp_name = _ae_experiment_name(weight, seed)
    return (
        f".venv/bin/python -m flatcfm.modelcore.train "
        f"experiment={AE_BASE_EXPERIMENT} "
        f"experiment_name={exp_name} "
        f"space.ae_export_artifact_tag={exp_name} "
        f"splitter.seed={seed} "
        f"loss.weights.distance={weight} "
        f"trainer.num_workers=12"
    )


def _fm_train_command(weight: float, seed: int) -> str:
    fm_exp = _fm_experiment_name(weight, seed)
    ae_tag = _ae_experiment_name(weight, seed)
    return (
        f".venv/bin/python -m flatcfm.modelcore.train "
        f"experiment={FM_BASE_EXPERIMENT} "
        f"experiment_name={fm_exp} "
        f"splitter.seed={seed} "
        f"space.projections.0.artifact_tag={ae_tag} "
        f"trainer.num_workers=12"
    )


def _predict_command(run_dir: Path) -> str:
    return f".venv/bin/python -m flatcfm.modelcore.predict predict.run_dir={run_dir}"


@app.command()
def check(
    generate: Annotated[bool, typer.Option("--generate", "-g", help="print commands for missing models")] = False,
    train: Annotated[
        bool, typer.Option("--train", "-t", help="train and predict all missing models sequentially")
    ] = False,
) -> None:
    """check which (weight seed) combinations are trained and predicted"""

    table = Table(
        title="Geometric Regularization Sweep x Seeds Status",
        show_header=True,
        header_style="bold",
    )
    table.add_column("distance weight", justify="right")
    table.add_column("seed", justify="right")
    table.add_column("ae projection", justify="center")
    table.add_column("fm run", justify="center")
    table.add_column("fm checkpoint", justify="center")
    table.add_column("fm predictions", justify="center")

    total = 0
    ready = 0

    for weight in WEIGHTS:
        for seed in SEEDS:
            total += 1
            ae_ok = _has_ae_projection(weight, seed)
            fm_exp = _fm_experiment_name(weight, seed)
            fm_run = _has_run(fm_exp)
            fm_ckpt = _has_checkpoint(fm_exp)
            fm_pred = _has_predictions(fm_exp)

            ae_str = "[green]yes[/]" if ae_ok else "[red]no[/]"
            run_str = "[green]yes[/]" if fm_run else "[red]no[/]"
            ckpt_str = "[green]yes[/]" if fm_ckpt else "[red]no[/]"
            pred_str = "[green]yes[/]" if fm_pred else "[red]no[/]"

            if fm_pred:
                ready += 1

            table.add_row(f"{weight:g}", str(seed), ae_str, run_str, ckpt_str, pred_str)

    console.print(table)
    console.print(f"\n[bold]{ready}/{total}[/] (weight seed) combos ready (have FM predictions)")

    missing = [
        (w, s)
        for w in WEIGHTS
        for s in SEEDS
        if not _has_predictions(_fm_experiment_name(w, s))
    ]

    if not missing:
        console.print("[green]all (weight seed) combos trained and predicted[/]")
        return

    console.print(f"[yellow]{len(missing)} combos need work[/]")

    if generate:
        console.print("\n[bold]commands:[/]\n")
        for weight, seed in missing:
            ae_exp = _ae_experiment_name(weight, seed)
            fm_exp = _fm_experiment_name(weight, seed)
            if not _has_ae_projection(weight, seed):
                console.print(f"# train AE distance weight={weight} seed={seed}")
                console.print(_ae_train_command(weight, seed))
            if not _has_checkpoint(fm_exp):
                console.print(f"# train FM in AE latent space distance weight={weight} seed={seed}")
                console.print(_fm_train_command(weight, seed))
            run_dir = _latest_run_dir(fm_exp)
            if run_dir and _has_checkpoint(fm_exp) and not _has_predictions(fm_exp):
                console.print(f"# predict FM distance weight={weight} seed={seed}")
                console.print(_predict_command(run_dir))
            console.print()

    if train:
        import subprocess
        import sys

        for i, (weight, seed) in enumerate(missing):
            ae_exp = _ae_experiment_name(weight, seed)
            fm_exp = _fm_experiment_name(weight, seed)

            if not _has_ae_projection(weight, seed):
                console.rule(
                    f"[bold][{i + 1}/{len(missing)}] train AE distance weight={weight} seed={seed}[/]"
                )
                args = _ae_train_command(weight, seed).split()
                args[0] = sys.executable
                result = subprocess.run(args)
                if result.returncode != 0:
                    console.print(f"[red]FAILED AE train weight={weight} seed={seed}[/]")
                    console.print("[yellow]continuing...[/]")
                    continue
                console.print(f"[green]DONE AE train weight={weight} seed={seed}[/]")

            if not _has_checkpoint(fm_exp):
                console.rule(
                    f"[bold][{i + 1}/{len(missing)}] train FM weight={weight} seed={seed}[/]"
                )
                args = _fm_train_command(weight, seed).split()
                args[0] = sys.executable
                result = subprocess.run(args)
                if result.returncode != 0:
                    console.print(f"[red]FAILED FM train weight={weight} seed={seed}[/]")
                    console.print("[yellow]continuing...[/]")
                    continue
                console.print(f"[green]DONE FM train weight={weight} seed={seed}[/]")

            run_dir = _latest_run_dir(fm_exp)
            if run_dir and _has_checkpoint(fm_exp) and not _has_predictions(fm_exp):
                console.rule(
                    f"[bold][{i + 1}/{len(missing)}] predict FM weight={weight} seed={seed}[/]"
                )
                pred_args = _predict_command(run_dir).split()
                pred_args[0] = sys.executable
                result = subprocess.run(pred_args)
                if result.returncode != 0:
                    console.print(f"[red]FAILED predict weight={weight} seed={seed}[/]")
                else:
                    console.print(f"[green]DONE predict weight={weight} seed={seed}[/]")
            console.print()

        done = sum(
            1
            for w in WEIGHTS
            for s in SEEDS
            if _has_predictions(_fm_experiment_name(w, s))
        )
        console.print(f"\n[bold]{done}/{total}[/] combos ready after training")


if __name__ == "__main__":
    app()
