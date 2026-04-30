#!/usr/bin/env python
"""check status of the disjoint 3 cell type partition split sweep across
distance loss weights and seeds

trains one hybrid AE per (weight seed) on the unified disjoint partition
split where the held out set is the union of 3 disjoint per cell type
buckets each cell type gets 30% of its drug universe held out and a
given drug is held out for at most one cell type then trains FM and ODE
in that single AE latent space

splitter is configs/splitter/disjoint3.yaml with split_policy=disjoint_partition
weight grid is 0 1e-4 1e-2 1 1e2 to span 6 orders of magnitude plus pure
recon seed grid is 42 43 44

with --generate prints the sbatch invocation for the missing array indices
in the unified train_disjoint3_ae_latent.sbatch which is laid out as
task = weight_idx*3 + seed_idx
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

WEIGHTS = [0.0, 1e-4, 1e-2, 1.0, 1e2]
SEEDS = [42, 43, 44]
MODEL_VARIANTS = ["fm", "ode"]

AE_BASE_EXPERIMENT = "sciplex/ae_log1p_hybrid"
FM_BASE_EXPERIMENT = "sciplex/fm_ae_latent_hybrid"
ODE_BASE_EXPERIMENT = "sciplex/ode_ae_latent_hybrid_d1e3"
SBATCH_PATH = "scripts/train_disjoint3_ae_latent.sbatch"


def _weight_tag(weight: float) -> str:
    """build weight tag like w0p01 w1e-4 etc must match the sbatch TAGS array"""

    if weight == 0:
        return "w0"
    return "w" + f"{weight:g}".replace(".", "p")


def _array_index(weight: float, seed: int) -> int:
    """sbatch array index for (weight seed) matches task = weight_idx*3 + seed_idx"""

    return WEIGHTS.index(weight) * len(SEEDS) + SEEDS.index(seed)


def _ae_experiment_name(weight: float, seed: int) -> str:
    return f"sciplex_ae_disjoint3_{_weight_tag(weight)}_s{seed}"


def _model_experiment_name(variant: str, weight: float, seed: int) -> str:
    return f"sciplex_{variant}_disjoint3_{_weight_tag(weight)}_s{seed}"


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


def _format_array_indices(indices: list[int]) -> str:
    """compact slurm array spec collapsing consecutive runs into ranges"""

    if not indices:
        return ""
    sorted_idx = sorted(set(indices))
    parts: list[str] = []
    start = prev = sorted_idx[0]
    for x in sorted_idx[1:]:
        if x == prev + 1:
            prev = x
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = x
    parts.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(parts)


@app.command()
def check(
    generate: Annotated[bool, typer.Option("--generate", "-g", help="print sbatch command for missing combos")] = False,
    partition: Annotated[
        str, typer.Option("--partition", "-p", help="slurm partition for --generate")
    ] = "gpu_rtx6000",
) -> None:
    """check which (weight seed model) combinations are trained and predicted"""

    table = Table(
        title="Disjoint3 Partition Split AE Latent Sweep Status",
        show_header=True,
        header_style="bold",
    )
    table.add_column("distance weight", justify="right")
    table.add_column("seed", justify="right")
    table.add_column("ae proj", justify="center")
    table.add_column("fm run", justify="center")
    table.add_column("fm ckpt", justify="center")
    table.add_column("fm pred", justify="center")
    table.add_column("ode run", justify="center")
    table.add_column("ode ckpt", justify="center")
    table.add_column("ode pred", justify="center")

    total_models = 0
    ready_models = 0

    for weight in WEIGHTS:
        for seed in SEEDS:
            ae_ok = _has_ae_projection(weight, seed)
            row = [f"{weight:g}", str(seed), "[green]yes[/]" if ae_ok else "[red]no[/]"]
            for variant in MODEL_VARIANTS:
                exp = _model_experiment_name(variant, weight, seed)
                run_ok = _has_run(exp)
                ckpt_ok = _has_checkpoint(exp)
                pred_ok = _has_predictions(exp)
                row.append("[green]yes[/]" if run_ok else "[red]no[/]")
                row.append("[green]yes[/]" if ckpt_ok else "[red]no[/]")
                row.append("[green]yes[/]" if pred_ok else "[red]no[/]")
                total_models += 1
                if pred_ok:
                    ready_models += 1
            table.add_row(*row)

    console.print(table)
    console.print(
        f"\n[bold]{ready_models}/{total_models}[/] (weight seed model) combos ready (have predictions)"
    )

    # array task is missing if any model variant for that (weight seed) lacks predictions
    missing_indices: list[int] = []
    for weight in WEIGHTS:
        for seed in SEEDS:
            any_missing = any(
                not _has_predictions(_model_experiment_name(v, weight, seed))
                for v in MODEL_VARIANTS
            )
            if any_missing:
                missing_indices.append(_array_index(weight, seed))

    if not missing_indices:
        console.print("[green]all combos trained and predicted[/]")
        return

    console.print(f"[yellow]{len(missing_indices)} array tasks need work[/]")

    if generate:
        spec = _format_array_indices(missing_indices)
        console.print("\n[bold]sbatch command:[/]\n")
        console.print(f"sbatch --partition={partition} --array={spec} {SBATCH_PATH}")
        console.print()
        console.print("[dim]missing tasks (task = weight_idx*3 + seed_idx):[/dim]")
        for idx in sorted(missing_indices):
            w_idx, s_idx = divmod(idx, len(SEEDS))
            console.print(
                f"[dim]  task {idx:>2}  weight={WEIGHTS[w_idx]:g}  seed={SEEDS[s_idx]}[/dim]"
            )


if __name__ == "__main__":
    app()
