#!/usr/bin/env python
"""check status of AE dimensionality sweep and generate training commands"""

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

DIMS = [16, 32, 64, 128, 256, 512, 1024]

def _mse_wide_hidden(dim: int) -> int:
    """scale hidden dim with latent dim so min(hidden, latent) = latent

    rule: hidden = max(512, 2 * latent) ensures the hidden layers are never
    narrower than the latent and always give at least 2x headroom so cascaded
    nonlinear layers do not starve the bottleneck
    """

    return max(512, 2 * dim)


VARIANTS = {
    "mse": {
        "experiment": "sciplex/ae_log1p_mse_d512",
        "label": "MSE AE",
        "extra_overrides": "task.epochs=100 +model.output_activation=relu",
    },
    "mse_wide": {
        "experiment": "sciplex/ae_log1p_mse_wide",
        "label": "MSE AE (wide)",
        "extra_overrides_fn": lambda dim: f"model.hidden_dim={_mse_wide_hidden(dim)}",
    },
    "linear": {
        "experiment": "sciplex/ae_log1p_mse_wide",
        "label": "Linear AE",
        "extra_overrides": "model.family=linear",
    },
    "nb": {
        "experiment": "sciplex/ae_log1p_nb_pullback",
        "label": "NB AE",
        "extra_overrides": "loss.weights.pullback=0.0",
    },
    "flatvi": {
        "experiment": "sciplex/ae_log1p_nb_pullback",
        "label": "FlatVI AE",
    },
    "phate": {
        "experiment": "sciplex/ae_log1p_hybrid",
        "label": "PHATE AE",
    },
}


def _experiment_name(variant: str, dim: int) -> str:
    return f"sciplex_ae_deg_{variant}_d{dim}"


def _has_run(experiment_name: str) -> bool:
    run_root = RUNS_ROOT / experiment_name
    if not run_root.is_dir():
        return False
    return any(d.is_dir() for d in run_root.iterdir())


def _has_checkpoint(experiment_name: str) -> bool:
    run_root = RUNS_ROOT / experiment_name
    if not run_root.is_dir():
        return False
    subdirs = sorted(d for d in run_root.iterdir() if d.is_dir())
    if not subdirs:
        return False
    latest = subdirs[-1]
    ckpt_dir = latest / "checkpoints"
    if not ckpt_dir.is_dir():
        return False
    return (ckpt_dir / "best.ckpt").exists() or (ckpt_dir / "last.ckpt").exists()


def _has_projection(tag: str) -> bool:
    pattern = f"sciplex_ae_projection_*_{tag}.pkl"
    return any(SPACES_ROOT.glob(pattern))


def _training_command(variant: str, dim: int) -> str:
    exp_name = _experiment_name(variant, dim)
    base = VARIANTS[variant]["experiment"]
    extra_static = VARIANTS[variant].get("extra_overrides", "")
    extra_fn = VARIANTS[variant].get("extra_overrides_fn")
    extra_dyn = extra_fn(dim) if extra_fn is not None else ""
    extra = " ".join(s for s in (extra_static, extra_dyn) if s)
    return (
        f".venv/bin/python -m flatcfm.modelcore.train "
        f"experiment={base} "
        f"model.latent_dim={dim} "
        f"experiment_name={exp_name} "
        f"space.ae_export_artifact_tag={exp_name} "
        f"trainer.num_workers=12"
        f"{' ' + extra if extra else ''}"
    )


def _collect_missing() -> list[tuple[str, str, int, str]]:
    """collect missing (variant_key, label, dim, command) tuples"""

    missing = []
    for variant in VARIANTS:
        for dim in DIMS:
            exp_name = _experiment_name(variant, dim)
            if not _has_projection(exp_name):
                missing.append((variant, VARIANTS[variant]["label"], dim, _training_command(variant, dim)))
    return missing


@app.command()
def check(
    generate: Annotated[
        bool, typer.Option("--generate", "-g", help="print shell commands for missing models")
    ] = False,
    train: Annotated[
        bool, typer.Option("--train", "-t", help="train all missing models sequentially")
    ] = False,
) -> None:
    """check which AE dimensionality sweep models are trained"""

    table = Table(
        title="AE Dimensionality Sweep Status",
        show_header=True,
        header_style="bold",
    )
    table.add_column("variant", no_wrap=True)
    table.add_column("dim", justify="right")
    table.add_column("run", justify="center")
    table.add_column("checkpoint", justify="center")
    table.add_column("projection", justify="center")

    total = 0
    ready = 0

    for variant in VARIANTS:
        for dim in DIMS:
            total += 1
            exp_name = _experiment_name(variant, dim)
            has_run = _has_run(exp_name)
            has_ckpt = _has_checkpoint(exp_name)
            has_proj = _has_projection(exp_name)

            run_str = "[green]yes[/]" if has_run else "[red]no[/]"
            ckpt_str = "[green]yes[/]" if has_ckpt else "[red]no[/]"
            proj_str = "[green]yes[/]" if has_proj else "[red]no[/]"

            if has_proj:
                ready += 1

            table.add_row(
                VARIANTS[variant]["label"],
                str(dim),
                run_str,
                ckpt_str,
                proj_str,
            )

    console.print(table)
    console.print(f"\n[bold]{ready}/{total}[/] models ready (have projection pickle)")

    missing = _collect_missing()

    if not missing:
        console.print("[green]all models trained[/]")
        return

    console.print(f"[yellow]{len(missing)} models need training[/]")

    if generate:
        console.print("\n[bold]training commands:[/]\n")
        for _, _, _, cmd in missing:
            console.print(cmd)

    if train:
        import subprocess
        import sys

        for i, (variant_key, label, dim, cmd) in enumerate(missing):
            console.rule(f"[bold][{i + 1}/{len(missing)}] {label} d={dim}[/]")
            args = cmd.split()
            # replace .venv/bin/python with current interpreter
            args[0] = sys.executable
            result = subprocess.run(args)
            if result.returncode != 0:
                console.print(f"[red]FAILED: {label} d={dim} (exit code {result.returncode})[/]")
                console.print("[yellow]continuing to next model...[/]")
            else:
                console.print(f"[green]DONE: {label} d={dim}[/]")
            console.print()

        # recount after training
        done = sum(1 for v in VARIANTS for d in DIMS if _has_projection(_experiment_name(v, d)))
        console.print(f"\n[bold]{done}/{total}[/] models ready after training")


if __name__ == "__main__":
    app()
