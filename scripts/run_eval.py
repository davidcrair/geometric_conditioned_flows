#!/usr/bin/env python
"""quick cli to evaluate trained runs: loss curves, metric tables, run comparison"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False)
console = Console()

RUNS_ROOT = Path("artifacts/runs")

# metrics where higher is better
HIGHER_IS_BETTER = {"cosine_log_fc", "top_k_recall", "deg_jaccard", "deg_overlap"}


# -- run path resolution ------------------------------------------------------


def _resolve_run_path(raw: str) -> Path:
    """resolve a run path from full path or bare experiment name"""

    p = Path(raw)
    if p.is_dir() and (p / "run_config.yaml").exists():
        return p.resolve()
    # bare name: find latest timestamp subdir
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


def _resolve_all_run_paths(raw: str) -> list[Path]:
    """resolve all run dirs for a bare experiment name"""

    p = Path(raw)
    if p.is_dir() and (p / "run_config.yaml").exists():
        return [p.resolve()]
    if "/" not in raw:
        experiment_dir = RUNS_ROOT / raw
        if not experiment_dir.is_dir():
            console.print(f"[red]experiment directory not found: {experiment_dir}[/]")
            raise typer.Exit(1)
        subdirs = sorted(d for d in experiment_dir.iterdir() if d.is_dir())
        valid = [d.resolve() for d in subdirs if (d / "run_config.yaml").exists()]
        if not valid:
            console.print(f"[red]no valid runs in {experiment_dir}[/]")
            raise typer.Exit(1)
        return valid
    console.print(f"[red]invalid run path: {raw}[/]")
    raise typer.Exit(1)


def _experiment_name(run_dir: Path) -> str:
    """extract experiment name from run_config.yaml"""

    from omegaconf import OmegaConf

    cfg = OmegaConf.load(run_dir / "run_config.yaml")
    return str(cfg.get("experiment_name", run_dir.parent.name))


def _short_name(run_dir: Path) -> str:
    """short display name from experiment name"""

    name = _experiment_name(run_dir)
    # strip common prefix for brevity
    for prefix in ("sciplex_", "norman_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


# -- loss curves ---------------------------------------------------------------


def _show_loss_curves(run_dirs: list[Path]) -> None:
    """render terminal loss curves with plotext"""

    import plotext as plt

    n = len(run_dirs)
    if n == 1:
        run_dir = run_dirs[0]
        history = json.loads((run_dir / "history.json").read_text())
        train_loss = history.get("train_loss", [])
        val_loss = history.get("val_loss", [])
        if not train_loss and not val_loss:
            console.print("[yellow]no loss history found[/]")
            return
        name = _experiment_name(run_dir)
        epochs = list(range(1, max(len(train_loss), len(val_loss)) + 1))
        plt.clear_figure()
        if train_loss:
            plt.plot(epochs[: len(train_loss)], train_loss, label="train")
        if val_loss:
            plt.plot(epochs[: len(val_loss)], val_loss, label="val")
        plt.title(f"{name} ({len(val_loss)} epochs)")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.plotsize(80, 20)
        plt.show()
        console.print()
    else:
        for run_dir in run_dirs:
            history = json.loads((run_dir / "history.json").read_text())
            train_loss = history.get("train_loss", [])
            val_loss = history.get("val_loss", [])
            name = _short_name(run_dir)
            epochs = list(range(1, max(len(train_loss), len(val_loss)) + 1))
            plt.clear_figure()
            if train_loss:
                plt.plot(epochs[: len(train_loss)], train_loss, label="train")
            if val_loss:
                plt.plot(epochs[: len(val_loss)], val_loss, label="val")
            plt.title(f"{name} ({len(val_loss)} epochs)")
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.plotsize(80, 15)
            plt.show()
        console.print()


# -- loss comparison table -----------------------------------------------------


def _show_loss_comparison_table(run_dirs: list[Path]) -> None:
    """show a quick table comparing final losses across runs"""

    table = Table(title="Loss Comparison", show_header=True, header_style="bold")
    table.add_column("run", no_wrap=True)
    table.add_column("epochs", justify="right")
    table.add_column("best val loss", justify="right")
    table.add_column("final train loss", justify="right")
    table.add_column("final val loss", justify="right")

    for run_dir in run_dirs:
        history = json.loads((run_dir / "history.json").read_text())
        train_loss = history.get("train_loss", [])
        val_loss = history.get("val_loss", [])
        name = _short_name(run_dir)
        n_epochs = str(len(val_loss))
        best_val = f"{min(val_loss):.4f}" if val_loss else "-"
        final_train = f"{train_loss[-1]:.4f}" if train_loss else "-"
        final_val = f"{val_loss[-1]:.4f}" if val_loss else "-"
        table.add_row(name, n_epochs, best_val, final_train, final_val)

    console.print(table)
    console.print()


# -- benchmark metrics ---------------------------------------------------------


def _load_and_evaluate(
    run_dir: Path,
    metrics: tuple[str, ...],
    group_columns: tuple[str, ...],
    reductions: tuple[str, ...],
    prediction_name: str = "held_out",
) -> dict:
    """load run and evaluate predictions"""

    from flatcfm.analysis.flow_results import load_flow_run, load_flow_predictions
    from flatcfm.analysis.benchmarking import evaluate_flow_predictions
    from flatcfm.analysis.benchmarks import MetricSpaceSpec

    name = _experiment_name(run_dir)
    console.print(f"[dim]loading {name} ({prediction_name})...[/]")
    bundle = load_flow_run(run_dir=str(run_dir))

    # for non-default prediction names read the saved spec so validation passes
    prediction_overrides = None
    if prediction_name != "held_out":
        meta_path = run_dir / "predictions" / prediction_name / "prediction_metadata.json"
        if meta_path.exists():
            prediction_overrides = json.loads(meta_path.read_text()).get("prediction_spec")

    predictions = load_flow_predictions(bundle, prediction_name, prediction_overrides=prediction_overrides)
    result = evaluate_flow_predictions(
        bundle,
        predictions,
        metric_space=MetricSpaceSpec(
            name="comparison_hvg", kind="comparison", fit_split="train"
        ),
        metrics=metrics,
        group_columns=group_columns,
        reductions=reductions,
        model_name=name,
    )
    return result


def _format_mean_std(mean: float, std: float) -> str:
    """format mean +/- std with adaptive precision"""

    if abs(mean) >= 100:
        return f"{mean:.1f} \u00b1 {std:.1f}"
    if abs(mean) >= 1:
        return f"{mean:.2f} \u00b1 {std:.2f}"
    return f"{mean:.4f} \u00b1 {std:.4f}"


def _metric_direction(metric_name: str) -> str:
    """arrow indicating whether higher or lower is better"""

    base = metric_name.split("@")[0]
    if base in HIGHER_IS_BETTER:
        return "\u2191"
    return "\u2193"


def _show_per_group_stats(per_group, metrics: tuple[str, ...]) -> None:
    """show per-group mean +/- std table"""

    import pandas as pd

    if per_group.empty:
        console.print("[yellow]no per-group metrics[/]")
        return

    # compute stats per metric_base
    stats = (
        per_group.groupby("metric_base")["value"]
        .agg(["mean", "std", "count"])
        .reindex([m for m in metrics if m in per_group["metric_base"].unique()])
    )
    n_groups = int(stats["count"].iloc[0]) if not stats.empty else 0

    table = Table(
        title=f"Per-Group Statistics ({n_groups} groups)",
        show_header=True,
        header_style="bold",
    )
    for metric_name in stats.index:
        direction = _metric_direction(metric_name)
        table.add_column(f"{metric_name} {direction}", justify="center")

    row = []
    for metric_name in stats.index:
        mean = stats.loc[metric_name, "mean"]
        std = stats.loc[metric_name, "std"]
        row.append(_format_mean_std(mean, std))
    table.add_row(*row)

    console.print(table)
    console.print()


# -- dose breakdown ------------------------------------------------------------


def _show_dose_breakdown(per_group, metrics: tuple[str, ...]) -> None:
    """show metric breakdown by dose level"""

    import pandas as pd
    import numpy as np

    if per_group.empty:
        console.print("[yellow]no per-group metrics for dose breakdown[/]")
        return

    # extract dose from product_dose column
    if "product_dose" not in per_group.columns:
        console.print("[yellow]no product_dose column; skipping dose breakdown[/]")
        return

    df = per_group.copy()
    df["dose"] = (
        df["product_dose"]
        .astype(str)
        .str.rsplit("_", n=1)
        .str[-1]
        .astype(float)
    )

    table = Table(
        title="Dose Breakdown",
        show_header=True,
        header_style="bold",
    )
    table.add_column("dose", justify="right")
    metric_bases = [m for m in metrics if m in df["metric_base"].unique()]
    for metric_name in metric_bases:
        direction = _metric_direction(metric_name)
        table.add_column(f"{metric_name} {direction}", justify="center")

    for dose in sorted(df["dose"].unique()):
        dose_df = df[df["dose"] == dose]
        row = [f"{dose:g}"]
        for metric_name in metric_bases:
            metric_df = dose_df[dose_df["metric_base"] == metric_name]
            if metric_df.empty:
                row.append("-")
            else:
                mean = metric_df["value"].mean()
                std = metric_df["value"].std()
                row.append(_format_mean_std(mean, std))
        table.add_row(*row)

    console.print(table)
    console.print()


# -- multi-run comparison ------------------------------------------------------


def _show_comparison_table(
    results: dict[str, dict],
    metrics: tuple[str, ...],
    reduction: str,
) -> None:
    """show side-by-side comparison table for multiple runs"""

    import numpy as np

    run_names = list(results.keys())

    # collect per-metric stats from per_group for each run
    metric_bases = []
    for m in metrics:
        for name in run_names:
            per_group = results[name]["per_group"]
            if m in per_group["metric_base"].unique():
                if m not in metric_bases:
                    metric_bases.append(m)
                break

    if not metric_bases:
        console.print("[yellow]no shared metrics for comparison[/]")
        return

    # for each run x metric, compute mean +/- std
    stats = {}
    for name in run_names:
        per_group = results[name]["per_group"]
        run_stats = {}
        for m in metric_bases:
            mdf = per_group[per_group["metric_base"] == m]
            if not mdf.empty:
                run_stats[m] = (mdf["value"].mean(), mdf["value"].std())
            else:
                run_stats[m] = (float("nan"), float("nan"))
        stats[name] = run_stats

    # find n_groups from first run
    first_pg = results[run_names[0]]["per_group"]
    if not first_pg.empty:
        n_groups = first_pg.groupby("metric_base").size().iloc[0]
    else:
        n_groups = 0

    table = Table(
        title=f"Metric Comparison ({reduction}, {n_groups} groups)",
        show_header=True,
        header_style="bold",
    )
    table.add_column("metric", no_wrap=True)
    for name in run_names:
        table.add_column(name, justify="center")

    for m in metric_bases:
        direction = _metric_direction(m)
        higher_better = m.split("@")[0] in HIGHER_IS_BETTER
        # find best
        means = {name: stats[name][m][0] for name in run_names}
        if higher_better:
            best_val = max(means.values())
        else:
            best_val = min(means.values())

        row = [f"{m} {direction}"]
        for name in run_names:
            mean, std = stats[name][m]
            text = _format_mean_std(mean, std)
            if np.isclose(mean, best_val):
                text = f"[bold green]{text} *[/]"
            row.append(text)
        table.add_row(*row)

    console.print(table)
    console.print()


def _show_aggregate_table(
    results: dict[str, dict],
    metrics: tuple[str, ...],
) -> None:
    """show mean +/- std across all runs for each metric"""

    import numpy as np

    run_names = list(results.keys())

    metric_bases = []
    for m in metrics:
        for name in run_names:
            per_group = results[name]["per_group"]
            if m in per_group["metric_base"].unique():
                if m not in metric_bases:
                    metric_bases.append(m)
                break

    if not metric_bases:
        console.print("[yellow]no metrics for aggregate table[/]")
        return

    # for each run compute the per-group mean for each metric
    per_run_means: dict[str, list[float]] = {m: [] for m in metric_bases}
    for name in run_names:
        per_group = results[name]["per_group"]
        for m in metric_bases:
            mdf = per_group[per_group["metric_base"] == m]
            per_run_means[m].append(mdf["value"].mean() if not mdf.empty else float("nan"))

    table = Table(
        title=f"Aggregate ({len(run_names)} runs)",
        show_header=True,
        header_style="bold",
    )
    table.add_column("metric", no_wrap=True)
    table.add_column("mean \u00b1 std", justify="center")
    table.add_column("n_runs", justify="right")

    for m in metric_bases:
        vals = np.array(per_run_means[m])
        direction = _metric_direction(m)
        table.add_row(
            f"{m} {direction}",
            _format_mean_std(float(np.nanmean(vals)), float(np.nanstd(vals))),
            str(len(vals)),
        )

    console.print(table)
    console.print()


# -- main command --------------------------------------------------------------


@app.command()
def evaluate(
    run_paths: Annotated[
        list[str],
        typer.Argument(help="run directories or experiment names"),
    ],
    losses_only: Annotated[
        bool, typer.Option("--losses-only", "-l", help="show only loss curves")
    ] = False,
    no_losses: Annotated[
        bool, typer.Option("--no-losses", help="skip loss curves")
    ] = False,
    no_dose: Annotated[
        bool, typer.Option("--no-dose", help="skip dose breakdown table")
    ] = False,
    all_runs: Annotated[
        bool, typer.Option("--all-runs", "-a", help="evaluate all runs per experiment (not just latest)")
    ] = False,
    split: Annotated[
        str, typer.Option("--split", "-s", help="prediction split to evaluate (held_out or train_same_control_rule)")
    ] = "held_out",
    metrics: Annotated[
        str, typer.Option(help="comma-separated metric names")
    ] = "mean_gene_w1,w2_squared,cosine_log_fc",
    reduction: Annotated[
        str, typer.Option(help="reduction for comparison table")
    ] = "unweighted_mean",
) -> None:
    """evaluate one or more trained runs with loss curves and benchmark metrics"""

    # resolve all run dirs
    if all_runs:
        run_dirs = []
        for rp in run_paths:
            run_dirs.extend(_resolve_all_run_paths(rp))
    else:
        run_dirs = [_resolve_run_path(rp) for rp in run_paths]
    metric_names = tuple(m.strip() for m in metrics.split(","))
    multi = len(run_dirs) > 1

    # loss curves
    if not no_losses:
        if multi:
            _show_loss_comparison_table(run_dirs)
        _show_loss_curves(run_dirs)

    if losses_only:
        raise typer.Exit()

    # benchmark evaluation
    group_columns = ("cell_type", "product_dose")
    reductions = ("unweighted_mean", "cell_weighted_mean")

    if multi:
        results = {}
        for run_dir in run_dirs:
            name = _short_name(run_dir)
            result = _load_and_evaluate(run_dir, metric_names, group_columns, reductions, prediction_name=split)
            results[name] = result

        # per-run stats
        for name, result in results.items():
            console.rule(f"[bold]{name}[/]")
            _show_per_group_stats(result["per_group"], metric_names)
            if not no_dose:
                _show_dose_breakdown(result["per_group"], metric_names)

        # comparison table
        console.rule("[bold]Comparison[/]")
        _show_comparison_table(results, metric_names, reduction)

        # aggregate table when --all-runs was used
        if all_runs and len(results) > 1:
            console.rule("[bold]Aggregate (mean \u00b1 std across runs)[/]")
            _show_aggregate_table(results, metric_names)
    else:
        run_dir = run_dirs[0]
        result = _load_and_evaluate(run_dir, metric_names, group_columns, reductions, prediction_name=split)
        _show_per_group_stats(result["per_group"], metric_names)
        if not no_dose:
            _show_dose_breakdown(result["per_group"], metric_names)


if __name__ == "__main__":
    app()
