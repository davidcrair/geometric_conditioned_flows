#!/usr/bin/env python
"""view perturbench evaluation results across all runs"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False)
console = Console(width=200, force_terminal=True)

DEFAULT_LOGS_ROOT = Path("/home/dac227/scratch_pi_sk2433/dac227/perturbench/logs/train/runs")

# perturbench metrics where higher is better
PB_HIGHER_IS_BETTER = {
    "cosine_pca_average",
    "cosine_logfc",
    "r2_score_scores",
    "top_k_recall_scores",
    "sig_deg_recall_pvals_adj",
}

# default display metrics (skip rank metrics for cleaner table)
DEFAULT_METRICS = (
    "cosine_logfc",
    "mmd_pca",
    "cosine_pca_average",
    "rmse_average",
    "r2_score_scores",
    "top_k_recall_scores",
    "sig_deg_recall_pvals_adj",
)


def _load_summary(summary_path: Path) -> tuple[str, dict[str, float]]:
    """load a perturbench summary csv returning (model_name, {metric: value})"""

    with open(summary_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        model_name = header[1] if len(header) > 1 else "unknown"
        metrics = {}
        for row in reader:
            if len(row) >= 2:
                metrics[row[0]] = float(row[1])
    return model_name, metrics


def _scan_runs(logs_root: Path) -> list[dict]:
    """scan all runs with evaluation summaries"""

    results = []
    for run_dir in sorted(logs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        summary = run_dir / "evaluation" / "summary.csv"
        if not summary.exists():
            continue
        model_name, metrics = _load_summary(summary)
        results.append({
            "run_dir": run_dir,
            "timestamp": run_dir.name,
            "model_name": model_name,
            "metrics": metrics,
        })
    return results


def _metric_direction(metric: str) -> str:
    """arrow for metric direction"""

    return "\u2191" if metric in PB_HIGHER_IS_BETTER else "\u2193"


def _format_val(val: float) -> str:
    """format a metric value"""

    if abs(val) >= 1:
        return f"{val:.4f}"
    return f"{val:.4f}"


@app.command()
def show(
    logs_root: Annotated[
        Path, typer.Option("--logs", "-l", help="perturbench logs/train/runs directory")
    ] = DEFAULT_LOGS_ROOT,
    all_metrics: Annotated[
        bool, typer.Option("--all-metrics", help="show all metrics including rank metrics")
    ] = False,
) -> None:
    """show perturbench results for all evaluated runs"""

    if not logs_root.is_dir():
        console.print(f"[red]logs directory not found: {logs_root}[/]")
        raise typer.Exit(1)

    results = _scan_runs(logs_root)
    if not results:
        console.print("[yellow]no evaluation summaries found[/]")
        raise typer.Exit(0)

    # collect all metric names across runs
    all_metric_names = set()
    for r in results:
        all_metric_names.update(r["metrics"].keys())

    if all_metrics:
        display_metrics = sorted(all_metric_names)
    else:
        display_metrics = [m for m in DEFAULT_METRICS if m in all_metric_names]

    import math

    def _rank_values(vals: list[float], higher_is_better: bool) -> tuple[float, float | None]:
        """return (best, second_best) from a list of values"""
        valid = sorted(set(v for v in vals if not math.isnan(v)), reverse=higher_is_better)
        best = valid[0] if valid else float("nan")
        second = valid[1] if len(valid) > 1 else None
        return best, second

    # find best and second best per metric across all per-run values
    per_run_ranks: dict[str, tuple[float, float | None]] = {}
    for metric in display_metrics:
        higher = metric in PB_HIGHER_IS_BETTER
        vals = [r["metrics"].get(metric, float("nan")) for r in results]
        per_run_ranks[metric] = _rank_values(vals, higher)

    def _style_val(val: float, best: float, second: float | None) -> str:
        """format value with bold for best and underline for second best"""
        text = _format_val(val)
        if math.isclose(val, best):
            return f"[bold]{text}[/]"
        if second is not None and math.isclose(val, second):
            return f"[underline]{text}[/]"
        return text

    # build per-run table
    table = Table(
        title="PerturBench Results",
        show_header=True,
        header_style="bold",
    )
    table.add_column("model", no_wrap=True)
    table.add_column("run", no_wrap=True, style="dim")
    for metric in display_metrics:
        direction = _metric_direction(metric)
        table.add_column(f"{metric} {direction}", justify="right")

    for r in results:
        row = [r["model_name"], r["timestamp"]]
        for metric in display_metrics:
            val = r["metrics"].get(metric, float("nan"))
            if math.isnan(val):
                row.append("-")
            else:
                best, second = per_run_ranks[metric]
                row.append(_style_val(val, best, second))
        table.add_row(*row)

    console.print(table)

    # aggregate by model name
    from collections import defaultdict

    import numpy as np

    model_runs: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        model_runs[r["model_name"]].append(r)

    has_multi = any(len(runs) > 1 for runs in model_runs.values())
    if has_multi:
        # compute aggregate means per model for ranking
        agg_means: dict[str, dict[str, float]] = {}
        for model_name, runs in model_runs.items():
            agg_means[model_name] = {}
            for metric in display_metrics:
                vals = [r["metrics"].get(metric, float("nan")) for r in runs]
                vals = [v for v in vals if not math.isnan(v)]
                agg_means[model_name][metric] = float(np.mean(vals)) if vals else float("nan")

        # rank aggregate means
        agg_ranks: dict[str, tuple[float, float | None]] = {}
        for metric in display_metrics:
            higher = metric in PB_HIGHER_IS_BETTER
            means = [agg_means[m][metric] for m in agg_means]
            agg_ranks[metric] = _rank_values(means, higher)

        console.print()
        agg_table = Table(
            title="Aggregate (mean \u00b1 std across runs)",
            show_header=True,
            header_style="bold",
        )
        agg_table.add_column("model", no_wrap=True)
        agg_table.add_column("n", justify="right")
        for metric in display_metrics:
            direction = _metric_direction(metric)
            agg_table.add_column(f"{metric} {direction}", justify="right")

        for model_name in sorted(model_runs.keys()):
            runs = model_runs[model_name]
            row = [model_name, str(len(runs))]
            for metric in display_metrics:
                vals = [r["metrics"].get(metric, float("nan")) for r in runs]
                vals = [v for v in vals if not math.isnan(v)]
                best, second = agg_ranks[metric]
                if not vals:
                    row.append("-")
                elif len(vals) == 1:
                    row.append(_style_val(vals[0], best, second))
                else:
                    arr = np.array(vals)
                    mean, std = float(arr.mean()), float(arr.std())
                    text = f"{mean:.4f} \u00b1 {std:.4f}"
                    if math.isclose(mean, best):
                        text = f"[bold]{text}[/]"
                    elif second is not None and math.isclose(mean, second):
                        text = f"[underline]{text}[/]"
                    row.append(text)
            agg_table.add_row(*row)

        console.print(agg_table)


if __name__ == "__main__":
    app()
