#!/usr/bin/env python
"""show free vs total gpus per slurm partition"""

from __future__ import annotations

import re
import subprocess
from collections import OrderedDict
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False)
console = Console()

_CFGTRES_GPU_RE = re.compile(r"CfgTRES=\S*?gres/gpu=(\d+)")
_ALLOCTRES_GPU_RE = re.compile(r"AllocTRES=\S*?gres/gpu=(\d+)")
_CFGTRES_CPU_RE = re.compile(r"CfgTRES=\S*?cpu=(\d+)")
_ALLOCTRES_CPU_RE = re.compile(r"AllocTRES=\S*?cpu=(\d+)")
_GRES_TYPE_RE = re.compile(r"Gres=gpu:([^:\s(]+)")
_PARTITIONS_RE = re.compile(r"Partitions=(\S+)")
_NODENAME_RE = re.compile(r"NodeName=(\S+)")
_STATE_RE = re.compile(r"State=(\S+)")

_SKIP_PARTITIONS = {"admintest", "pi_co54", "scavenge", "education_gpu", "priority_gpu"}


def _run(cmd: list[str]) -> str:
    """run a command and return stdout or empty string on failure"""

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return result.stdout if result.returncode == 0 else ""


def _parse_all_nodes() -> list[dict]:
    """parse scontrol show node --all into per-node gpu info dicts"""

    out = _run(["scontrol", "show", "node", "--all"])
    nodes: list[dict] = []
    for block in re.split(r"\n\s*\n", out):
        if not block.strip():
            continue
        text = " ".join(block.split())
        name_match = _NODENAME_RE.search(text)
        if not name_match:
            continue
        total_match = _CFGTRES_GPU_RE.search(text)
        used_match = _ALLOCTRES_GPU_RE.search(text)
        cpu_total_match = _CFGTRES_CPU_RE.search(text)
        cpu_alloc_match = _ALLOCTRES_CPU_RE.search(text)
        type_match = _GRES_TYPE_RE.search(text)
        parts_match = _PARTITIONS_RE.search(text)
        state_match = _STATE_RE.search(text)
        total = int(total_match.group(1)) if total_match else 0
        if total == 0:
            continue
        used = int(used_match.group(1)) if used_match else 0
        gpu_type = type_match.group(1) if type_match else "unknown"
        partitions = parts_match.group(1).split(",") if parts_match else []
        state = state_match.group(1) if state_match else ""
        # nodes in RESERVED DRAIN DOWN or MAINT state cannot accept general jobs
        # count their gpus as used so the free counter reflects actually usable
        # gpus PLANNED means slurm is holding the idle slots for a specific
        # pending high priority job - they cannot be backfilled by low priority
        # jobs so they are not really free either ALLOCATED with zero free
        # cpu means the remaining gpus cannot be scheduled because there are
        # no cpu slots left for a new job to claim one
        state_tokens = {tok.upper() for tok in re.split(r"[+,]", state) if tok}
        unavailable = bool(state_tokens & {"RESERVED", "DRAIN", "DRAINING", "DRAINED", "DOWN", "MAINT", "FAIL", "FAILING", "NOT_RESPONDING", "PLANNED"})
        if not unavailable:
            cpu_total = int(cpu_total_match.group(1)) if cpu_total_match else 0
            cpu_used = int(cpu_alloc_match.group(1)) if cpu_alloc_match else 0
            if cpu_total > 0 and cpu_used >= cpu_total and used < total:
                unavailable = True
        if unavailable:
            used = total
        nodes.append(
            {
                "name": name_match.group(1),
                "total": total,
                "used": used,
                "type": gpu_type,
                "partitions": partitions,
                "state": state,
            }
        )
    return nodes


def _aggregate(nodes: list[dict], filter_substr: str | None = None) -> list[dict]:
    """aggregate gpu counts per partition from parsed node records"""

    by_part: dict[str, dict] = {}
    for node in nodes:
        for partition in node["partitions"]:
            if partition in _SKIP_PARTITIONS:
                continue
            if filter_substr and filter_substr.lower() not in partition.lower():
                continue
            row = by_part.setdefault(
                partition,
                {
                    "partition": partition,
                    "total": 0,
                    "used": 0,
                    "types": OrderedDict(),
                    "n_nodes": 0,
                },
            )
            row["total"] += node["total"]
            row["used"] += node["used"]
            row["n_nodes"] += 1
            entry = row["types"].setdefault(node["type"], {"total": 0, "used": 0})
            entry["total"] += node["total"]
            entry["used"] += node["used"]
    for row in by_part.values():
        row["free"] = row["total"] - row["used"]
    return list(by_part.values())


@app.command()
def show(
    filter: Annotated[
        str | None,
        typer.Option("--filter", "-f", help="substring to filter partitions by name"),
    ] = None,
    sort_by: Annotated[
        str,
        typer.Option("--sort", "-s", help="sort key: free partition total used"),
    ] = "free",
) -> None:
    """show free vs total gpus per partition"""

    nodes = _parse_all_nodes()
    rows = _aggregate(nodes, filter_substr=filter)
    if not rows:
        console.print("[yellow]no gpu partitions found[/]")
        raise typer.Exit(0)

    sort_keys = {
        "free": lambda r: -r["free"],
        "partition": lambda r: r["partition"],
        "total": lambda r: -r["total"],
        "used": lambda r: -r["used"],
    }
    rows.sort(key=sort_keys.get(sort_by, sort_keys["free"]))

    table = Table(title="GPU availability by partition", show_header=True, header_style="bold")
    table.add_column("partition", no_wrap=True)
    table.add_column("free", justify="right")
    table.add_column("total", justify="right")
    table.add_column("used", justify="right")
    table.add_column("nodes", justify="right", style="dim")
    table.add_column("gpu types", style="dim")

    for row in rows:
        free = row["free"]
        if free >= 8:
            free_str = f"[bold green]{free}[/]"
        elif free > 0:
            free_str = f"[yellow]{free}[/]"
        else:
            free_str = f"[red]{free}[/]"
        types_str = (
            ", ".join(
                f"{t}:{e['total'] - e['used']}/{e['total']}" for t, e in row["types"].items()
            )
            or "-"
        )
        table.add_row(
            row["partition"],
            free_str,
            str(row["total"]),
            str(row["used"]),
            str(row["n_nodes"]),
            types_str,
        )

    console.print(table)

    total_free = sum(r["free"] for r in rows)
    total_all = sum(r["total"] for r in rows)
    console.print(f"\n[dim]total across listed partitions: {total_free} free / {total_all} gpus[/]")


if __name__ == "__main__":
    app()
