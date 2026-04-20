"""plot training curves for the geometric regularization sweep

for each of the 5 distance weights we plot
  - AE total loss (train vs val)
  - AE recon and distance components
  - FM total loss (train vs val)
on a shared x axis so you can see how the regularization strength
affects both the AE training dynamics and the downstream FM convergence

writes figures/geom_reg_sweep_curves.html
"""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/geom_reg_sweep_curves.html"

WEIGHTS = [
    ("w0", "0.0"),
    ("w0p001", "0.001"),
    ("w0p01", "0.01"),
    ("w0p1", "0.1"),
    ("w1", "1.0"),
]

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def _latest_history(exp_name: str) -> dict | None:
    run_dir = sorted((ROOT / "artifacts/runs" / exp_name).glob("*"))
    if not run_dir:
        return None
    hist = run_dir[-1] / "history.json"
    if not hist.exists():
        return None
    return json.loads(hist.read_text())


fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
        "AE total loss",
        "AE distance component (train)",
        "AE recon (train)",
        "FM total loss",
    ),
    vertical_spacing=0.11,
    horizontal_spacing=0.08,
)

for (tag, label), color in zip(WEIGHTS, PALETTE):
    ae_hist = _latest_history(f"sciplex_ae_deg_geom_reg_{tag}")
    fm_hist = _latest_history(f"sciplex_fm_deg_geom_reg_{tag}")

    if ae_hist is not None:
        epochs = list(range(1, len(ae_hist.get("train_loss", [])) + 1))
        # AE total (train and val on one axis)
        fig.add_trace(
            go.Scatter(
                x=epochs, y=ae_hist.get("train_loss", []), mode="lines",
                name=f"AE w={label} train", line=dict(color=color),
                legendgroup=tag,
            ),
            row=1, col=1,
        )
        val = ae_hist.get("val_loss", [])
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(val) + 1)), y=val, mode="lines",
                name=f"AE w={label} val", line=dict(color=color, dash="dash"),
                legendgroup=tag, showlegend=False,
            ),
            row=1, col=1,
        )

        # distance component (train)
        dist = ae_hist.get("individual_train_losses", {}).get("distance", [])
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(dist) + 1)), y=dist, mode="lines",
                name=f"AE w={label} dist", line=dict(color=color),
                legendgroup=tag, showlegend=False,
            ),
            row=1, col=2,
        )

        # recon component (train)
        recon = ae_hist.get("individual_train_losses", {}).get("recon", [])
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(recon) + 1)), y=recon, mode="lines",
                name=f"AE w={label} recon", line=dict(color=color),
                legendgroup=tag, showlegend=False,
            ),
            row=2, col=1,
        )

    if fm_hist is not None:
        epochs = list(range(1, len(fm_hist.get("train_loss", [])) + 1))
        fig.add_trace(
            go.Scatter(
                x=epochs, y=fm_hist.get("train_loss", []), mode="lines",
                name=f"FM w={label} train", line=dict(color=color),
                legendgroup=tag, showlegend=False,
            ),
            row=2, col=2,
        )
        val = fm_hist.get("val_loss", [])
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(val) + 1)), y=val, mode="lines",
                name=f"FM w={label} val", line=dict(color=color, dash="dash"),
                legendgroup=tag, showlegend=False,
            ),
            row=2, col=2,
        )

fig.update_xaxes(title_text="epoch", row=1, col=1)
fig.update_xaxes(title_text="epoch", row=1, col=2)
fig.update_xaxes(title_text="epoch", row=2, col=1)
fig.update_xaxes(title_text="epoch", row=2, col=2)
fig.update_yaxes(title_text="loss", row=1, col=1, type="log")
fig.update_yaxes(title_text="distance (log)", row=1, col=2, type="log")
fig.update_yaxes(title_text="recon", row=2, col=1)
fig.update_yaxes(title_text="fm mse (log)", row=2, col=2, type="log")

fig.update_layout(
    title_text="geom_reg sweep training curves  dist loss weight 0 to 1",
    height=900,
    width=1400,
    hovermode="x unified",
)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.write_html(OUT, include_plotlyjs="cdn")
print(f"wrote {OUT}")

# also print a quick numerical summary to stdout
print("\nFinal epoch metrics:")
print(f"{'weight':>8} {'AE train':>10} {'AE val':>10} {'AE recon':>10} {'AE dist':>10} {'FM train':>10} {'FM val':>10}")
for tag, label in WEIGHTS:
    ae = _latest_history(f"sciplex_ae_deg_geom_reg_{tag}")
    fm = _latest_history(f"sciplex_fm_deg_geom_reg_{tag}")
    if not ae or not fm:
        continue
    print(
        f"{label:>8} "
        f"{ae['train_loss'][-1]:>10.4f} "
        f"{ae['val_loss'][-1]:>10.4f} "
        f"{ae['individual_train_losses']['recon'][-1]:>10.4f} "
        f"{ae['individual_train_losses']['distance'][-1]:>10.4f} "
        f"{fm['train_loss'][-1]:>10.4f} "
        f"{fm['val_loss'][-1]:>10.4f}"
    )
