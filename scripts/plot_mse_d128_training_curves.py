"""plot interactive loss curves for the mse d128 ae run

writes figures/mse_d128_training_curves.html with train and val
total loss and each active component on a shared x axis
"""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "artifacts/runs/sciplex_ae_deg_log1p_mse_d128_dist_k562/20260418_233202_8884380"
OUT = ROOT / "figures/mse_d128_training_curves.html"

history = json.loads((RUN_DIR / "history.json").read_text())

train_total = history.get("train_loss", [])
val_total = history.get("val_loss", [])
comp_train = history.get("individual_train_losses", {})
comp_val = history.get("individual_val_losses", {})

# keep only components with non zero signal
active_components = [
    name for name in sorted(comp_train.keys())
    if any(abs(float(v)) > 1e-12 for v in (comp_train.get(name) or []))
]

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=("total loss", "loss components (active)"),
)

epochs = list(range(1, len(train_total) + 1))

fig.add_trace(
    go.Scatter(x=epochs, y=train_total, mode="lines", name="train total", line=dict(color="#1f77b4")),
    row=1, col=1,
)
fig.add_trace(
    go.Scatter(x=epochs, y=val_total, mode="lines", name="val total", line=dict(color="#d62728")),
    row=1, col=1,
)

palette = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]
for idx, name in enumerate(active_components):
    color = palette[idx % len(palette)]
    tr = comp_train.get(name, [])
    vl = comp_val.get(name, [])
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(tr) + 1)),
            y=tr,
            mode="lines",
            name=f"train {name}",
            line=dict(color=color),
            legendgroup=name,
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(vl) + 1)),
            y=vl,
            mode="lines",
            name=f"val {name}",
            line=dict(color=color, dash="dash"),
            legendgroup=name,
        ),
        row=2, col=1,
    )

fig.update_xaxes(title_text="epoch", row=2, col=1)
fig.update_yaxes(title_text="loss", row=1, col=1)
fig.update_yaxes(title_text="loss", row=2, col=1, type="log")

fig.update_layout(
    height=820,
    width=1100,
    title_text="sciplex_ae_deg_log1p_mse_d128_dist_k562 training curves",
    hovermode="x unified",
    legend=dict(orientation="v"),
)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.write_html(OUT, include_plotlyjs="cdn")
print(f"wrote {OUT}")
