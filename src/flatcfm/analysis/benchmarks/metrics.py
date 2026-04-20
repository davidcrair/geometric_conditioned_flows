"""benchmark metric helpers"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import ot
import pandas as pd
from scipy import stats
from scipy.stats import false_discovery_control, wasserstein_distance
import torch


@dataclass(frozen=True)
class MetricSpec:
    """metric registry entry"""

    name: str
    category: str
    default_aggregation: str
    requires_control: bool = False


def _to_numpy(x: Any) -> np.ndarray:
    """convert input to numpy"""

    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False)
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(x, dtype=np.float32)


def _to_torch(x: Any) -> torch.Tensor:
    """convert input to torch"""

    if torch.is_tensor(x):
        return x.detach()
    return torch.as_tensor(_to_numpy(x), dtype=torch.float32)


def _to_vector(x: Any) -> np.ndarray:
    """convert input to vector"""

    vector = _to_numpy(x).reshape(-1)
    return vector.astype(np.float64, copy=False)


def _sample_torch_pair(x: torch.Tensor, y: torch.Tensor, max_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """subsample torch matrices"""

    if x.shape[0] > max_samples:
        idx = torch.randperm(x.shape[0], device=x.device)[:max_samples]
        x = x.index_select(0, idx)
    if y.shape[0] > max_samples:
        idx = torch.randperm(y.shape[0], device=y.device)[:max_samples]
        y = y.index_select(0, idx)
    return x, y


def compute_energy_distance(x: Any, y: Any, max_samples: int = 2000) -> float:
    """compute energy distance"""

    x_tensor, y_tensor = _sample_torch_pair(_to_torch(x), _to_torch(y), max_samples=max_samples)
    dist_xy = torch.cdist(x_tensor, y_tensor, p=2)
    dist_xx = torch.cdist(x_tensor, x_tensor, p=2)
    dist_yy = torch.cdist(y_tensor, y_tensor, p=2)
    return float((2 * dist_xy.mean() - dist_xx.mean() - dist_yy.mean()).item())


def compute_mmd(x: Any, y: Any, bandwidth: float | None = None, max_samples: int = 2000) -> float:
    """compute mmd"""

    x_tensor, y_tensor = _sample_torch_pair(_to_torch(x), _to_torch(y), max_samples=max_samples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_tensor = x_tensor.to(device)
    y_tensor = y_tensor.to(device)
    if bandwidth is None:
        bandwidth = float(torch.cdist(x_tensor, y_tensor, p=2).reshape(-1).median().item())
    bandwidth = max(float(bandwidth), 1e-8)
    gamma = 1.0 / (2 * bandwidth**2)
    k_xx = torch.exp(-gamma * torch.cdist(x_tensor, x_tensor, p=2).pow(2)).mean()
    k_yy = torch.exp(-gamma * torch.cdist(y_tensor, y_tensor, p=2).pow(2)).mean()
    k_xy = torch.exp(-gamma * torch.cdist(x_tensor, y_tensor, p=2).pow(2)).mean()
    return float((k_xx + k_yy - 2 * k_xy).item())


def compute_mean_gene_w1(x: Any, y: Any) -> float:
    """compute mean per gene w1

    when both inputs have the same number of rows the 1d wasserstein distance
    simplifies to mean(|sort(x) - sort(y)|) per gene which is fully vectorizable
    """

    x_matrix = _to_numpy(x)
    y_matrix = _to_numpy(y)
    if x_matrix.ndim != 2 or y_matrix.ndim != 2:
        raise ValueError("mean_gene_w1 expects 2d matrices")
    if x_matrix.shape[1] != y_matrix.shape[1]:
        raise ValueError("mean_gene_w1 expects matching feature counts")
    if x_matrix.shape[0] == y_matrix.shape[0]:
        # fast vectorized path for equal sample sizes
        x_sorted = np.sort(x_matrix, axis=0)
        y_sorted = np.sort(y_matrix, axis=0)
        per_gene_w1 = np.mean(np.abs(x_sorted - y_sorted), axis=0, dtype=np.float64)
        return float(np.mean(per_gene_w1))
    values = [
        wasserstein_distance(x_matrix[:, idx], y_matrix[:, idx])
        for idx in range(x_matrix.shape[1])
    ]
    return float(np.mean(values, dtype=np.float64))


def compute_w2_squared(x: Any, y: Any, max_samples: int | None = None) -> float:
    """compute w2 squared"""

    x_matrix = _to_numpy(x).astype(np.float64, copy=False)
    y_matrix = _to_numpy(y).astype(np.float64, copy=False)
    if x_matrix.ndim != 2 or y_matrix.ndim != 2:
        raise ValueError("w2_squared expects 2d matrices")
    if max_samples is not None and x_matrix.shape[0] > max_samples:
        idx = np.random.default_rng(0).choice(x_matrix.shape[0], size=max_samples, replace=False)
        x_matrix = x_matrix[idx]
    if max_samples is not None and y_matrix.shape[0] > max_samples:
        idx = np.random.default_rng(0).choice(y_matrix.shape[0], size=max_samples, replace=False)
        y_matrix = y_matrix[idx]
    if x_matrix.shape[0] == 0 or y_matrix.shape[0] == 0:
        return float("nan")
    a = np.full(x_matrix.shape[0], 1.0 / x_matrix.shape[0], dtype=np.float64)
    b = np.full(y_matrix.shape[0], 1.0 / y_matrix.shape[0], dtype=np.float64)
    cost = ot.dist(x_matrix, y_matrix, metric="sqeuclidean")
    return float(ot.emd2(a, b, cost))


def compute_mse(x: Any, y: Any) -> float:
    """compute mse"""

    x_vector = _to_vector(x)
    y_vector = _to_vector(y)
    return float(np.mean((x_vector - y_vector) ** 2, dtype=np.float64))


def compute_rmse(x: Any, y: Any) -> float:
    """compute rmse"""

    return float(np.sqrt(compute_mse(x, y)))


def compute_mae(x: Any, y: Any) -> float:
    """compute mae"""

    x_vector = _to_vector(x)
    y_vector = _to_vector(y)
    return float(np.mean(np.abs(x_vector - y_vector), dtype=np.float64))


def compute_r2_score(x: Any, y: Any) -> float:
    """compute r2 score"""

    x_vector = _to_vector(x)
    y_vector = _to_vector(y)
    denom = np.sum((y_vector - y_vector.mean()) ** 2, dtype=np.float64)
    if denom < 1e-12:
        return float("nan")
    numer = np.sum((y_vector - x_vector) ** 2, dtype=np.float64)
    return float(1.0 - numer / denom)


def compute_cosine(x: Any, y: Any, eps: float = 1e-8) -> float:
    """compute cosine similarity"""

    x_vector = _to_vector(x)
    y_vector = _to_vector(y)
    denom = max(np.linalg.norm(x_vector) * np.linalg.norm(y_vector), eps)
    return float(np.dot(x_vector, y_vector) / denom)


def compute_pearson(x: Any, y: Any) -> float:
    """compute pearson correlation"""

    x_vector = _to_vector(x)
    y_vector = _to_vector(y)
    if np.std(x_vector) < 1e-12 or np.std(y_vector) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x_vector, y_vector)[0, 1])


def compute_cosine_log_fc(x: Any, y: Any, eps: float = 1e-8) -> float:
    """compute cosine similarity on logfc vectors"""

    x_vec = np.asarray(x, dtype=np.float64).ravel()
    y_vec = np.asarray(y, dtype=np.float64).ravel()
    denom = max(float(np.linalg.norm(x_vec) * np.linalg.norm(y_vec)), eps)
    return float(np.dot(x_vec, y_vec) / denom)


def precompute_deg_info(
    x_true: np.ndarray,
    x_ctrl: np.ndarray,
    fdr_alpha: float = 0.05,
    min_cells: int = 5,
) -> dict[str, Any]:
    """precompute deg info"""

    true_matrix = _to_numpy(x_true)
    ctrl_matrix = _to_numpy(x_ctrl)
    if true_matrix.shape[0] < int(min_cells) or ctrl_matrix.shape[0] < int(min_cells):
        return {"ranked_indices": np.array([], dtype=int), "n_sig": 0, "sig_indices": np.array([], dtype=int)}

    _, pvals = stats.ttest_ind(true_matrix, ctrl_matrix, axis=0, equal_var=False)
    pvals = np.nan_to_num(pvals, nan=1.0)
    adj_pvals = false_discovery_control(pvals, method="bh")
    sig_mask = adj_pvals < float(fdr_alpha)
    lfc = true_matrix.mean(axis=0, dtype=np.float64) - ctrl_matrix.mean(axis=0, dtype=np.float64)
    sig_indices = np.where(sig_mask)[0]
    if sig_indices.size == 0:
        return {"ranked_indices": np.array([], dtype=int), "n_sig": 0, "sig_indices": sig_indices}
    order = np.argsort(-np.abs(lfc[sig_indices]), kind="stable")
    ranked_indices = sig_indices[order]
    return {"ranked_indices": ranked_indices, "n_sig": int(ranked_indices.size), "sig_indices": sig_indices}


def precompute_true_deg_info(
    x_true: np.ndarray,
    x_ctrl: np.ndarray,
    fdr_alpha: float = 0.05,
    min_cells: int = 5,
) -> dict[str, Any]:
    """precompute true deg info"""

    return precompute_deg_info(
        x_true=x_true,
        x_ctrl=x_ctrl,
        fdr_alpha=fdr_alpha,
        min_cells=min_cells,
    )


def _deg_sets(
    x_pred: np.ndarray,
    x_ctrl: np.ndarray,
    true_deg_info: dict[str, Any],
    pred_deg_info: dict[str, Any] | None,
    fdr_alpha: float,
    min_cells: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """build deg ranking sets"""

    if pred_deg_info is None:
        pred_deg_info = precompute_deg_info(
            x_true=x_pred,
            x_ctrl=x_ctrl,
            fdr_alpha=fdr_alpha,
            min_cells=min_cells,
        )
    true_ranked = np.asarray(true_deg_info["ranked_indices"], dtype=int)
    pred_ranked = np.asarray(pred_deg_info["ranked_indices"], dtype=int)
    n_sig = int(true_deg_info["n_sig"])
    return true_ranked, pred_ranked, n_sig


def compute_deg_overlap(
    x_pred: np.ndarray,
    x_ctrl: np.ndarray,
    true_deg_info: dict[str, Any],
    ks: list[int | None] | None = None,
    pred_deg_info: dict[str, Any] | None = None,
    fdr_alpha: float = 0.05,
    min_cells: int = 5,
) -> dict[str, float]:
    """compute deg overlap"""

    if ks is None:
        ks = [50, 100, 200, None]
    true_ranked, pred_ranked, n_sig = _deg_sets(
        x_pred=x_pred,
        x_ctrl=x_ctrl,
        true_deg_info=true_deg_info,
        pred_deg_info=pred_deg_info,
        fdr_alpha=fdr_alpha,
        min_cells=min_cells,
    )
    overlaps = {}
    for k in ks:
        actual_k = n_sig if k is None else int(k)
        label = "DEG@N" if k is None else f"DEG@{k}"
        if actual_k == 0 or n_sig == 0:
            overlaps[label] = 0.0
            continue
        true_top = set(true_ranked[:actual_k])
        pred_top = set(pred_ranked[:actual_k])
        overlaps[label] = float(len(true_top & pred_top) / actual_k)
    return overlaps


def compute_top_k_recall(
    x_pred: np.ndarray,
    x_ctrl: np.ndarray,
    true_deg_info: dict[str, Any],
    ks: tuple[int, ...] = (50, 100, 200),
    pred_deg_info: dict[str, Any] | None = None,
    fdr_alpha: float = 0.05,
    min_cells: int = 5,
) -> dict[str, float]:
    """compute top k recall"""

    overlaps = compute_deg_overlap(
        x_pred=x_pred,
        x_ctrl=x_ctrl,
        true_deg_info=true_deg_info,
        ks=list(ks),
        pred_deg_info=pred_deg_info,
        fdr_alpha=fdr_alpha,
        min_cells=min_cells,
    )
    return {f"top_k_recall@{key.split('@')[-1]}": value for key, value in overlaps.items()}


def compute_sig_deg_intersect_count(
    x_pred: np.ndarray,
    x_ctrl: np.ndarray,
    true_deg_info: dict[str, Any],
    pred_deg_info: dict[str, Any] | None = None,
    fdr_alpha: float = 0.05,
    min_cells: int = 5,
) -> dict[str, float]:
    """absolute count of genes that are fdr significant in both ref and pred"""

    if pred_deg_info is None:
        pred_deg_info = precompute_deg_info(
            x_true=x_pred,
            x_ctrl=x_ctrl,
            fdr_alpha=fdr_alpha,
            min_cells=min_cells,
        )
    true_sig = set(np.asarray(true_deg_info["sig_indices"], dtype=int).tolist())
    pred_sig = set(np.asarray(pred_deg_info["sig_indices"], dtype=int).tolist())
    label = f"sig_deg_intersect_count@fdr{fdr_alpha:g}"
    return {label: float(len(true_sig & pred_sig))}


def compute_sig_deg_recall(
    x_pred: np.ndarray,
    x_ctrl: np.ndarray,
    true_deg_info: dict[str, Any],
    pred_deg_info: dict[str, Any] | None = None,
    fdr_alpha: float = 0.05,
    min_cells: int = 5,
) -> dict[str, float]:
    """recall of fdr significant reference degs that are also significant in pred

    returns nan when the reference has no fdr significant genes
    """

    if pred_deg_info is None:
        pred_deg_info = precompute_deg_info(
            x_true=x_pred,
            x_ctrl=x_ctrl,
            fdr_alpha=fdr_alpha,
            min_cells=min_cells,
        )
    true_sig = set(np.asarray(true_deg_info["sig_indices"], dtype=int).tolist())
    pred_sig = set(np.asarray(pred_deg_info["sig_indices"], dtype=int).tolist())
    label = f"sig_deg_recall@fdr{fdr_alpha:g}"
    if not true_sig:
        return {label: float("nan")}
    return {label: float(len(true_sig & pred_sig) / len(true_sig))}


def compute_deg_jaccard(
    x_pred: np.ndarray,
    x_ctrl: np.ndarray,
    true_deg_info: dict[str, Any],
    ks: tuple[int, ...] = (50, 100, 200),
    pred_deg_info: dict[str, Any] | None = None,
    fdr_alpha: float = 0.05,
    min_cells: int = 5,
) -> dict[str, float]:
    """compute deg jaccard"""

    true_ranked, pred_ranked, n_sig = _deg_sets(
        x_pred=x_pred,
        x_ctrl=x_ctrl,
        true_deg_info=true_deg_info,
        pred_deg_info=pred_deg_info,
        fdr_alpha=fdr_alpha,
        min_cells=min_cells,
    )
    results = {}
    for k in ks:
        actual_k = min(int(k), max(n_sig, 0))
        label = f"deg_jaccard@{k}"
        if actual_k == 0:
            results[label] = 0.0
            continue
        true_top = set(true_ranked[:actual_k])
        pred_top = set(pred_ranked[:actual_k])
        union = true_top | pred_top
        results[label] = 0.0 if not union else float(len(true_top & pred_top) / len(union))
    return results


METRIC_SPECS = {
    "mean_gene_w1": MetricSpec("mean_gene_w1", "cell_distribution", "none"),
    "w2_squared": MetricSpec("w2_squared", "cell_distribution", "none"),
    "energy_distance": MetricSpec("energy_distance", "cell_distribution", "none"),
    "mmd": MetricSpec("mmd", "cell_distribution", "none"),
    "pearson": MetricSpec("pearson", "profile", "average"),
    "cosine": MetricSpec("cosine", "profile", "average"),
    "mse": MetricSpec("mse", "profile", "average"),
    "rmse": MetricSpec("rmse", "profile", "average"),
    "mae": MetricSpec("mae", "profile", "average"),
    "r2_score": MetricSpec("r2_score", "profile", "average"),
    "cosine_log_fc": MetricSpec("cosine_log_fc", "profile", "logfc", requires_control=True),
    "top_k_recall": MetricSpec("top_k_recall", "de", "none", requires_control=True),
    "deg_jaccard": MetricSpec("deg_jaccard", "de", "none", requires_control=True),
    "deg_overlap_at_k": MetricSpec("deg_overlap_at_k", "de", "none", requires_control=True),
    "sig_deg_recall": MetricSpec("sig_deg_recall", "de", "none", requires_control=True),
    "sig_deg_intersect_count": MetricSpec("sig_deg_intersect_count", "de", "none", requires_control=True),
}


def list_available_metrics() -> pd.DataFrame:
    """list available metrics"""

    return pd.DataFrame(
        [
            {
                "metric": spec.name,
                "category": spec.category,
                "default_aggregation": spec.default_aggregation,
                "requires_control": spec.requires_control,
            }
            for spec in METRIC_SPECS.values()
        ]
    ).sort_values("metric").reset_index(drop=True)


def compute_metric(
    metric_name: str,
    pred: Any,
    ref: Any,
    ctrl: Any | None = None,
    **kwargs,
) -> dict[str, float]:
    """compute one metric"""

    if metric_name not in METRIC_SPECS:
        raise KeyError(f"Unknown metric: {metric_name}")
    spec = METRIC_SPECS[metric_name]
    if spec.requires_control and ctrl is None:
        return {metric_name: float("nan")}
    if metric_name == "mean_gene_w1":
        return {metric_name: compute_mean_gene_w1(pred, ref)}
    if metric_name == "w2_squared":
        return {metric_name: compute_w2_squared(pred, ref, max_samples=kwargs.get("max_samples"))}
    if metric_name == "energy_distance":
        return {metric_name: compute_energy_distance(pred, ref, max_samples=int(kwargs.get("max_samples", 2000)))}
    if metric_name == "mmd":
        return {
            metric_name: compute_mmd(
                pred,
                ref,
                bandwidth=kwargs.get("bandwidth"),
                max_samples=int(kwargs.get("max_samples", 2000)),
            )
        }
    if metric_name == "pearson":
        return {metric_name: compute_pearson(pred, ref)}
    if metric_name == "cosine":
        return {metric_name: compute_cosine(pred, ref)}
    if metric_name == "mse":
        return {metric_name: compute_mse(pred, ref)}
    if metric_name == "rmse":
        return {metric_name: compute_rmse(pred, ref)}
    if metric_name == "mae":
        return {metric_name: compute_mae(pred, ref)}
    if metric_name == "r2_score":
        return {metric_name: compute_r2_score(pred, ref)}
    if metric_name == "cosine_log_fc":
        return {metric_name: compute_cosine_log_fc(pred, ref)}
    if metric_name in {"top_k_recall", "deg_jaccard", "deg_overlap_at_k", "sig_deg_recall", "sig_deg_intersect_count"}:
        fdr = float(kwargs.get("fdr_alpha", 0.05))
        mc = int(kwargs.get("min_cells", 5))
        true_deg_info = kwargs.get("true_deg_info") or precompute_true_deg_info(ref, ctrl, fdr_alpha=fdr, min_cells=mc)
        pred_deg_info = kwargs.get("pred_deg_info")
        deg_kwargs = dict(x_pred=pred, x_ctrl=ctrl, true_deg_info=true_deg_info, fdr_alpha=fdr, min_cells=mc, pred_deg_info=pred_deg_info)
        if metric_name == "top_k_recall":
            return compute_top_k_recall(**deg_kwargs, ks=tuple(kwargs.get("top_ks", (50, 100, 200))))
        if metric_name == "deg_jaccard":
            return compute_deg_jaccard(**deg_kwargs, ks=tuple(kwargs.get("top_ks", (50, 100, 200))))
        if metric_name == "sig_deg_recall":
            return compute_sig_deg_recall(**deg_kwargs)
        if metric_name == "sig_deg_intersect_count":
            return compute_sig_deg_intersect_count(**deg_kwargs)
        overlaps = compute_deg_overlap(**deg_kwargs, ks=list(kwargs.get("top_ks", (50, 100, 200, None))))
        return {f"deg_overlap_at_k:{key}": value for key, value in overlaps.items()}
    raise KeyError(f"Unknown metric: {metric_name}")
