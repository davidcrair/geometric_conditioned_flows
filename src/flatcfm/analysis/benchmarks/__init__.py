"""benchmark evaluation package"""

from .aggregation import aggregate_adata, list_available_aggregations
from .evaluation import Evaluation, build_deg_cache, reduce_group_metrics
from .evaluator import Evaluator
from .metric_space import (
    MetricSpaceSpec,
    MetricSpaceViews,
    metric_space_metadata,
    normalize_metric_space_spec,
    prepare_metric_space_views,
)
from .metrics import (
    compute_cosine_log_fc,
    compute_deg_jaccard,
    compute_deg_overlap,
    compute_energy_distance,
    compute_mean_gene_w1,
    compute_metric,
    compute_mmd,
    compute_mse,
    compute_top_k_recall,
    compute_w2_squared,
    list_available_metrics,
    precompute_deg_info,
    precompute_true_deg_info,
)

__all__ = [
    "Evaluation",
    "build_deg_cache",
    "Evaluator",
    "MetricSpaceSpec",
    "MetricSpaceViews",
    "aggregate_adata",
    "compute_cosine_log_fc",
    "compute_deg_jaccard",
    "compute_deg_overlap",
    "compute_energy_distance",
    "compute_mean_gene_w1",
    "compute_metric",
    "compute_mmd",
    "compute_mse",
    "compute_top_k_recall",
    "compute_w2_squared",
    "list_available_aggregations",
    "list_available_metrics",
    "metric_space_metadata",
    "normalize_metric_space_spec",
    "prepare_metric_space_views",
    "precompute_deg_info",
    "precompute_true_deg_info",
    "reduce_group_metrics",
]
