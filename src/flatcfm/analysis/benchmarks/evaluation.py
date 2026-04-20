"""benchmark evaluation helpers"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import anndata as ad
import numpy as np
import pandas as pd

from flatcfm._utils import dense_array
from .aggregation import _fit_pca, aggregate_matrix
from .metrics import METRIC_SPECS, compute_metric, precompute_deg_info, precompute_true_deg_info


def _unique_rows(matrix: np.ndarray) -> np.ndarray:
    """deduplicate rows by exact value match

    prevents pseudoreplication when control cells are reused across
    perturbation groups which inflates t-test sample size
    """

    seen: set[bytes] = set()
    mask = np.zeros(matrix.shape[0], dtype=bool)
    for i in range(matrix.shape[0]):
        key = matrix[i].tobytes()
        if key not in seen:
            seen.add(key)
            mask[i] = True
    return matrix[mask]


def _remove_overlapping_rows(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    """remove rows from target that exactly match any row in source

    ensures independence when the source rows are a subset of the target
    eg when no_effect predictions are literally control cells
    """

    if source.shape[0] == 0:
        return target
    source_keys = {row.tobytes() for row in source}
    mask = np.array([row.tobytes() not in source_keys for row in target])
    return target[mask]


def _group_columns(
    perturbation_key: str,
    covariate_keys: Sequence[str],
    group_columns: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """resolve group columns"""

    if group_columns is not None:
        return tuple(str(column) for column in group_columns)
    return tuple([*(str(column) for column in covariate_keys), str(perturbation_key)])


def _filter_obs(adata: ad.AnnData, obs_filters: dict[str, Any] | None) -> ad.AnnData:
    """filter adata by obs values"""

    if not obs_filters:
        return adata
    mask = np.ones(adata.n_obs, dtype=bool)
    for column, value in obs_filters.items():
        if column not in adata.obs.columns:
            raise KeyError(f"Missing filter column: {column}")
        mask &= adata.obs[column].astype(str).to_numpy() == str(value)
    return adata[mask].copy()


def _group_key_tuple(group_key) -> tuple[str, ...]:
    """normalize group key"""

    if isinstance(group_key, tuple):
        return tuple(str(item) for item in group_key)
    return (str(group_key),)


def _key_dict(columns: Sequence[str], key: tuple[str, ...]) -> dict[str, str]:
    """build key dict"""

    return {column: value for column, value in zip(columns, key, strict=False)}


def _shared_control_columns(
    covariate_keys: Sequence[str],
    group_columns: Sequence[str],
) -> tuple[str, ...]:
    """resolve control grouping columns"""

    group_column_set = {str(column) for column in group_columns}
    return tuple(str(key) for key in covariate_keys if str(key) in group_column_set)


@dataclass(frozen=True)
class Evaluation:
    """one benchmark evaluation"""

    prediction_adata: ad.AnnData
    reference_adata: ad.AnnData
    perturbation_key: str
    covariate_keys: tuple[str, ...] = ()
    control_value: str | None = None
    model_name: str = "flatcfm"
    group_columns: tuple[str, ...] | None = None
    control_adata: ad.AnnData | None = None
    reference_filters: dict[str, Any] | None = None

    def _reference_view(self) -> ad.AnnData:
        """build filtered reference"""

        return _filter_obs(self.reference_adata, self.reference_filters)

    def _group_columns(self) -> tuple[str, ...]:
        """resolve group columns"""

        return _group_columns(
            perturbation_key=self.perturbation_key,
            covariate_keys=self.covariate_keys,
            group_columns=self.group_columns,
        )

    def _perturbed_reference(self, reference_view: ad.AnnData) -> ad.AnnData:
        """select perturbed reference rows"""

        if self.control_value is None or self.perturbation_key not in reference_view.obs.columns:
            return reference_view
        mask = reference_view.obs[self.perturbation_key].astype(str).to_numpy() != str(self.control_value)
        return reference_view[mask].copy()

    def _controls(self, reference_view: ad.AnnData) -> ad.AnnData | None:
        """select control rows"""

        if self.control_adata is not None:
            return _filter_obs(self.control_adata, self.reference_filters)
        if self.control_value is None or self.perturbation_key not in reference_view.obs.columns:
            return None
        mask = reference_view.obs[self.perturbation_key].astype(str).to_numpy() == str(self.control_value)
        return reference_view[mask].copy()

    def _matrix_lookup(
        self,
        adata_obj: ad.AnnData,
        group_columns: Sequence[str],
    ) -> dict[tuple[str, ...], np.ndarray]:
        """group adata into matrix lookup

        pre-extracts the dense matrix to avoid per-group anndata slicing overhead
        """

        lookup = {}
        if adata_obj is None or adata_obj.n_obs == 0:
            return lookup
        dense_x = dense_array(adata_obj.X)
        obs_reset = adata_obj.obs.reset_index(drop=True)
        for group_key, frame in obs_reset.groupby(list(group_columns), sort=True, observed=False):
            indices = frame.index.to_numpy()
            if indices.size > 0:
                lookup[_group_key_tuple(group_key)] = dense_x[indices]
        return lookup

    def _control_lookup(self, control_adata: ad.AnnData | None) -> dict[tuple[str, ...], np.ndarray]:
        """group controls by covariates"""

        if control_adata is None or control_adata.n_obs == 0:
            return {}
        control_group_columns = _shared_control_columns(self.covariate_keys, self._group_columns())
        if control_group_columns:
            return self._matrix_lookup(control_adata, control_group_columns)
        return {(): dense_array(control_adata.X)}

    def _control_key(self, group_key: tuple[str, ...], group_columns: Sequence[str]) -> tuple[str, ...]:
        """build control key for one group"""

        control_group_columns = _shared_control_columns(self.covariate_keys, group_columns)
        if not control_group_columns:
            return ()
        values = _key_dict(group_columns, group_key)
        return tuple(values[key] for key in control_group_columns)

    def reference_logfc_norms(self) -> pd.DataFrame:
        """compute reference logfc norms per group"""

        group_cols = self._group_columns()
        reference_view = self._reference_view()
        perturbed_reference = self._perturbed_reference(reference_view)
        control_adata = self._controls(reference_view)
        ref_lookup = self._matrix_lookup(perturbed_reference, group_cols)
        control_lookup = self._control_lookup(control_adata)
        rows = []
        for group_key, ref_matrix in ref_lookup.items():
            ctrl_matrix = control_lookup.get(self._control_key(group_key, group_cols))
            if ctrl_matrix is None or ctrl_matrix.shape[0] == 0:
                continue
            ref_logfc = ref_matrix.mean(axis=0, dtype=np.float64) - ctrl_matrix.mean(axis=0, dtype=np.float64)
            row = _key_dict(group_cols, group_key)
            row["n_ref"] = int(ref_matrix.shape[0])
            row["n_ctrl"] = int(ctrl_matrix.shape[0])
            row["ref_logfc_norm"] = float(np.linalg.norm(ref_logfc))
            rows.append(row)
        return pd.DataFrame(rows)

    def evaluate(
        self,
        metrics: Sequence[str] | None = None,
        aggregation: str | None = None,
        pca_n_components: int = 50,
        top_ks: Sequence[int | None] | None = None,
        max_samples: int = 2000,
        fdr_alpha: float = 0.05,
        min_cells: int = 5,
        deg_cache: dict[tuple[str, ...], dict[str, Any]] | None = None,
    ) -> pd.DataFrame:
        """evaluate predictions against reference

        Args:
            deg_cache: optional pre-computed deg cache keyed by group_key
                each entry has "true_deg_info" and "unique_ctrl" to avoid
                redundant computation when evaluating multiple models
        """

        metric_names = list(metrics or ["mean_gene_w1", "w2_squared"])
        group_columns = self._group_columns()
        reference_view = self._reference_view()
        perturbed_reference = self._perturbed_reference(reference_view)
        control_adata = self._controls(reference_view)

        pred_lookup = self._matrix_lookup(self.prediction_adata, group_columns)
        ref_lookup = self._matrix_lookup(perturbed_reference, group_columns)
        control_lookup = self._control_lookup(control_adata)

        needs_pca = any(
            (aggregation or METRIC_SPECS[name].default_aggregation) in {"pca", "pca_average"} for name in metric_names
        )
        pca_model = None
        if needs_pca:
            pca_model = _fit_pca(dense_array(reference_view.X), n_components=int(pca_n_components))

        # precompute control means so logfc/scaled/average aggregation avoids
        # redundantly computing mean over large shared control matrices
        ctrl_mean_lookup: dict[tuple[str, ...], np.ndarray] = {}
        for ctrl_key, ctrl_mat in control_lookup.items():
            ctrl_mean_lookup[ctrl_key] = ctrl_mat.mean(axis=0, dtype=np.float64, keepdims=True).astype(
                np.float32, copy=False,
            )

        _deg_metrics = {"top_k_recall", "deg_jaccard", "deg_overlap_at_k", "sig_deg_recall", "sig_deg_intersect_count"}
        needs_deg = bool(_deg_metrics & set(metric_names))

        rows = []
        for group_key in sorted(set(pred_lookup) & set(ref_lookup)):
            pred_matrix = pred_lookup[group_key]
            ref_matrix = ref_lookup[group_key]
            ctrl_key = self._control_key(group_key, group_columns)
            ctrl_matrix = control_lookup.get(ctrl_key)
            # 1-row mean matrix for aggregations that only need the control mean
            ctrl_mean_matrix = ctrl_mean_lookup.get(ctrl_key)
            base_row = {
                "model_name": str(self.model_name),
                **_key_dict(group_columns, group_key),
                "n_pred": int(pred_matrix.shape[0]),
                "n_ref": int(ref_matrix.shape[0]),
                "n_ctrl": 0 if ctrl_matrix is None else int(ctrl_matrix.shape[0]),
            }

            true_deg_info = None
            pred_deg_info = None
            if needs_deg and ctrl_matrix is not None:
                cached = deg_cache.get(group_key) if deg_cache is not None else None
                if cached is not None:
                    true_deg_info = cached["true_deg_info"]
                    unique_ctrl = cached["unique_ctrl"]
                else:
                    true_deg_info = precompute_true_deg_info(
                        ref_matrix, ctrl_matrix, fdr_alpha=float(fdr_alpha), min_cells=int(min_cells),
                    )
                    unique_ctrl = _unique_rows(ctrl_matrix)
                ctrl_for_pred_deg = _remove_overlapping_rows(unique_ctrl, pred_matrix)
                pred_deg_info = precompute_deg_info(
                    pred_matrix, ctrl_for_pred_deg, fdr_alpha=float(fdr_alpha), min_cells=int(min_cells),
                )

            for metric_name in metric_names:
                spec = METRIC_SPECS[metric_name]
                metric_aggregation = str(aggregation or spec.default_aggregation)
                # logfc only needs the control mean, not the full distribution
                ctrl_for_agg = ctrl_mean_matrix if metric_aggregation == "logfc" and ctrl_mean_matrix is not None else ctrl_matrix
                pred_input = aggregate_matrix(
                    pred_matrix,
                    aggregation=metric_aggregation,
                    control_matrix=ctrl_for_agg,
                    pca_model=pca_model,
                )
                ref_input = aggregate_matrix(
                    ref_matrix,
                    aggregation=metric_aggregation,
                    control_matrix=ctrl_for_agg,
                    pca_model=pca_model,
                )
                if metric_aggregation in {"scaled", "logfc"} and ctrl_matrix is not None:
                    ctrl_input = ctrl_mean_matrix
                elif metric_aggregation in {"pca", "pca_average"} and ctrl_matrix is not None:
                    ctrl_input = aggregate_matrix(
                        ctrl_matrix,
                        aggregation=metric_aggregation,
                        control_matrix=ctrl_matrix,
                        pca_model=pca_model,
                    )
                else:
                    ctrl_input = ctrl_matrix
                values = compute_metric(
                    metric_name,
                    pred=pred_input,
                    ref=ref_input,
                    ctrl=ctrl_input,
                    top_ks=tuple(top_ks or (50, 100, 200)),
                    max_samples=int(max_samples),
                    fdr_alpha=float(fdr_alpha),
                    min_cells=int(min_cells),
                    true_deg_info=true_deg_info,
                    pred_deg_info=pred_deg_info,
                )
                for output_name, value in values.items():
                    rows.append(
                        {
                            **base_row,
                            "aggregation": metric_aggregation,
                            "metric": output_name,
                            "metric_base": metric_name,
                            "category": spec.category,
                            "value": float(value),
                        }
                    )
        return pd.DataFrame(rows)


def build_deg_cache(
    evaluation: Evaluation,
    fdr_alpha: float = 0.05,
    min_cells: int = 5,
) -> dict[tuple[str, ...], dict[str, Any]]:
    """precompute true deg info and deduplicated control per group

    computing these once and passing via deg_cache to evaluate() avoids
    redundant t-tests and row deduplication when evaluating multiple models
    against the same reference and control data
    """

    group_columns = evaluation._group_columns()
    reference_view = evaluation._reference_view()
    perturbed_reference = evaluation._perturbed_reference(reference_view)
    control_adata = evaluation._controls(reference_view)

    ref_lookup = evaluation._matrix_lookup(perturbed_reference, group_columns)
    control_lookup = evaluation._control_lookup(control_adata)

    cache: dict[tuple[str, ...], dict[str, Any]] = {}
    for group_key in sorted(ref_lookup):
        ref_matrix = ref_lookup[group_key]
        ctrl_key = evaluation._control_key(group_key, group_columns)
        ctrl_matrix = control_lookup.get(ctrl_key)
        if ctrl_matrix is None:
            continue
        true_deg_info = precompute_true_deg_info(
            ref_matrix, ctrl_matrix, fdr_alpha=float(fdr_alpha), min_cells=int(min_cells),
        )
        unique_ctrl = _unique_rows(ctrl_matrix)
        cache[group_key] = {"true_deg_info": true_deg_info, "unique_ctrl": unique_ctrl}
    return cache


def reduce_group_metrics(
    metrics_df: pd.DataFrame,
    reductions: Sequence[str] = ("unweighted_mean", "cell_weighted_mean"),
    weight_column: str = "n_ref",
) -> pd.DataFrame:
    """reduce grouped metric rows to model level summaries"""

    if metrics_df.empty:
        return pd.DataFrame(
            columns=[
                "model_name",
                "metric",
                "metric_base",
                "category",
                "aggregation",
                "reduction",
                "value",
                "n_groups",
            ]
        )

    allowed = {"unweighted_mean", "cell_weighted_mean"}
    invalid = [item for item in reductions if str(item) not in allowed]
    if invalid:
        raise ValueError(f"Unsupported reductions: {invalid}")

    metadata_columns = [
        column
        for column in [
            "model_name",
            "metric",
            "metric_base",
            "category",
            "aggregation",
            "metric_space",
            "metric_space_label",
            "prediction_name",
        ]
        if column in metrics_df.columns
    ]
    rows = []
    for _, frame in metrics_df.groupby(metadata_columns, sort=True, dropna=False):
        base = {column: frame.iloc[0][column] for column in metadata_columns}
        values = frame["value"].astype(float).to_numpy()
        for reduction in reductions:
            if reduction == "unweighted_mean":
                reduced_value = float(np.nanmean(values)) if values.size else float("nan")
            else:
                if weight_column not in frame.columns:
                    raise KeyError(f"Missing reduction weight column: {weight_column}")
                weights = frame[weight_column].astype(float).to_numpy()
                valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
                if not np.any(valid):
                    reduced_value = float("nan")
                else:
                    reduced_value = float(np.average(values[valid], weights=weights[valid]))
            rows.append(
                {
                    **base,
                    "reduction": str(reduction),
                    "value": reduced_value,
                    "n_groups": int(frame.shape[0]),
                }
            )
    return pd.DataFrame(rows)
