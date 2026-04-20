import unittest

import anndata as ad
import numpy as np
import pandas as pd

from flatcfm.analysis.benchmarks import (
    Evaluation,
    Evaluator,
    aggregate_adata,
    compute_metric,
    list_available_aggregations,
    list_available_metrics,
    reduce_group_metrics,
)
from flatcfm.analysis.perturbench_adapter import PerturbBenchTaskMetadata, to_perturbench_predictions


def _make_reference_adata() -> ad.AnnData:
    x_ctrl = np.array(
        [
            [0.00, 0.00, 0.00, 0.00],
            [0.10, -0.10, 0.05, 0.00],
            [-0.10, 0.10, -0.05, 0.02],
            [0.05, 0.00, 0.02, -0.02],
            [-0.05, 0.05, 0.01, 0.00],
            [0.00, -0.05, -0.02, 0.01],
        ],
        dtype=np.float32,
    )
    x_druga = np.array(
        [
            [4.00, 3.00, 0.00, 0.00],
            [4.10, 2.90, 0.05, 0.00],
            [3.90, 3.10, -0.05, 0.02],
            [4.05, 3.00, 0.02, -0.02],
            [3.95, 3.05, 0.01, 0.00],
            [4.00, 2.95, -0.02, 0.01],
        ],
        dtype=np.float32,
    )
    x_drugb = np.array(
        [
            [0.00, 0.00, 2.00, 2.00],
            [0.10, -0.10, 2.10, 1.90],
            [-0.10, 0.10, 1.90, 2.10],
            [0.05, 0.00, 2.05, 1.95],
            [-0.05, 0.05, 1.95, 2.05],
            [0.00, -0.05, 2.02, 2.01],
        ],
        dtype=np.float32,
    )
    x = np.vstack([x_ctrl, x_druga, x_drugb])
    obs = pd.DataFrame(
        {
            "perturbation": ["ctrl"] * 6 + ["drugA"] * 6 + ["drugB"] * 6,
            "cell_type": ["K562"] * 18,
            "replicate": ["r1", "r1", "r1", "r2", "r2", "r2"] * 3,
            "product_dose": ["ctrl"] * 6 + ["drugA_10"] * 6 + ["drugB_10"] * 6,
        }
    )
    adata = ad.AnnData(X=x, obs=obs)
    adata.var_names = ["gene1", "gene2", "gene3", "gene4"]
    return adata


def _make_prediction_adata(scale: float = 1.0) -> ad.AnnData:
    ref = _make_reference_adata()
    mask = ref.obs["perturbation"].isin(["drugA", "drugB"]).to_numpy()
    pred = ref[mask].copy()
    pred.X = np.asarray(pred.X, dtype=np.float32) * scale
    return pred


class BenchmarkAggregationTest(unittest.TestCase):
    def test_list_available_aggregations(self):
        self.assertEqual(
            list_available_aggregations(),
            ("none", "average", "scaled", "logfc", "var", "pca", "pca_average"),
        )

    def test_aggregate_adata_returns_grouped_representations(self):
        ref = _make_reference_adata()
        pert = ref[ref.obs["perturbation"] != "ctrl"].copy()
        ctrl = ref[ref.obs["perturbation"] == "ctrl"].copy()
        out = aggregate_adata(
            pert,
            group_columns=("perturbation",),
            aggregation="logfc",
            control_adata=ctrl,
            control_group_columns=(),
        )
        self.assertEqual(set(out["perturbation"]), {"drugA", "drugB"})
        self.assertEqual(out.iloc[0]["representation"].shape, (4,))


class BenchmarkMetricRegistryTest(unittest.TestCase):
    def test_list_available_metrics_contains_expected_entries(self):
        metrics = list_available_metrics()
        self.assertIn("mean_gene_w1", set(metrics["metric"]))
        self.assertIn("top_k_recall", set(metrics["metric"]))

    def test_compute_metric_dispatch(self):
        pred = np.array([[1.0, 2.0], [1.0, 2.0]], dtype=np.float32)
        ref = np.array([[1.0, 2.0], [1.0, 2.0]], dtype=np.float32)
        out = compute_metric("w2_squared", pred, ref)
        self.assertEqual(out["w2_squared"], 0.0)


class BenchmarkEvaluationTest(unittest.TestCase):
    def test_evaluation_returns_tidy_metrics(self):
        ref = _make_reference_adata()
        pred = _make_prediction_adata(scale=0.95)
        evaluation = Evaluation(
            prediction_adata=pred,
            reference_adata=ref,
            perturbation_key="perturbation",
            covariate_keys=("cell_type",),
            control_value="ctrl",
        )
        out = evaluation.evaluate(
            metrics=("mean_gene_w1", "w2_squared", "mse", "cosine_log_fc"),
            top_ks=(1, 2),
        )
        self.assertFalse(out.empty)
        self.assertIn("metric", out.columns)
        self.assertIn("value", out.columns)
        self.assertIn("drugA", set(out["perturbation"]))
        self.assertTrue({"mean_gene_w1", "w2_squared", "mse", "cosine_log_fc"}.issubset(set(out["metric_base"])))

    def test_de_metrics_are_supported(self):
        ref = _make_reference_adata()
        pred = _make_prediction_adata(scale=1.0)
        evaluation = Evaluation(
            prediction_adata=pred,
            reference_adata=ref,
            perturbation_key="perturbation",
            covariate_keys=("cell_type",),
            control_value="ctrl",
        )
        out = evaluation.evaluate(
            metrics=("top_k_recall", "deg_jaccard", "deg_overlap_at_k"),
            top_ks=(1, 2),
        )
        self.assertFalse(out.empty)
        self.assertTrue(any(metric.startswith("top_k_recall@") for metric in out["metric"]))
        self.assertTrue(any(metric.startswith("deg_jaccard@") for metric in out["metric"]))
        self.assertTrue(any(metric.startswith("deg_overlap_at_k:DEG@") for metric in out["metric"]))

    def test_evaluation_pools_controls_when_grouping_omits_covariates(self):
        ref = _make_reference_adata()
        pred = _make_prediction_adata(scale=1.0)
        evaluation = Evaluation(
            prediction_adata=pred,
            reference_adata=ref,
            perturbation_key="perturbation",
            covariate_keys=("cell_type",),
            control_value="ctrl",
            group_columns=("perturbation",),
        )
        out = evaluation.evaluate(metrics=("w2_squared",))
        self.assertFalse(out.empty)
        self.assertEqual(set(out["perturbation"]), {"drugA", "drugB"})
        self.assertTrue((out["n_ctrl"] == 6).all())

    def test_reduce_group_metrics_returns_unweighted_and_weighted_means(self):
        metrics = pd.DataFrame(
            {
                "model_name": ["flatcfm", "flatcfm"],
                "metric": ["w2_squared", "w2_squared"],
                "metric_base": ["w2_squared", "w2_squared"],
                "category": ["cell_distribution", "cell_distribution"],
                "aggregation": ["none", "none"],
                "value": [1.0, 3.0],
                "n_ref": [1, 3],
            }
        )
        reduced = reduce_group_metrics(metrics)
        self.assertEqual(set(reduced["reduction"]), {"unweighted_mean", "cell_weighted_mean"})
        unweighted = float(reduced.loc[reduced["reduction"] == "unweighted_mean", "value"].iloc[0])
        weighted = float(reduced.loc[reduced["reduction"] == "cell_weighted_mean", "value"].iloc[0])
        self.assertAlmostEqual(unweighted, 2.0)
        self.assertAlmostEqual(weighted, 2.5)


class BenchmarkEvaluatorTest(unittest.TestCase):
    def test_evaluator_compares_multiple_models(self):
        ref = _make_reference_adata()
        evaluator = Evaluator(
            reference_adata=ref,
            perturbation_key="perturbation",
            covariate_keys=("cell_type",),
            control_value="ctrl",
        )
        out = evaluator.evaluate(
            {
                "flatcfm_a": _make_prediction_adata(scale=1.0),
                "flatcfm_b": _make_prediction_adata(scale=0.9),
            },
            metrics=("mean_gene_w1", "w2_squared"),
        )
        self.assertFalse(out.empty)
        self.assertEqual(set(out["model_name"]), {"flatcfm_a", "flatcfm_b"})

    def test_perturbench_adapter_compatibility(self):
        ref = _make_reference_adata()
        pred = _make_prediction_adata(scale=1.0)
        task = PerturbBenchTaskMetadata(
            perturbation_key="condition",
            covariate_keys=("cell_type",),
            obs_map={"perturbation": "condition"},
        )
        converted = to_perturbench_predictions(pred, ref[:, ["gene1", "gene2"]].copy(), task, model_name="flatcfm")
        self.assertIn("flatcfm", converted)
        evaluator = Evaluator(
            reference_adata=ref,
            perturbation_key="perturbation",
            covariate_keys=("cell_type",),
            control_value="ctrl",
        )
        out = evaluator.evaluate({"flatcfm": pred}, metrics=("mean_gene_w1",))
        self.assertFalse(out.empty)


if __name__ == "__main__":
    unittest.main()
