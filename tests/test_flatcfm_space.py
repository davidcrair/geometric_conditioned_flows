import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np

from flatcfm.data.space import NonlinearRFFLiftProjection, OrthogonalLiftProjection
from flatcfm.data.space import get_or_build_pipeline, normalize_space_config, pipeline_label, pipeline_tag_from_config


def _make_counts_adata() -> ad.AnnData:
    x = np.array(
        [
            [2.0, 0.0, 1.0, 3.0],
            [1.0, 4.0, 0.0, 2.0],
            [0.0, 2.0, 5.0, 1.0],
            [3.0, 1.0, 2.0, 0.0],
            [4.0, 0.0, 1.0, 2.0],
            [2.0, 3.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    adata = ad.AnnData(X=x)
    adata.var_names = ["gene1", "gene2", "gene3", "gene4"]
    return adata


def _make_toy_adata() -> ad.AnnData:
    x = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    adata = ad.AnnData(X=x)
    adata.var_names = ["x0", "x1"]
    return adata


class FlatCFMSpaceTest(unittest.TestCase):
    def test_normalize_space_config_and_labels(self):
        cfg = normalize_space_config(
            {
                "base": {"kind": "normalized_log1p", "feature_set": "all_genes", "n_hvgs": None, "target_sum": 1e4},
                "projections": [{"kind": "pca", "n_components": 2}],
                "fit_scope": "full_dataset",
            }
        )
        self.assertEqual(cfg["base"]["feature_set"], "all_genes")
        self.assertEqual(cfg["projections"][0]["kind"], "pca")
        self.assertEqual(pipeline_label(cfg), "normalized_log1p -> pca")
        self.assertIn("pca_npc2", pipeline_tag_from_config(cfg))

    def test_pca_pipeline_round_trip_shapes(self):
        adata = _make_counts_adata()
        cfg = {
            "base": {"kind": "normalized_log1p", "feature_set": "all_genes", "n_hvgs": None, "target_sum": 1e4},
            "projections": [{"kind": "pca", "n_components": 2}],
            "fit_scope": "train",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = get_or_build_pipeline(adata, cfg, space_path=Path(tmpdir) / "space.pkl")
        projected, lib, _ = pipeline.transform(adata)
        recovered = pipeline.inverse_to_raw(projected, lib)
        self.assertEqual(projected.shape, (adata.n_obs, 2))
        self.assertEqual(recovered.shape, adata.X.shape)
        self.assertTrue(np.all(recovered >= 0.0))

    def test_orthogonal_lift_preserves_distances_and_inverts(self):
        adata = _make_toy_adata()
        projection = OrthogonalLiftProjection(ambient_dim=16, seed=7)
        projection.fit(np.asarray(adata.X, dtype=np.float32), list(adata.var_names))
        lifted = projection.transform(np.asarray(adata.X, dtype=np.float32))
        recovered = projection.inverse_transform(lifted, np.ones(adata.n_obs, dtype=np.float32))

        original_dist = np.linalg.norm(np.asarray(adata.X)[0] - np.asarray(adata.X)[3])
        lifted_dist = np.linalg.norm(lifted[0] - lifted[3])
        self.assertAlmostEqual(float(original_dist), float(lifted_dist), places=5)
        self.assertTrue(np.allclose(recovered, np.asarray(adata.X, dtype=np.float32), atol=1e-5))

    def test_nonlinear_rff_config_and_tags(self):
        cfg = normalize_space_config(
            {
                "base": {"kind": "raw_counts", "feature_set": "all_genes", "n_hvgs": None, "target_sum": 1.0},
                "projections": [
                    {
                        "kind": "nonlinear_rff_lift",
                        "ambient_dim": 8,
                        "seed": 3,
                        "feature_scale": 2.0,
                    }
                ],
                "fit_scope": "train",
            }
        )
        self.assertEqual(cfg["projections"][0]["kind"], "nonlinear_rff_lift")
        self.assertEqual(pipeline_label(cfg), "raw_counts -> nonlinear_rff_lift")
        self.assertIn("nonlinearrff_d8_s3_fs2", pipeline_tag_from_config(cfg))

    def test_nonlinear_rff_lift_round_trip_and_identity_at_input_dim(self):
        adata = _make_toy_adata()
        matrix = np.asarray(adata.X, dtype=np.float32)
        projection = NonlinearRFFLiftProjection(ambient_dim=16, seed=11, feature_scale=1.5)
        projection.fit(matrix, list(adata.var_names))
        lifted = projection.transform(matrix)
        recovered = projection.inverse_transform(lifted, np.ones(adata.n_obs, dtype=np.float32))

        self.assertEqual(lifted.shape, (adata.n_obs, 16))
        self.assertEqual(projection.feature_names_out()[:2], list(adata.var_names))
        self.assertTrue(np.allclose(recovered, matrix, atol=1e-6))

        identity_projection = NonlinearRFFLiftProjection(ambient_dim=2, seed=11, feature_scale=1.5)
        identity_projection.fit(matrix, list(adata.var_names))
        identity_lift = identity_projection.transform(matrix)
        self.assertEqual(identity_lift.shape, matrix.shape)
        self.assertTrue(np.allclose(identity_lift, matrix, atol=1e-6))

    def test_orthogonal_lift_pca_pipeline_round_trip(self):
        adata = _make_toy_adata()
        cfg = {
            "base": {"kind": "raw_counts", "feature_set": "all_genes", "n_hvgs": None, "target_sum": 1.0},
            "projections": [
                {"kind": "orthogonal_lift", "ambient_dim": 16, "seed": 7},
                {"kind": "pca", "n_components": 2},
            ],
            "fit_scope": "train",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = get_or_build_pipeline(adata, cfg, space_path=Path(tmpdir) / "space.pkl")
        projected, lib, _ = pipeline.transform(adata)
        recovered = pipeline.inverse_to_raw(projected, lib)

        self.assertEqual(projected.shape, (adata.n_obs, 2))
        self.assertTrue(np.allclose(recovered, np.asarray(adata.X, dtype=np.float32), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
