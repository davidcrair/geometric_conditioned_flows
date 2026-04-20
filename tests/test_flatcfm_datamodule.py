import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
import torch

from flatcfm.data.splitters import SplitConfig, build_holdout_manifest, save_manifest_json
from flatcfm.data.ae_dataloader import AEBatchDataset
from flatcfm.data.accessors import resolve_sciplex_artifacts
from flatcfm.data.datamodules import SciplexDataModule, ToyDataModule
from flatcfm.data.space import load_ae_projection
from flatcfm.models.autoencoder import NBAutoEncoder, StandardAutoEncoder


def _make_sciplex_stub(n_per_group: int = 4) -> ad.AnnData:
    obs_rows = []
    idx = 0
    for product in ["drugA", "drugB"]:
        for rep in ["R1", "R2"]:
            for _ in range(n_per_group):
                obs_rows.append(
                    {
                        "obs_name": f"cell_{idx}",
                        "product_name": product,
                        "product_dose": f"{product}_10",
                        "cell_type": "K562",
                        "vehicle": 0,
                        "replicate": rep,
                    }
                )
                idx += 1
    for rep in ["R1", "R2"]:
        for _ in range(n_per_group):
            obs_rows.append(
                {
                    "obs_name": f"cell_{idx}",
                    "product_name": "vehicle",
                    "product_dose": "vehicle_0",
                    "cell_type": "K562",
                    "vehicle": 1,
                    "replicate": rep,
                }
            )
            idx += 1
    obs = pd.DataFrame(obs_rows).set_index("obs_name")
    x = np.arange(len(obs) * 4, dtype=np.float32).reshape(len(obs), 4) + 1.0
    adata = ad.AnnData(X=x, obs=obs)
    adata.var_names = ["gene1", "gene2", "gene3", "gene4"]
    return adata


def _make_sciplex_multitype_stub(n_per_group: int = 3) -> ad.AnnData:
    obs_rows = []
    idx = 0
    for cell_type in ["K562", "A549"]:
        for product in ["drugA", "drugB"]:
            for rep in ["R1", "R2"]:
                for _ in range(n_per_group):
                    obs_rows.append(
                        {
                            "obs_name": f"cell_{idx}",
                            "product_name": product,
                            "product_dose": f"{product}_10",
                            "cell_type": cell_type,
                            "vehicle": 0,
                            "replicate": rep,
                        }
                    )
                    idx += 1
        for rep in ["R1", "R2"]:
            for _ in range(n_per_group):
                obs_rows.append(
                    {
                        "obs_name": f"cell_{idx}",
                        "product_name": "vehicle",
                        "product_dose": "vehicle_0",
                        "cell_type": cell_type,
                        "vehicle": 1,
                        "replicate": rep,
                    }
                )
                idx += 1
    obs = pd.DataFrame(obs_rows).set_index("obs_name")
    x = np.arange(len(obs) * 4, dtype=np.float32).reshape(len(obs), 4) + 1.0
    adata = ad.AnnData(X=x, obs=obs)
    adata.var_names = ["gene1", "gene2", "gene3", "gene4"]
    return adata


class FlatCFMDataModuleTest(unittest.TestCase):
    def test_ae_batch_dataset_uses_input_library_size_for_normalized_inputs(self):
        obs = pd.DataFrame(index=["cell_0", "cell_1"])
        adata = ad.AnnData(X=np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float32), obs=obs)
        adata.var_names = ["gene1", "gene2"]
        dataset = AEBatchDataset(
            adata,
            batch_size=2,
            shuffle=False,
            library_size=np.array([2.0, 1.0], dtype=np.float32),
            input_library_size=np.array([4.0, 2.0], dtype=np.float32),
            input_space_kind="normalized_log1p",
            target_sum=10.0,
        )

        batch = dataset[0]
        expected = np.log1p(np.array([[2.0 / 4.0 * 10.0, 0.0], [0.0, 1.0 / 2.0 * 10.0]], dtype=np.float32))

        self.assertIn("input_lib_size", batch)
        np.testing.assert_allclose(batch["x_input"].numpy(), expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(batch["lib_size"].numpy(), np.array([2.0, 1.0], dtype=np.float32))
        np.testing.assert_allclose(batch["input_lib_size"].numpy(), np.array([4.0, 2.0], dtype=np.float32))

    def test_standard_autoencoder_reconstruct_counts_accepts_sample_flag(self):
        model = StandardAutoEncoder(n_genes=3, latent_dim=2)
        z = torch.zeros((2, 2), dtype=torch.float32)
        library_size = torch.ones(2, dtype=torch.float32)

        counts = model.reconstruct_counts(z, library_size, sample=True)

        self.assertEqual(tuple(counts.shape), (2, 3))

    def test_nb_autoencoder_decode_is_not_simplex_constrained(self):
        model = NBAutoEncoder(n_genes=3, latent_dim=2, mean_head="per_cell_gene")
        with torch.no_grad():
            model.dec_log_rate.weight.zero_()
            model.dec_log_rate.bias.zero_()

        z = torch.zeros((2, 2), dtype=torch.float32)
        library_size = torch.tensor([10.0, 20.0], dtype=torch.float32)

        mu, theta = model.decode(z, library_size)

        self.assertEqual(tuple(theta.shape), (3,))
        np.testing.assert_allclose(mu.sum(dim=-1).detach().cpu().numpy(), np.array([30.0, 60.0], dtype=np.float32))
        self.assertFalse(torch.allclose(mu.sum(dim=-1), library_size))

    def test_nb_autoencoder_reconstruct_input_uses_input_library_size(self):
        model = NBAutoEncoder(
            n_genes=2,
            latent_dim=2,
            mean_head="per_cell_gene",
            input_space_kind="normalized_log1p",
            target_sum=10.0,
        )
        with torch.no_grad():
            model.dec_log_rate.weight.zero_()
            model.dec_log_rate.bias.copy_(torch.log(torch.tensor([0.5, 0.25], dtype=torch.float32)))

        z = torch.zeros((1, 2), dtype=torch.float32)
        library_size = torch.tensor([20.0], dtype=torch.float32)
        input_library_size = torch.tensor([40.0], dtype=torch.float32)

        recon = model.reconstruct_input(z, library_size, input_library_size=input_library_size)

        expected_mu = np.array([[10.0, 5.0]], dtype=np.float32)
        expected = np.log1p(expected_mu / 40.0 * 10.0)
        np.testing.assert_allclose(recon.detach().cpu().numpy(), expected, rtol=1e-6, atol=1e-6)

    def test_nb_autoencoder_forward_is_finite_with_full_library_size(self):
        model = NBAutoEncoder(
            n_genes=3,
            latent_dim=2,
            mean_head="per_cell_gene",
            input_space_kind="normalized_log1p",
            target_sum=10.0,
        )

        x_raw = torch.tensor([[2.0, 1.0, 0.0], [0.0, 1.0, 4.0]], dtype=torch.float32)
        library_size = torch.tensor([20.0, 30.0], dtype=torch.float32)
        x_input = torch.log1p(x_raw / library_size.unsqueeze(-1) * 10.0)

        loss, z = model(x_input, x_raw, library_size)

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(z).all())

    def test_sciplex_datamodule_builds_fit_and_predict_loaders(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cfg = {
                "paths": {
                    "split_artifact_dir": str(root / "splits"),
                    "data_dir": str(root / "data"),
                    "model_dir": str(root / "models"),
                    "space_dir": str(root / "spaces"),
                    "run_dir": str(root / "runs"),
                },
                "data": {"name": "sciplex", "source": "local", "data_path": None},
                "splitter": {
                    "seed": 42,
                    "test_cell_type": "K562",
                    "holdout_fraction": 0.5,
                    "subsample_seed": 0,
                    "subsample_n_cells": 100,
                    "split_policy": "strict_no_leakage",
                    "ae_subsample_seed": 42,
                    "ae_subsample_n_cells": 10,
                    "ae_subsample_group_cols": ["cell_type", "vehicle"],
                    "val_fraction": 0.2,
                },
                "space": {
                    "base": {"kind": "normalized_log1p", "feature_set": "all_genes", "n_hvgs": None, "target_sum": 1e4},
                    "projections": [],
                    "fit_scope": "train",
                    "chunk_size": 8,
                },
                "evaluation_space": {"copy_from_space": True, "fit_scope": "full_dataset"},
                "condition": {
                    "perturbation": {"name": "perturbation", "source_column": "product_dose"},
                    "control_column": "vehicle",
                    "control_value": 1,
                    "perturbation_covariates": [],
                    "sample_covariates": [
                        {"name": "cell_type", "source_column": "cell_type"},
                    ],
                    "output_obs_map": {"perturbation": "condition"},
                },
                "ae_geometry": {"mode": "none", "source": "none", "cache_path": None, "phate": {}},
                "task": {"name": "fm", "batch_size": 8, "use_sampler": True, "steps_per_epoch": 2},
                "predict": {
                    "batch_size": 4,
                    "name": "held_out",
                    "split": "held_out",
                    "target_subset": "perturbed",
                    "target_filters": {"obs_equals": {}, "obs_in": {}},
                    "control_source": {"split": "all_controls", "match_cell_types_from": "test_cell_type"},
                    "sample_decode": False,
                },
                "trainer": {"num_workers": 0, "pin_memory": False},
            }

            artifacts = resolve_sciplex_artifacts(cfg["splitter"], cfg["paths"])
            adata = _make_sciplex_stub()
            artifacts.subsample_h5ad_path.parent.mkdir(parents=True, exist_ok=True)
            adata.write_h5ad(artifacts.subsample_h5ad_path)
            manifest = build_holdout_manifest(adata, SplitConfig(seed=42, test_cell_type="K562", holdout_fraction=0.5))
            artifacts.holdout_json_path.parent.mkdir(parents=True, exist_ok=True)
            save_manifest_json(manifest, artifacts.holdout_json_path)

            dm = SciplexDataModule(**cfg)
            dm.setup("fit")
            train_batch = next(iter(dm.train_dataloader()))
            self.assertEqual(train_batch["x_0"].shape[1], 4)
            self.assertEqual(dm.get_input_dim(), 4)
            self.assertEqual(dm.get_space_mode(), "normalized_log1p")
            self.assertEqual(dm.get_evaluation_space_mode(), "normalized_log1p")

            dm.setup("predict")
            predict_batch = next(iter(dm.predict_dataloader()))
            self.assertIn("x_ctrl", predict_batch)
            self.assertGreater(len(predict_batch["obs"]), 0)
            self.assertEqual(dm.predict_spec["split"], "held_out")
            self.assertEqual(dm.predict_metadata["prediction_spec"]["control_source"]["match_cell_types_from"], "test_cell_type")

    def test_toy_datamodule_builds_predict_loader(self):
        dm = ToyDataModule(
            data={"name": "toy", "dataset_name": "gaussian_to_moons", "n_samples": 128},
            splitter={"seed": 42, "val_fraction": 0.1},
            space={
                "base": {"kind": "raw_counts", "feature_set": "all_genes", "n_hvgs": None, "target_sum": 1.0},
                "projections": [],
                "fit_scope": "train",
                "chunk_size": 128,
            },
            evaluation_space={"copy_from_space": True, "fit_scope": "full_dataset"},
            condition={
                "perturbation": {"name": "perturbation", "source_column": "perturbation"},
                "control_column": "vehicle",
                "control_value": 1.0,
                "perturbation_covariates": [],
                "sample_covariates": [],
                "output_obs_map": {"perturbation": "condition"},
            },
            paths={},
            ae_geometry={"mode": "none", "source": "none", "cache_path": None, "phate": {}},
            task={"name": "fm", "batch_size": 16, "use_sampler": True, "steps_per_epoch": 2},
            predict={
                "batch_size": 8,
                "name": "held_out",
                "split": "held_out",
                "target_subset": "perturbed",
                "target_filters": {"obs_equals": {}, "obs_in": {}},
                "control_source": {"split": "all_controls", "match_cell_types_from": "test_cell_type"},
            },
            trainer={"num_workers": 0, "pin_memory": False},
        )
        dm.setup("fit")
        self.assertEqual(dm.get_input_dim(), 2)
        batch = next(iter(dm.train_dataloader()))
        self.assertEqual(batch["x_1"].shape[1], 2)
        dm.setup("predict")
        predict_batch = next(iter(dm.predict_dataloader()))
        self.assertEqual(predict_batch["x_ctrl"].shape[1], 2)
        self.assertGreater(pd.Index(dm.predict_dataset.control_obs_names).nunique(), 1)

    def test_toy_lifted_datamodule_exports_predictions_in_canonical_2d(self):
        dm = ToyDataModule(
            data={"name": "toy", "dataset_name": "gaussian_to_moons", "n_samples": 128},
            splitter={"seed": 42, "val_fraction": 0.1},
            space={
                "base": {"kind": "raw_counts", "feature_set": "all_genes", "n_hvgs": None, "target_sum": 1.0},
                "projections": [{"kind": "orthogonal_lift", "ambient_dim": 16, "seed": 0}],
                "fit_scope": "train",
                "chunk_size": 128,
            },
            evaluation_space={
                "base": {"kind": "raw_counts", "feature_set": "all_genes", "n_hvgs": None, "target_sum": 1.0},
                "projections": [],
                "fit_scope": "full_dataset",
                "chunk_size": 128,
            },
            condition={
                "perturbation": {"name": "perturbation", "source_column": "perturbation"},
                "control_column": "vehicle",
                "control_value": 1.0,
                "perturbation_covariates": [],
                "sample_covariates": [],
                "output_obs_map": {"perturbation": "condition"},
            },
            paths={},
            ae_geometry={"mode": "none", "source": "none", "cache_path": None, "phate": {}},
            task={"name": "fm", "batch_size": 16, "use_sampler": True, "steps_per_epoch": 2},
            predict={
                "batch_size": 8,
                "name": "held_out",
                "split": "held_out",
                "target_subset": "perturbed",
                "target_filters": {"obs_equals": {}, "obs_in": {}},
                "control_source": {"split": "all_controls", "match_cell_types_from": "test_cell_type"},
                "sample_decode": False,
            },
            trainer={"num_workers": 0, "pin_memory": False},
        )
        dm.setup("fit")
        self.assertEqual(dm.get_input_dim(), 16)
        dm.setup("predict")
        decoded = dm.decode_predictions(torch.zeros((4, 16), dtype=torch.float32), np.ones(4, dtype=np.float32))
        self.assertEqual(decoded.shape[1], 2)

    def test_toy_lifted_pca_datamodule_exports_predictions_in_canonical_2d(self):
        dm = ToyDataModule(
            data={"name": "toy", "dataset_name": "gaussian_to_moons", "n_samples": 128},
            splitter={"seed": 42, "val_fraction": 0.1},
            space={
                "base": {"kind": "raw_counts", "feature_set": "all_genes", "n_hvgs": None, "target_sum": 1.0},
                "projections": [
                    {"kind": "orthogonal_lift", "ambient_dim": 16, "seed": 0},
                    {"kind": "pca", "n_components": 2},
                ],
                "fit_scope": "train",
                "chunk_size": 128,
            },
            evaluation_space={
                "base": {"kind": "raw_counts", "feature_set": "all_genes", "n_hvgs": None, "target_sum": 1.0},
                "projections": [],
                "fit_scope": "full_dataset",
                "chunk_size": 128,
            },
            condition={
                "perturbation": {"name": "perturbation", "source_column": "perturbation"},
                "control_column": "vehicle",
                "control_value": 1.0,
                "perturbation_covariates": [],
                "sample_covariates": [],
                "output_obs_map": {"perturbation": "condition"},
            },
            paths={},
            ae_geometry={"mode": "none", "source": "none", "cache_path": None, "phate": {}},
            task={"name": "fm", "batch_size": 16, "use_sampler": True, "steps_per_epoch": 2},
            predict={
                "batch_size": 8,
                "name": "held_out",
                "split": "held_out",
                "target_subset": "perturbed",
                "target_filters": {"obs_equals": {}, "obs_in": {}},
                "control_source": {"split": "all_controls", "match_cell_types_from": "test_cell_type"},
                "sample_decode": False,
            },
            trainer={"num_workers": 0, "pin_memory": False},
        )
        dm.setup("fit")
        self.assertEqual(dm.get_input_dim(), 2)
        dm.setup("predict")
        decoded = dm.decode_predictions(torch.zeros((4, 2), dtype=torch.float32), np.ones(4, dtype=np.float32))
        self.assertEqual(decoded.shape[1], 2)

    def test_sciplex_prediction_spec_supports_train_split_and_filters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cfg = {
                "paths": {
                    "split_artifact_dir": str(root / "splits"),
                    "data_dir": str(root / "data"),
                    "model_dir": str(root / "models"),
                    "space_dir": str(root / "spaces"),
                    "run_dir": str(root / "runs"),
                },
                "data": {"name": "sciplex", "source": "local", "data_path": None},
                "splitter": {
                    "seed": 42,
                    "test_cell_type": "K562",
                    "holdout_fraction": 0.5,
                    "subsample_seed": 0,
                    "subsample_n_cells": 200,
                    "split_policy": "strict_no_leakage",
                    "ae_subsample_seed": 42,
                    "ae_subsample_n_cells": 10,
                    "ae_subsample_group_cols": ["cell_type", "vehicle"],
                    "val_fraction": 0.2,
                },
                "space": {
                    "base": {"kind": "normalized_log1p", "feature_set": "all_genes", "n_hvgs": None, "target_sum": 1e4},
                    "projections": [],
                    "fit_scope": "train",
                    "chunk_size": 8,
                },
                "evaluation_space": {"copy_from_space": True, "fit_scope": "full_dataset"},
                "condition": {
                    "perturbation": {"name": "perturbation", "source_column": "product_dose"},
                    "control_column": "vehicle",
                    "control_value": 1,
                    "perturbation_covariates": [],
                    "sample_covariates": [
                        {"name": "cell_type", "source_column": "cell_type"},
                    ],
                    "output_obs_map": {"perturbation": "condition"},
                },
                "ae_geometry": {"mode": "none", "source": "none", "cache_path": None, "phate": {}},
                "task": {"name": "fm", "batch_size": 8, "use_sampler": True, "steps_per_epoch": 2},
                "predict": {
                    "batch_size": 4,
                    "name": "train_k562_drugA",
                    "split": "train",
                    "target_subset": "perturbed",
                    "target_filters": {"obs_equals": {}, "obs_in": {}},
                    "control_source": {"split": "train_controls", "match_cell_types_from": "held_out"},
                    "sample_decode": False,
                },
                "trainer": {"num_workers": 0, "pin_memory": False},
            }

            artifacts = resolve_sciplex_artifacts(cfg["splitter"], cfg["paths"])
            adata = _make_sciplex_multitype_stub()
            artifacts.subsample_h5ad_path.parent.mkdir(parents=True, exist_ok=True)
            adata.write_h5ad(artifacts.subsample_h5ad_path)
            manifest = build_holdout_manifest(adata, SplitConfig(seed=42, test_cell_type="K562", holdout_fraction=0.5))
            artifacts.holdout_json_path.parent.mkdir(parents=True, exist_ok=True)
            save_manifest_json(manifest, artifacts.holdout_json_path)
            train_product = next(
                product for product in ["drugA", "drugB"] if product not in set(manifest["selected_holdout_product_names"])
            )
            cfg["predict"]["target_filters"]["obs_equals"]["product_name"] = train_product
            cfg["predict"]["target_filters"]["obs_equals"]["cell_type"] = "K562"

            dm = SciplexDataModule(**cfg)
            dm.setup("predict")
            obs = dm.predict_dataset.obs
            self.assertTrue((obs["cell_type"].astype(str) == "K562").all())
            self.assertTrue((obs["product_name"].astype(str) == train_product).all())
            self.assertTrue((obs["vehicle"].to_numpy() == 0).all())
            control_obs = dm.adata_full.obs.loc[dm.predict_dataset.control_obs_names]
            self.assertTrue((control_obs["cell_type"].astype(str) == "K562").all())
            self.assertEqual(dm.predict_spec["split"], "train")
            self.assertEqual(dm.predict_spec["control_source"]["split"], "train_controls")
            self.assertEqual(dm.predict_metadata["prediction_spec"]["target_filters"]["obs_equals"]["product_name"], train_product)

    def test_sciplex_ae_datamodule_supports_phate_geometry_and_export(self):
        class FakePHATE:
            def __init__(self, **kwargs):
                del kwargs
                self.diff_potential = None

            def fit(self, matrix):
                matrix = np.asarray(matrix, dtype=np.float32)
                width = min(3, matrix.shape[1])
                self.diff_potential = matrix[:, :width]
                return self

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cfg = {
                "paths": {
                    "split_artifact_dir": str(root / "splits"),
                    "data_dir": str(root / "data"),
                    "model_dir": str(root / "models"),
                    "space_dir": str(root / "spaces"),
                    "run_dir": str(root / "runs"),
                },
                "data": {"name": "sciplex", "source": "local", "data_path": None},
                "splitter": {
                    "seed": 42,
                    "test_cell_type": "K562",
                    "holdout_fraction": 0.5,
                    "subsample_seed": 0,
                    "subsample_n_cells": 100,
                    "split_policy": "strict_no_leakage",
                    "ae_subsample_seed": 42,
                    "ae_subsample_n_cells": 12,
                    "ae_subsample_group_cols": ["cell_type", "vehicle"],
                    "val_fraction": 0.25,
                },
                "space": {
                    "base": {"kind": "normalized_log1p", "feature_set": "all_genes", "n_hvgs": None, "target_sum": 1e4},
                    "projections": [],
                    "fit_scope": "train",
                    "chunk_size": 8,
                },
                "evaluation_space": {"copy_from_space": True, "fit_scope": "full_dataset"},
                "condition": {
                    "perturbation": {"name": "perturbation", "source_column": "product_dose"},
                    "control_column": "vehicle",
                    "control_value": 1,
                    "perturbation_covariates": [],
                    "sample_covariates": [
                        {"name": "cell_type", "source_column": "cell_type"},
                    ],
                    "output_obs_map": {"perturbation": "condition"},
                },
                "ae_geometry": {
                    "mode": "phate_potential",
                    "source": "phate_potential",
                    "cache_path": None,
                    "phate": {"n_components": 2, "knn": 5, "n_landmark": 10, "t": "auto", "verbose": False},
                },
                "task": {"name": "ae", "batch_size": 6, "lr": 1e-3},
                "predict": {
                    "batch_size": 4,
                    "name": "held_out",
                    "split": "held_out",
                    "target_subset": "perturbed",
                    "target_filters": {"obs_equals": {}, "obs_in": {}},
                    "control_source": {"split": "all_controls", "match_cell_types_from": "test_cell_type"},
                    "sample_decode": False,
                },
                "trainer": {"num_workers": 0, "pin_memory": False},
            }

            artifacts = resolve_sciplex_artifacts(cfg["splitter"], cfg["paths"])
            adata = _make_sciplex_stub()
            artifacts.subsample_h5ad_path.parent.mkdir(parents=True, exist_ok=True)
            adata.write_h5ad(artifacts.subsample_h5ad_path)
            manifest = build_holdout_manifest(adata, SplitConfig(seed=42, test_cell_type="K562", holdout_fraction=0.5))
            artifacts.holdout_json_path.parent.mkdir(parents=True, exist_ok=True)
            save_manifest_json(manifest, artifacts.holdout_json_path)

            fake_module = types.SimpleNamespace(PHATE=FakePHATE)
            with patch.dict("sys.modules", {"phate": fake_module}):
                dm = SciplexDataModule(**cfg)
                dm.setup("fit")
                batch = next(iter(dm.train_dataloader()))
                self.assertIn("distances", batch)
                self.assertEqual(batch["x_input"].shape[1], 4)
                self.assertIn("input_lib_size", batch)

                ae_model = NBAutoEncoder(n_genes=4, latent_dim=2)
                exported = dm.export_ae_artifacts(ae_model, checkpoint_path=None, run_dir=root / "run")
                self.assertTrue(Path(exported["ae_projection_path"]).exists())
                self.assertIn("normalized_log1p_all_genes", Path(exported["ae_projection_path"]).name)
                loaded_projection = load_ae_projection(Path(exported["ae_projection_path"]))
                latent = loaded_projection.transform(np.asarray(dm.ae_train_loader.dataset.x_input[:2], dtype=np.float32))
                self.assertEqual(tuple(latent.shape), (2, 2))


if __name__ == "__main__":
    unittest.main()
