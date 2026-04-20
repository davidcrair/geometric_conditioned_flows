import json
import tempfile
import unittest
from pathlib import Path

import anndata as ad
from hydra import compose, initialize_config_dir
from lightning.pytorch import Trainer
from omegaconf import OmegaConf

from flatcfm.analysis import (
    build_flow_distribution_views,
    compute_grouped_flow_metrics,
    compute_per_perturbation_space_metrics,
    compute_flow_summary_metrics,
    evaluate_flow_predictions,
    get_or_build_flow_predictions,
    load_flow_run,
    select_heldout_perturbation,
)
from flatcfm.analysis.prediction_export import build_prediction_adata
from flatcfm.modelcore.utils import instantiate_datamodule, instantiate_model, save_json


CONFIG_DIR = Path(__file__).resolve().parents[1] / "src" / "flatcfm" / "configs"


def _history_for_task(task_name: str) -> dict:
    """build history payload"""

    if task_name == "ode":
        return {
            "train_loss": [1.0, 0.7],
            "val_loss": [1.1, 0.8],
            "individual_train_losses": {
                "ot": [0.4, 0.3],
                "density": [0.35, 0.25],
                "energy": [0.25, 0.15],
            },
            "individual_val_losses": {
                "ot": [0.5, 0.35],
                "density": [0.35, 0.3],
                "energy": [0.25, 0.15],
            },
        }
    return {
        "train_loss": [0.9, 0.4],
        "val_loss": [1.0, 0.5],
        "individual_train_losses": {"fm_mse": [0.9, 0.4]},
        "individual_val_losses": {"fm_mse": [1.0, 0.5]},
    }


def _materialize_run(root: Path, experiment: str, save_predictions: bool) -> Path:
    """materialize toy flow run"""

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(
            config_name="train",
            overrides=[
                f"experiment={experiment}",
                "trainer.accelerator=cpu",
                "trainer.devices=1",
                "task.epochs=1",
                "task.steps_per_epoch=2",
                "task.batch_size=32",
                "predict.batch_size=64",
            ],
        )

    datamodule = instantiate_datamodule(cfg)
    datamodule.setup("fit")
    model = instantiate_model(cfg, datamodule)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        limit_train_batches=2,
        limit_val_batches=1,
    )
    trainer.fit(model=model, datamodule=datamodule)
    datamodule.setup("predict")
    outputs = trainer.predict(model=model, datamodule=datamodule)
    matrix, obs, feature_names, prediction_metadata = datamodule.export_prediction_outputs(outputs)

    run_dir = root / str(experiment).replace("/", "_")
    checkpoint_path = run_dir / "checkpoints" / "best.ckpt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(str(checkpoint_path))
    cfg_for_save = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    cfg_for_save.callbacks = {}
    cfg_for_save.logger = {}
    OmegaConf.save(cfg_for_save, run_dir / "run_config.yaml", resolve=False)
    save_json(run_dir / "history.json", _history_for_task(str(cfg.task.name)))
    metadata = {
        "task_name": str(cfg.task.name),
        "space_mode": datamodule.get_space_mode(),
        "evaluation_space_mode": datamodule.get_evaluation_space_mode(),
        "checkpoint_path": str(checkpoint_path),
        "task_metadata": model.export_metadata(),
        "covariate_dicts": datamodule.covariate_dicts,
        "vocab_maps": datamodule.vocab_maps,
        "feature_names": datamodule.get_export_feature_names(),
        "condition_output_obs_map": datamodule.schema.output_obs_map,
        "ae_artifacts": {},
    }
    save_json(run_dir / "run_metadata.json", metadata)

    if save_predictions:
        pred_adata = build_prediction_adata(
            matrix,
            obs=obs,
            var_names=feature_names,
            uns={
                "run_dir": str(run_dir),
                "task_name": str(cfg.task.name),
                "prediction_name": "held_out",
                "generated_in_memory": False,
                "prediction_metadata": prediction_metadata,
            },
        )
        output_path = run_dir / "predictions" / "held_out"
        output_path.mkdir(parents=True, exist_ok=True)
        pred_adata.write_h5ad(output_path / "predictions.h5ad")

    return run_dir


class FlowResultsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory()
        root = Path(cls.tempdir.name)
        cls.fm_run_dir = _materialize_run(root, "toy/fm_identity", save_predictions=True)
        cls.ode_run_dir = _materialize_run(root, "toy/ode_identity", save_predictions=False)

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    def test_load_flow_run_fm(self):
        bundle = load_flow_run(run_dir=self.fm_run_dir)
        self.assertEqual(str(bundle.cfg.task.name), "fm")
        self.assertIn("fm_mse", bundle.history["individual_val_losses"])

    def test_load_flow_run_ode(self):
        bundle = load_flow_run(run_dir=self.ode_run_dir)
        self.assertEqual(str(bundle.cfg.task.name), "ode")
        self.assertIn("ot", bundle.history["individual_val_losses"])

    def test_get_or_build_predictions_uses_cache(self):
        bundle = load_flow_run(run_dir=self.fm_run_dir)
        pred_adata = get_or_build_flow_predictions(bundle, accelerator="cpu")
        self.assertIsInstance(pred_adata, ad.AnnData)
        self.assertFalse(bool(pred_adata.uns.get("generated_in_memory", False)))

    def test_get_or_build_predictions_fallback(self):
        bundle = load_flow_run(run_dir=self.ode_run_dir)
        pred_adata = get_or_build_flow_predictions(bundle, accelerator="cpu")
        self.assertIsInstance(pred_adata, ad.AnnData)
        self.assertTrue(bool(pred_adata.uns.get("generated_in_memory", False)))

    def test_get_or_build_predictions_supports_prediction_overrides(self):
        bundle = load_flow_run(run_dir=self.fm_run_dir)
        pred_adata = get_or_build_flow_predictions(
            bundle,
            prediction_name="train_same_split",
            prediction_overrides={
                "split": "train",
                "target_subset": "perturbed",
                "control_source": {"split": "train_controls", "match_cell_types_from": "none"},
            },
            accelerator="cpu",
        )
        spec = pred_adata.uns["prediction_metadata"]["prediction_spec"]
        self.assertEqual(spec["split"], "train")
        self.assertEqual(spec["control_source"]["split"], "train_controls")
        self.assertTrue(bool(pred_adata.uns.get("generated_in_memory", False)))

    def test_distribution_views_and_metrics(self):
        bundle = load_flow_run(run_dir=self.fm_run_dir)
        predictions = get_or_build_flow_predictions(bundle, accelerator="cpu")
        perturbation = select_heldout_perturbation(bundle, predictions)
        views = build_flow_distribution_views(bundle, predictions, perturbation)
        self.assertGreater(views["control_observed"].shape[0], 0)
        self.assertEqual(views["control_observed"].shape[0], views["perturbed_predicted_decoded"].shape[0])
        self.assertEqual(views["control_observed"].shape[1], len(views["feature_names"]))
        self.assertIn("prediction_vs_observed_w1", views["gene_metrics"].columns)

        metrics = compute_flow_summary_metrics(bundle, predictions)
        self.assertIn("best_val_loss", metrics)
        self.assertIn("mean_w2_squared", metrics)
        self.assertFalse(metrics["per_perturbation"].empty)
        self.assertTrue((metrics["per_perturbation"]["w2_squared"] >= 0).all())

        grouped = compute_grouped_flow_metrics(
            bundle,
            predictions,
            group_column=bundle.datamodule.schema.perturbation_source,
        )
        self.assertFalse(grouped.empty)
        self.assertIn(bundle.datamodule.schema.perturbation_source, grouped.columns)

    def test_space_metrics_for_train_and_val_predictions(self):
        bundle = load_flow_run(run_dir=self.fm_run_dir)
        train_predictions = get_or_build_flow_predictions(
            bundle,
            prediction_name="train_same_heldout_cell_type",
            prediction_overrides={
                "split": "train",
                "target_subset": "perturbed",
                "control_source": {
                    "split": "all_controls",
                    "match_cell_types_from": "held_out",
                },
            },
            accelerator="cpu",
        )
        train_w2 = compute_per_perturbation_space_metrics(
            bundle,
            train_predictions,
            prediction_name="train_same_heldout_cell_type",
            analysis_space="train_pca",
            pca_n_components=2,
            metric_names=("w2_squared",),
        )
        self.assertFalse(train_w2.empty)
        self.assertIn("w2_squared", train_w2.columns)
        self.assertTrue((train_w2["w2_squared"] >= 0).all())

        val_eval = evaluate_flow_predictions(
            bundle,
            get_or_build_flow_predictions(bundle, accelerator="cpu"),
            metric_space={"kind": "train_pca", "name": "train_pca", "pca_n_components": 2, "fit_split": "train"},
            metrics=("w2_squared",),
            prediction_name="held_out",
        )
        self.assertFalse(val_eval["per_group"].empty)
        self.assertEqual(set(val_eval["summary"]["reduction"]), {"unweighted_mean", "cell_weighted_mean"})
        self.assertIn("metric_space", val_eval["per_group"].columns)

        val_cosine = compute_per_perturbation_space_metrics(
            bundle,
            get_or_build_flow_predictions(bundle, accelerator="cpu"),
            analysis_space="train_base",
            metric_names=("cosine_log_fc",),
        )
        self.assertFalse(val_cosine.empty)
        self.assertIn("cosine_log_fc", val_cosine.columns)


if __name__ == "__main__":
    unittest.main()
