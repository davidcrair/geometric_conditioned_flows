import json
import tempfile
import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir
from lightning.pytorch import Trainer
from omegaconf import OmegaConf

from flatcfm.analysis import run_benchmark_suite
from flatcfm.analysis.prediction_export import build_prediction_adata
from flatcfm.modelcore.predictors import (
    BaselineRunPredictor,
    FMRunPredictor,
    ODERunPredictor,
    load_predictor_from_config,
)
from flatcfm.modelcore.utils import instantiate_datamodule, instantiate_model, save_json


CONFIG_DIR = Path(__file__).resolve().parents[1] / "src" / "flatcfm" / "configs"


def _history_for_task(task_name: str) -> dict:
    """build history payload"""

    if task_name == "ode":
        return {
            "train_loss": [1.0, 0.7],
            "val_loss": [1.1, 0.8],
            "individual_train_losses": {"ot": [0.4, 0.3], "density": [0.35, 0.25], "energy": [0.25, 0.15]},
            "individual_val_losses": {"ot": [0.5, 0.35], "density": [0.35, 0.3], "energy": [0.25, 0.15]},
        }
    return {
        "train_loss": [0.9, 0.4],
        "val_loss": [1.0, 0.5],
        "individual_train_losses": {"fm_mse": [0.9, 0.4]},
        "individual_val_losses": {"fm_mse": [1.0, 0.5]},
    }


def _materialize_run(root: Path, experiment: str) -> Path:
    """materialize toy run"""

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
    save_json(
        run_dir / "run_metadata.json",
        {
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
        },
    )
    output_path = run_dir / "predictions" / "held_out"
    output_path.mkdir(parents=True, exist_ok=True)
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
    pred_adata.write_h5ad(output_path / "predictions.h5ad")
    return run_dir


class BenchmarkRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory()
        root = Path(cls.tempdir.name)
        cls.fm_run_dir = _materialize_run(root, "toy/fm_identity")
        cls.ode_run_dir = _materialize_run(root, "toy/ode_identity")
        cls.baseline_run_dir = _materialize_run(root, "toy/baseline_no_effect")

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    def test_predictor_factory(self):
        fm = load_predictor_from_config({"kind": "run_fm", "name": "fm", "run_dir": str(self.fm_run_dir)})
        ode = load_predictor_from_config({"kind": "run_ode", "name": "ode", "run_dir": str(self.ode_run_dir)})
        baseline = load_predictor_from_config({"kind": "run_baseline", "name": "no_effect", "run_dir": str(self.baseline_run_dir)})
        self.assertIsInstance(fm, FMRunPredictor)
        self.assertIsInstance(ode, ODERunPredictor)
        self.assertIsInstance(baseline, BaselineRunPredictor)

    def test_run_benchmark_suite(self):
        result = run_benchmark_suite(
            {
                "anchor_run_dir": str(self.fm_run_dir),
                "prediction_name": "held_out",
                "predictors": [
                    {"kind": "run_fm", "name": "fm", "run_dir": str(self.fm_run_dir)},
                    {"kind": "run_ode", "name": "ode", "run_dir": str(self.ode_run_dir)},
                    {"kind": "run_baseline", "name": "no_effect", "run_dir": str(self.baseline_run_dir)},
                ],
                "metric_spaces": [{"name": "comparison", "kind": "comparison", "fit_split": "full_dataset"}],
                "metrics": ["mean_gene_w1", "w2_squared"],
                "group_columns": ["perturbation"],
                "reductions": ["unweighted_mean", "cell_weighted_mean"],
                "accelerator": "cpu",
            }
        )
        self.assertIn("per_group_metrics", result)
        self.assertIn("summary_metrics", result)
        self.assertIn("predictions", result)
        self.assertFalse(result["per_group_metrics"].empty)
        self.assertFalse(result["summary_metrics"].empty)
        self.assertEqual(set(result["predictions"].keys()), {"fm", "ode", "no_effect"})
        self.assertTrue({"unweighted_mean", "cell_weighted_mean"}.issubset(set(result["summary_metrics"]["reduction"])))


if __name__ == "__main__":
    unittest.main()
