import tempfile
import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir
from lightning.pytorch import Trainer

from flatcfm.modelcore.utils import instantiate_datamodule, instantiate_model


CONFIG_DIR = Path(__file__).resolve().parents[1] / "src" / "flatcfm" / "configs"


class FlatCFMRuntimeTest(unittest.TestCase):
    def _fit_model(self, experiment: str):
        with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
            cfg = compose(
                config_name="train",
                overrides=[
                    f"experiment={experiment}",
                    "task.epochs=1",
                    "task.steps_per_epoch=2",
                    "task.batch_size=32",
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
        outputs = trainer.predict(model=model, datamodule=datamodule)
        matrix, obs, feature_names, metadata = datamodule.export_prediction_outputs(outputs)
        self.assertGreater(matrix.shape[0], 0)
        self.assertEqual(matrix.shape[1], len(feature_names))
        self.assertGreater(len(obs), 0)
        self.assertGreater(len(feature_names), 0)
        self.assertIn("task_name", metadata)
        self.assertIn("prediction_spec", metadata)
        self.assertEqual(metadata["prediction_spec"]["split"], "held_out")

    def test_toy_fm_runtime(self):
        self._fit_model("toy/fm_identity")

    def test_toy_ode_runtime(self):
        self._fit_model("toy/ode_identity")

    def test_toy_lifted_fm_runtime(self):
        self._fit_model("toy/lifted")

    def test_toy_lifted_ode_runtime(self):
        with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
            cfg = compose(
                config_name="train",
                overrides=[
                    "experiment=toy/lifted",
                    "task=ode",
                    "model=ode",
                    "loss=ode",
                    "task.epochs=1",
                    "task.steps_per_epoch=2",
                    "task.batch_size=32",
                    "space.projections.0.ambient_dim=16",
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
        outputs = trainer.predict(model=model, datamodule=datamodule)
        matrix, obs, feature_names, metadata = datamodule.export_prediction_outputs(outputs)
        self.assertEqual(matrix.shape[1], 2)
        self.assertEqual(matrix.shape[1], len(feature_names))
        self.assertGreater(len(obs), 0)
        self.assertGreater(len(feature_names), 0)
        self.assertTrue(metadata["used_inverse_export_path"])


if __name__ == "__main__":
    unittest.main()
