import unittest
from pathlib import Path

import optuna
from hydra import compose, initialize_config_dir
from hydra.utils import get_method

from flatcfm.modelcore.utils import resolve_hpo_objective


CONFIG_DIR = Path(__file__).resolve().parents[1] / "src" / "flatcfm" / "configs"


class HPOConfigTest(unittest.TestCase):
    def test_resolve_hpo_objective_best_and_last(self):
        history = {
            "train_loss": [2.0, 1.0],
            "val_loss": [1.5, 0.75],
            "individual_train_losses": {"fm_mse": [2.0, 1.0]},
            "individual_val_losses": {"fm_mse": [1.5, 0.75]},
        }

        metric_name, best_value = resolve_hpo_objective(
            history,
            {"objective": {"name": "val_loss", "mode": "min", "source": "best"}},
        )
        self.assertEqual(metric_name, "val_loss")
        self.assertEqual(best_value, 0.75)

        metric_name, last_value = resolve_hpo_objective(
            history,
            {"objective": {"name": "val_fm_mse", "mode": "min", "source": "last"}},
        )
        self.assertEqual(metric_name, "val_fm_mse")
        self.assertEqual(last_value, 0.75)

    def test_resolve_hpo_objective_raises_for_missing_metric(self):
        with self.assertRaisesRegex(ValueError, "cannot resolve hpo objective"):
            resolve_hpo_objective(
                {"train_loss": [], "val_loss": [], "individual_train_losses": {}, "individual_val_losses": {}},
                {"objective": {"name": "val_loss", "mode": "min", "source": "best"}},
            )

    def test_hpo_configs_compose_and_search_space_callables_resolve(self):
        fixed_trials = {
            "hpo=baseline_linear": {
                "task.lr": 1e-3,
                "task.weight_decay": 1e-4,
                "task.batch_size": 1024,
                "task.steps_per_epoch": 100,
                "task.epochs": 100,
                "model.condition_dim": 128,
                "model.embedding_dim": 64,
                "model.projection_dim": 64,
            },
            "hpo=baseline_decoder": {
                "task.lr": 1e-3,
                "task.weight_decay": 1e-4,
                "task.batch_size": 1024,
                "task.steps_per_epoch": 100,
                "task.epochs": 100,
                "model.hidden_dim": 256,
                "model.condition_dim": 128,
                "model.embedding_dim": 64,
                "model.projection_dim": 64,
            },
            "hpo=fm": {
                "task.lr": 1e-4,
                "task.weight_decay": 1e-3,
                "task.batch_size": 512,
                "task.steps_per_epoch": 100,
                "task.use_ot_coupling": True,
                "task.ot_reg": 0.1,
                "task.flow_noise": 1e-3,
                "model.hidden_dim": 256,
                "model.hidden_layers": 4,
                "model.dropout": 0.1,
                "model.condition_dim": 128,
                "model.embedding_dim": 64,
                "model.projection_dim": 64,
            },
            "hpo=ode": {
                "task.lr": 1e-4,
                "task.weight_decay": 1e-3,
                "task.batch_size": 256,
                "task.steps_per_epoch": 50,
                "task.ode_method": "midpoint",
                "task.adjoint": False,
                "task.n_energy_steps": 10,
                "loss.weights.ot": 1.0,
                "loss.weights.density": 1.0,
                "loss.weights.energy": 0.01,
                "loss.sinkhorn_reg": 0.1,
                "loss.sinkhorn_max_iter": 50,
                "model.hidden_dim": 256,
                "model.hidden_layers": 4,
                "model.dropout": 0.1,
                "model.condition_dim": 128,
                "model.embedding_dim": 64,
                "model.projection_dim": 64,
            },
            "hpo=ae": {
                "task.lr": 1e-3,
                "task.weight_decay": 1e-5,
                "task.batch_size": 256,
                "model.latent_dim": 128,
                "model.hidden_dim": 256,
                "model.n_layers": 3,
                "model.dropout": 0.1,
                "loss.weights.log1p_mse": 0.05,
            },
        }
        experiment_overrides = {
            "hpo=baseline_linear": "experiment=sciplex/baseline_linear",
            "hpo=baseline_decoder": "experiment=sciplex/baseline_decoder",
            "hpo=fm": "experiment=sciplex/fm_log1p",
            "hpo=ode": "experiment=sciplex/ode_log1p",
            "hpo=ae": "experiment=sciplex/ae_log1p_recon",
        }

        with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
            for hpo_override, fixed_params in fixed_trials.items():
                cfg = compose(
                    config_name="train",
                    overrides=[experiment_overrides[hpo_override], hpo_override],
                    return_hydra_config=True,
                )
                self.assertTrue(cfg.hpo.enabled)
                self.assertEqual(cfg.hpo.objective.name, "val_loss")
                self.assertEqual(str(cfg.hydra.mode.name), "MULTIRUN")
                self.assertEqual(cfg.hydra.sweeper.direction, "minimize")
                custom_search_space = cfg.hydra.sweeper.custom_search_space
                if custom_search_space is None:
                    continue
                trial = optuna.trial.FixedTrial(fixed_params)
                search_space_fn = get_method(custom_search_space)
                search_space_fn(cfg, trial)
                self.assertTrue(trial.params)


if __name__ == "__main__":
    unittest.main()
