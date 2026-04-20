"""optuna search spaces for hydra sweeps"""

from __future__ import annotations

from typing import Any


def _projection_kinds(cfg: Any) -> set[str]:
    """collect projection kinds"""

    projections = getattr(cfg.space, "projections", [])
    return {str(item.kind) for item in projections}


def _suggest_condition_encoder(trial: Any) -> None:
    """suggest shared condition encoder params"""

    trial.suggest_categorical("model.condition_dim", [64, 128, 256])
    trial.suggest_categorical("model.embedding_dim", [32, 64, 128])
    trial.suggest_categorical("model.projection_dim", [32, 64, 128])


def configure_baseline_linear_search_space(cfg: Any, trial: Any) -> None:
    """configure linear baseline search space"""

    del cfg
    trial.suggest_float("task.lr", 1e-4, 5e-2, log=True)
    trial.suggest_float("task.weight_decay", 1e-6, 1e-2, log=True)
    trial.suggest_categorical("task.batch_size", [512, 1024, 2048])
    trial.suggest_categorical("task.steps_per_epoch", [50, 100, 200])
    trial.suggest_categorical("task.epochs", [50, 100, 200])
    _suggest_condition_encoder(trial)


def configure_baseline_decoder_search_space(cfg: Any, trial: Any) -> None:
    """configure decoder baseline search space"""

    del cfg
    trial.suggest_float("task.lr", 1e-4, 5e-2, log=True)
    trial.suggest_float("task.weight_decay", 1e-6, 1e-2, log=True)
    trial.suggest_categorical("task.batch_size", [512, 1024, 2048])
    trial.suggest_categorical("task.steps_per_epoch", [50, 100, 200])
    trial.suggest_categorical("task.epochs", [50, 100, 200])
    trial.suggest_categorical("model.hidden_dim", [128, 256, 512])
    _suggest_condition_encoder(trial)


def configure_flow_matching_search_space(cfg: Any, trial: Any) -> None:
    """configure flow matching search space"""

    projection_kinds = _projection_kinds(cfg)
    batch_choices = [256, 512, 1024] if "ae_latent" in projection_kinds else [512, 1024]

    trial.suggest_float("task.lr", 3e-5, 3e-3, log=True)
    trial.suggest_float("task.weight_decay", 1e-6, 3e-2, log=True)
    trial.suggest_categorical("task.batch_size", batch_choices)
    trial.suggest_categorical("task.steps_per_epoch", [50, 100, 200])
    use_ot = trial.suggest_categorical("task.use_ot_coupling", [True, False])
    if use_ot:
        trial.suggest_float("task.ot_reg", 0.01, 5e-1, log=True)
    trial.suggest_float("task.flow_noise", 1e-4, 1e-1, log=True)
    trial.suggest_categorical("model.hidden_dim", [128, 256, 512])
    trial.suggest_int("model.hidden_layers", 2, 6)
    trial.suggest_float("model.dropout", 0.0, 0.3)
    _suggest_condition_encoder(trial)


def configure_ode_search_space(cfg: Any, trial: Any) -> None:
    """configure ode search space"""

    projection_kinds = _projection_kinds(cfg)
    batch_choices = [128, 256, 512] if "ae_latent" not in projection_kinds else [128, 256]

    trial.suggest_float("task.lr", 1e-5, 1e-3, log=True)
    trial.suggest_float("task.weight_decay", 1e-6, 3e-2, log=True)
    trial.suggest_categorical("task.batch_size", batch_choices)
    trial.suggest_categorical("task.steps_per_epoch", [25, 50, 100])
    trial.suggest_categorical("task.ode_method", ["midpoint", "rk4"])
    trial.suggest_categorical("task.adjoint", [False, True])
    trial.suggest_categorical("task.n_energy_steps", [5, 10, 20])
    trial.suggest_float("loss.weights.ot", 1e-1, 5.0, log=True)
    trial.suggest_float("loss.weights.density", 1e-1, 5.0, log=True)
    trial.suggest_float("loss.weights.energy", 1e-4, 1e-1, log=True)
    trial.suggest_float("loss.sinkhorn_reg", 1e-3, 5e-1, log=True)
    trial.suggest_categorical("loss.sinkhorn_max_iter", [20, 50, 100])
    trial.suggest_categorical("model.hidden_dim", [128, 256, 512])
    trial.suggest_int("model.hidden_layers", 2, 6)
    trial.suggest_float("model.dropout", 0.0, 0.2)
    _suggest_condition_encoder(trial)


def configure_autoencoder_search_space(cfg: Any, trial: Any) -> None:
    """configure autoencoder search space"""

    trial.suggest_float("task.lr", 1e-4, 3e-3, log=True)
    trial.suggest_float("task.weight_decay", 1e-7, 1e-2, log=True)
    trial.suggest_categorical("task.batch_size", [128, 256, 512])
    trial.suggest_categorical("model.latent_dim", [32, 64, 128, 256])
    trial.suggest_categorical("model.hidden_dim", [128, 256, 512])
    trial.suggest_int("model.n_layers", 2, 4)
    trial.suggest_float("model.dropout", 0.0, 0.3)

    geometry_mode = str(getattr(cfg.ae_geometry, "mode", "none"))
    if geometry_mode != "none":
        trial.suggest_float("loss.weights.distance", 1e-3, 1.0, log=True)
        trial.suggest_categorical("loss.distance_loss_type", ["mse", "cosine"])
        trial.suggest_float("loss.distance_zeta", 0.0, 0.5)

    if str(cfg.space.base.kind) == "normalized_log1p":
        trial.suggest_float("loss.weights.log1p_mse", 1e-3, 1.0, log=True)
