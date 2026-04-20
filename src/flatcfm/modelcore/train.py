"""hydra lightning training entrypoint"""

from __future__ import annotations

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
from omegaconf import DictConfig, OmegaConf

import warnings

from flatcfm._utils import to_plain_dict
from flatcfm.modelcore.callbacks import AutoencoderPhaseCallback
from flatcfm.modelcore.utils import extract_history_from_metrics, get_metrics_csv_path, instantiate_collection, instantiate_datamodule, instantiate_loggers, instantiate_model, resolve_hpo_objective, save_json, save_run_config


def _verify_two_phase_checkpoint(cfg: DictConfig, checkpoint_path: str | None) -> None:
    """verify that the saved checkpoint contains phase 2 weights

    in two-phase ae training the best checkpoint must come from phase 2
    (recon) not phase 1 (distance) otherwise the decoder is untrained
    """

    if checkpoint_path is None:
        return
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    log_theta = sd.get("model.log_theta")
    if log_theta is not None and log_theta.abs().max().item() < 1e-6:
        warnings.warn(
            "two-phase AE checkpoint has log_theta = 0 for all genes — "
            "this means the checkpoint is from phase 1 (decoder untrained). "
            "Check that AutoencoderPhaseCallback resets ModelCheckpoint.best_model_score "
            "at the phase transition.",
            stacklevel=2,
        )


def _resolve_checkpoint_path(callbacks: list) -> str | None:
    """resolve checkpoint path"""

    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint):
            if callback.best_model_path:
                return callback.best_model_path
            if callback.last_model_path:
                return callback.last_model_path
    return None


def _trainer_cfg(cfg: DictConfig) -> dict:
    """build trainer cfg"""

    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer_cfg.pop("num_workers", None)
    trainer_cfg.pop("pin_memory", None)
    return trainer_cfg


def _build_callbacks(cfg: DictConfig) -> list:
    """build callbacks"""

    callbacks = instantiate_collection(cfg.callbacks)
    if str(cfg.task.name) == "ae" and str(cfg.ae_schedule.name) == "two_phase":
        callbacks.append(AutoencoderPhaseCallback(to_plain_dict(cfg.ae_schedule)))
    return callbacks


def _apply_ae_schedule(cfg: DictConfig, model) -> None:
    """apply ae schedule defaults"""

    if str(cfg.task.name) != "ae":
        return
    if str(cfg.ae_schedule.name) == "two_phase":
        model.set_loss_weights(to_plain_dict(cfg.ae_schedule.phase1_loss_weights))
        model.set_trainable_parts(
            encoder=not bool(cfg.ae_schedule.freeze_encoder_phase1),
            decoder=not bool(cfg.ae_schedule.freeze_decoder_phase1),
        )
        return
    model.set_loss_weights(to_plain_dict(cfg.loss.weights))
    model.set_trainable_parts(encoder=True, decoder=True)


def _apply_best_checkpoint_if_present(model, checkpoint_path: str | None) -> None:
    """apply best checkpoint if present"""

    if checkpoint_path is None:
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)


def _maybe_export_ae_artifacts(cfg: DictConfig, datamodule, model, checkpoint_path: str | None, output_dir: Path) -> dict:
    """export ae artifacts if needed"""

    if str(cfg.task.name) != "ae":
        return {}
    _apply_best_checkpoint_if_present(model, checkpoint_path)
    return datamodule.export_ae_artifacts(model.model, checkpoint_path, output_dir)


def _trainer_cfg_for_run(cfg: DictConfig) -> dict:
    """build trainer cfg for run"""

    trainer_cfg = _trainer_cfg(cfg)
    if str(cfg.task.name) == "ae" and str(cfg.ae_schedule.name) == "two_phase":
        trainer_cfg["max_epochs"] = int(cfg.ae_schedule.phase1_epochs) + int(cfg.ae_schedule.phase2_epochs)
    return trainer_cfg


def _hpo_enabled(cfg: DictConfig) -> bool:
    """check if hpo is enabled"""

    if "hpo" not in cfg or cfg.hpo is None:
        return False
    return bool(cfg.hpo.get("enabled", False))


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    """main"""

    seed_everything(int(cfg.seed), workers=True)
    torch.set_float32_matmul_precision("high")
    output_dir = Path(HydraConfig.get().runtime.output_dir)

    callbacks = _build_callbacks(cfg)
    logger = instantiate_loggers(cfg.logger)
    datamodule = instantiate_datamodule(cfg)
    datamodule.setup("fit")
    model = instantiate_model(cfg, datamodule)
    _apply_ae_schedule(cfg, model)
    trainer = instantiate(
        _trainer_cfg_for_run(cfg),
        callbacks=callbacks,
        logger=logger,
        default_root_dir=str(output_dir),
    )
    try:
        trainer.fit(model=model, datamodule=datamodule)
        checkpoint_path = _resolve_checkpoint_path(callbacks)
        save_run_config(output_dir / "run_config.yaml", cfg)
        history = extract_history_from_metrics(get_metrics_csv_path(logger))
        hpo_metric_name = None
        hpo_objective = None
        if _hpo_enabled(cfg):
            hpo_metric_name, hpo_objective = resolve_hpo_objective(history, to_plain_dict(cfg.hpo))
            import math
            if hpo_objective is None or math.isnan(hpo_objective) or math.isinf(hpo_objective):
                raise ValueError(f"invalid hpo objective value: {hpo_objective}")
    except Exception as e:
        if _hpo_enabled(cfg):
            import logging
            logging.getLogger(__name__).error("trial failed: %s", e)
            return float("inf")
        raise
    metadata = {
        "task_name": cfg.task.name,
        "space_mode": datamodule.get_space_mode(),
        "evaluation_space_mode": datamodule.get_evaluation_space_mode(),
        "checkpoint_path": checkpoint_path,
        "task_metadata": model.export_metadata(),
        "covariate_dicts": datamodule.covariate_dicts,
        "vocab_maps": datamodule.vocab_maps,
        "feature_names": datamodule.get_export_feature_names(),
        "condition_output_obs_map": datamodule.schema.output_obs_map,
        "ae_artifacts": _maybe_export_ae_artifacts(cfg, datamodule, model, checkpoint_path, output_dir),
        "hpo": {
            "enabled": _hpo_enabled(cfg),
            "objective_metric": hpo_metric_name,
            "objective_value": hpo_objective,
        },
    }
    save_json(output_dir / "run_metadata.json", metadata)
    save_json(output_dir / "history.json", history)

    if str(cfg.task.name) == "ae" and str(cfg.ae_schedule.name) == "two_phase":
        _verify_two_phase_checkpoint(cfg, checkpoint_path)

    print(str(output_dir))
    if checkpoint_path is not None:
        print(checkpoint_path)
    if hpo_objective is not None:
        return float(hpo_objective)
    return {
        "run_dir": str(output_dir),
        "checkpoint_path": checkpoint_path,
        "history": history,
        "metadata": metadata,
    }


if __name__ == "__main__":
    main()
