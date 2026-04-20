"""custom lightning callbacks"""

from __future__ import annotations

import sys

import lightning.pytorch as pl


class EpochProgressBar(pl.Callback):
    """progress bar that tracks epochs instead of batches

    uses two tqdm bars: one for training metrics and one for
    validation metrics so that loss components dont overflow
    a single line
    """

    def __init__(self):
        super().__init__()
        self._train_bar = None
        self._val_bar = None

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        from tqdm.auto import tqdm

        self._train_bar = tqdm(
            total=trainer.max_epochs,
            desc="Training",
            unit="epoch",
            file=sys.stderr,
            dynamic_ncols=True,
            position=0,
        )
        self._val_bar = tqdm(
            total=trainer.max_epochs,
            desc="     Val:",
            unit="epoch",
            file=sys.stderr,
            dynamic_ncols=True,
            position=1,
            bar_format="{desc}",
        )

    @staticmethod
    def _format_val(val: float) -> str:
        """format a metric value with adaptive precision"""

        if abs(val) >= 100:
            return f"{val:.1f}"
        if abs(val) >= 1:
            return f"{val:.3f}"
        return f"{val:.4f}"

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._train_bar is None:
            return
        train_metrics = {}
        val_metrics = {}
        for k, v in trainer.callback_metrics.items():
            try:
                fval = float(v)
                if fval == 0.0 and k not in ("train_loss", "val_loss"):
                    continue
                short_k = k.replace("train_", "").replace("val_", "")
                formatted = self._format_val(fval)
                if k.startswith("val_"):
                    val_metrics[short_k] = formatted
                else:
                    train_metrics[short_k] = formatted
            except (TypeError, ValueError):
                pass

        self._train_bar.set_postfix(train_metrics)
        self._train_bar.update(1)

        if self._val_bar is not None and val_metrics:
            parts = ", ".join(f"{k}={v}" for k, v in val_metrics.items())
            self._val_bar.set_description_str(f"     Val: {parts}")
            self._val_bar.update(1)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._val_bar is not None:
            self._val_bar.close()
        if self._train_bar is not None:
            self._train_bar.close()


class AutoencoderPhaseCallback(pl.Callback):
    """two phase ae schedule callback"""

    def __init__(self, schedule_cfg: dict):
        super().__init__()
        self.schedule_cfg = dict(schedule_cfg)
        self.phase1_epochs = int(self.schedule_cfg.get("phase1_epochs", 0))
        self.phase2_started = False

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """initialize phase 1"""

        del trainer
        if not hasattr(pl_module, "set_loss_weights") or not hasattr(pl_module, "set_trainable_parts"):
            return
        pl_module.set_loss_weights(self.schedule_cfg.get("phase1_loss_weights", {}))
        pl_module.set_trainable_parts(
            encoder=not bool(self.schedule_cfg.get("freeze_encoder_phase1", False)),
            decoder=not bool(self.schedule_cfg.get("freeze_decoder_phase1", False)),
        )

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """switch to phase 2"""

        if self.phase2_started:
            return
        if self.phase1_epochs <= 0 or pl_module.current_epoch < self.phase1_epochs:
            return
        if not hasattr(pl_module, "set_loss_weights") or not hasattr(pl_module, "set_trainable_parts"):
            return
        pl_module.set_loss_weights(self.schedule_cfg.get("phase2_loss_weights", {}))
        pl_module.set_trainable_parts(
            encoder=not bool(self.schedule_cfg.get("freeze_encoder_phase2", False)),
            decoder=not bool(self.schedule_cfg.get("freeze_decoder_phase2", False)),
        )
        self.phase2_started = True

        # reset checkpoint state so phase 1 scores do not prevent
        # phase 2 checkpoints from being saved and remove stale
        # phase 1 checkpoint files
        for cb in trainer.callbacks:
            if isinstance(cb, pl.callbacks.ModelCheckpoint):
                old_path = cb.best_model_path
                cb.best_model_score = None
                cb.best_model_path = ""
                cb.best_k_models.clear()
                cb.kth_best_model_path = ""
                cb.kth_value = None
                cb.current_score = None
                if old_path:
                    from pathlib import Path
                    p = Path(old_path)
                    if p.exists():
                        p.unlink()
                        cb.last_model_path = ""
