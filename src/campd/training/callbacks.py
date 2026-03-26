from __future__ import annotations

from abc import ABC
from typing import Any, Dict, Optional
import os
import copy
from typing import TYPE_CHECKING

import numpy as np
import torch

from campd.training.registry import CALLBACKS

if TYPE_CHECKING:
    from campd.training.base import Trainer


class Callback(ABC):
    def _is_main_process(self, trainer) -> bool:
        """Return True if this is the main process (or no accelerator is in use)."""
        acc = getattr(trainer, "accelerator", None)
        if acc is None:
            return True
        return acc.is_main_process

    def on_train_start(self, trainer):
        pass

    def on_fit_start(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass

    def on_epoch_start(self, trainer):
        pass

    def on_epoch_end(self, trainer, train_losses: Optional[Dict[str, float]] = None):
        pass

    def on_batch_start(self, trainer, batch):
        pass

    def on_batch_end(self, trainer, batch, loss_dict: Dict[str, float]):
        pass

    def on_validation_start(self, trainer):
        pass

    def on_validation_end(self, trainer, val_losses: Dict[str, float]):
        pass

    def on_summary_end(self, trainer, summary, step: int, payload: Any):
        pass


@CALLBACKS.register("PrinterCallback")
class PrinterCallback(Callback):
    def on_train_start(self, trainer):
        if self._is_main_process(trainer):
            print("Training started")

    def on_train_end(self, trainer):
        if self._is_main_process(trainer):
            print("Training ended")


class EMA:
    """
    (empirical) exponential moving average parameters
    """

    def __init__(self, beta=0.995):
        self.beta = beta

    def update_model_average(self, ema_model, current_model):
        for ema_params, current_params in zip(ema_model.parameters(), current_model.parameters()):
            old_weight, up_weight = ema_params.data, current_params.data
            ema_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


@CALLBACKS.register("EMACallback")
class EMACallback(Callback):
    def __init__(self, decay: float = 0.995, start_step: int = 2000, update_every: int = 10):
        self.decay = decay
        self.start_step = start_step
        self.update_every = update_every
        self.ema: Optional[EMA] = None

    def on_train_start(self, trainer):
        if trainer.model is None:
            return

        # Initialize EMA model in trainer
        trainer.ema_model = copy.deepcopy(trainer.model)
        for param in trainer.ema_model.parameters():
            param.requires_grad = False

        self.ema = EMA(beta=self.decay)

    def on_batch_end(self, trainer, batch, loss_dict):
        if trainer.ema_model is None:
            return

        if trainer.global_step % self.update_every == 0:
            if trainer.global_step < self.start_step:
                # Reset ema parameters to match matches
                # Handle DDP wrapping
                current_model = trainer.model
                if getattr(trainer, "accelerator", None) is not None:
                    current_model = trainer.accelerator.unwrap_model(
                        current_model)
                trainer.ema_model.load_state_dict(current_model.state_dict())

            self.ema.update_model_average(trainer.ema_model, trainer.model)


@CALLBACKS.register("EarlyStoppingCallback")
class EarlyStoppingCallback(Callback):
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def on_validation_end(self, trainer, val_losses):
        # Assuming 'total_loss' is in val_losses or we take the mean of all losses
        # If 'total' key exists, use it, otherwise sum all values
        if not val_losses:
            return

        current_val_loss = val_losses.get('total', sum(val_losses.values()))

        if current_val_loss < (self.min_validation_loss - self.min_delta):
            self.min_validation_loss = current_val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self._is_main_process(trainer):
                    print(
                        f"Early stopping triggered at epoch {getattr(trainer, 'current_epoch', 0)}")
                trainer.stop_training = True


@CALLBACKS.register("CheckpointCallback")
class CheckpointCallback(Callback):
    def __init__(self, save_every_epochs: int = 1, save_best: bool = True, save_last: bool = True,
                 save_last_every_steps: int = 5000):
        self.save_every_epochs = save_every_epochs
        self.save_best = save_best
        self.save_last = save_last
        self.save_last_every_steps = save_last_every_steps
        self.best_val_loss = float('inf')

    def on_batch_end(self, trainer: Trainer, *args, **kwargs):
        if self.save_last and self.save_last_every_steps is not None \
                and trainer.global_step % self.save_last_every_steps == 0:
            self.save_checkpoint(trainer, "last.pth")

    def on_epoch_end(self, trainer: Trainer, train_losses: Optional[Dict[str, float]] = None):
        if (trainer.current_epoch + 1) % self.save_every_epochs == 0:
            self.save_checkpoint(
                trainer, f"calibration_epoch_{trainer.current_epoch:04d}.pth")

        if self.save_last:
            self.save_checkpoint(trainer, "last.pth")

    def on_validation_end(self, trainer: Trainer, val_losses: Dict[str, float]):
        if not self.save_best:
            return

        current_val_loss = val_losses.get('total', sum(val_losses.values()))
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.save_checkpoint(trainer, "best.pth")

    def save_checkpoint(self, trainer: Trainer, filename: str):
        if not self._is_main_process(trainer):
            return
        if not trainer.results_dir:
            return

        checkpoints_dir = os.path.join(trainer.results_dir, 'checkpoints')
        os.makedirs(checkpoints_dir, exist_ok=True)

        path = os.path.join(checkpoints_dir, filename)

        model = trainer.model
        if getattr(trainer, "accelerator", None) is not None:
            model = trainer.accelerator.unwrap_model(model)

        state = {
            'epoch': trainer.current_epoch,
            'global_step': trainer.global_step,
            'state_dict': model.state_dict(),
            'optimizer': trainer.optimizer.state_dict(),
        }

        if trainer.ema_model is not None:
            state['ema_state_dict'] = trainer.ema_model.state_dict()

        torch.save(state, path)


def _as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (float, int, bool, np.number)):
        return float(x)
    if torch.is_tensor(x):
        if x.numel() == 0:
            return None
        if x.numel() == 1:
            return float(x.detach().item())
        return None
    return None


def _is_image_array(arr: Any) -> bool:
    if not isinstance(arr, np.ndarray):
        return False
    if arr.ndim == 2:
        return True
    if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
        return True
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        return True
    return False


@CALLBACKS.register("WandBCallback")
class WandBCallback(Callback):
    """Minimal Weights & Biases logging callback.

    Assumptions:
      - A wandb run is already initialized externally (per your start_wandb())
      - If wandb is not installed or no run is active, this callback becomes a no-op.

    Logs:
      - on_batch_end: training losses (step-based)
      - on_validation_end: validation losses (step-based)
      - on_summary_end: any summary payload (scalars + common visualization types)
    """

    def __init__(
        self,
        log_train_every_n_steps: int = 1,
        log_val_every_n_epochs: int = 1,
        log_summaries: bool = True,
        train_prefix: str = "TRAIN: ",
        val_prefix: str = "VAL: ",
        commit: bool = False,
    ):
        self.log_train_every_n_steps = int(log_train_every_n_steps)
        self.log_val_every_n_epochs = int(log_val_every_n_epochs)
        self.log_summaries = bool(log_summaries)
        self.train_prefix = train_prefix
        self.val_prefix = val_prefix
        self.commit = bool(commit)

    def _get_wandb(self):
        try:
            import wandb  # type: ignore
        except Exception:
            return None
        try:
            if getattr(wandb, "run", None) is None:
                return None
        except Exception:
            return None
        return wandb

    def _log(self, metrics: Dict[str, Any], step: Optional[int] = None, trainer=None) -> None:
        if trainer is not None and not self._is_main_process(trainer):
            return
        wandb = self._get_wandb()
        if wandb is None:
            return
        if not metrics:
            return
        kwargs: Dict[str, Any] = {"commit": self.commit}
        if step is not None:
            kwargs["step"] = int(step)
        wandb.log(metrics, **kwargs)

    def on_batch_end(self, trainer, batch, loss_dict: Dict[str, float]):
        if self.log_train_every_n_steps <= 0:
            return
        step = int(getattr(trainer, "global_step", 0))
        if step % self.log_train_every_n_steps != 0:
            return

        metrics: Dict[str, Any] = {}
        for k, v in (loss_dict or {}).items():
            metrics[f"{self.train_prefix}{k}"] = float(v)

        if metrics:
            self._log(metrics, step=step, trainer=trainer)

    def on_validation_end(self, trainer, val_losses: Dict[str, float]):
        if self.log_val_every_n_epochs <= 0:
            return
        epoch = int(getattr(trainer, "current_epoch", 0))
        if (epoch + 1) % self.log_val_every_n_epochs != 0:
            return

        step = int(getattr(trainer, "global_step", 0))
        metrics: Dict[str, Any] = {}
        for k, v in (val_losses or {}).items():
            metrics[f"{self.val_prefix}{k}"] = float(v)
        if metrics:
            self._log(metrics, step=step, trainer=trainer)

    def on_summary_end(self, trainer, summary, step: int, payload: Any):
        if not self.log_summaries:
            return
        if payload is None:
            return

        wandb = self._get_wandb()
        if wandb is None:
            return

        summary_name = getattr(summary, "__class__", type(summary)).__name__
        base = f"{summary_name}: "

        def convert_value(v: Any) -> Any:
            fv = _as_float(v)
            if fv is not None:
                return fv

            if torch.is_tensor(v):
                t = v.detach().cpu()
                arr = t.numpy()
                if _is_image_array(arr):
                    return wandb.Image(arr)
                return None

            if isinstance(v, np.ndarray):
                if v.size == 1:
                    return float(v.reshape(-1)[0])
                if _is_image_array(v):
                    return wandb.Image(v)
                return None

            # matplotlib figure support (optional)
            try:
                import matplotlib.figure  # type: ignore

                if isinstance(v, matplotlib.figure.Figure):
                    return wandb.Image(v)
            except Exception:
                pass

            # PIL Image support (optional)
            try:
                from PIL import Image as PILImage  # type: ignore

                if isinstance(v, PILImage.Image):
                    return wandb.Image(v)
            except Exception:
                pass

            # pass-through for already-converted wandb types
            if v.__class__.__module__.startswith("wandb"):
                return v

            return None

        metrics: Dict[str, Any] = {}
        if isinstance(payload, dict):
            for k, v in payload.items():
                cv = convert_value(v)
                if cv is None:
                    continue
                metrics[f"{base}{k}"] = cv
        else:
            cv = convert_value(payload)
            if cv is not None:
                metrics[f"{base}value"] = cv

        if metrics:
            self._log(metrics, step=int(step), trainer=trainer)
