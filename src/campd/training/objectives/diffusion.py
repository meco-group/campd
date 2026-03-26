from campd.models import apply_hard_conditioning
from pydantic import BaseModel
from campd.utils.registry import Spec
from campd.data import TrajectorySample
from typing import Dict, Tuple
from campd.training.objectives.base import TrainingObjective
import torch
import torch.nn.functional as F
from campd.models.diffusion.model import ContextTrajectoryDiffusionModel

from ..registry import OBJECTIVES


class DiffusionObjectiveCfg(BaseModel):
    """
    Configuration for the DiffusionObjective.
    """
    loss_fn: Spec[torch.nn.Module]


@OBJECTIVES.register("DiffusionObjective")
class DiffusionObjective(TrainingObjective):
    """
    Training objective for diffusion models.

    Args:
        loss_fn (Spec[torch.nn.Module]): Specification for the loss function. Must be a callable that takes two arguments: the model output and the target.

    Example:
        >>> cfg = {"loss_fn": {"cls": "torch.nn.MSELoss"}}
        >>> objective = DiffusionObjective.from_config(cfg)
    """

    def __init__(self, cfg: DiffusionObjectiveCfg):
        super().__init__()
        self.cfg = cfg
        self.loss_fn = cfg.loss_fn.build()

    @classmethod
    def from_config(cls, cfg: DiffusionObjectiveCfg | dict):
        if isinstance(cfg, dict):
            cfg = DiffusionObjectiveCfg.model_validate(cfg)
        return cls(cfg)

    def step(self, model: ContextTrajectoryDiffusionModel, batch: TrajectorySample) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Computes the diffusion loss.
        """
        # Assumes model has noise_scheduler and provides add_noise, context_network, network, model_type
        # functionality similar to ContextTrajectoryDiffusionModel

        data = batch
        batch_size = data.trajectory.shape[0]

        # Access scheduler from model (handle DDP wrapper)
        unwrapped_model = model.module if hasattr(model, "module") else model
        noise_scheduler = unwrapped_model.noise_scheduler

        t = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                          (batch_size,), device=data.trajectory.device).long()
        noise = torch.randn_like(data.trajectory)
        noise[:, 0, :].fill_(0.0)
        noise[:, -1, :].fill_(0.0)
        hard_conds = batch.get_hard_conditions()

        x_noisy = unwrapped_model.add_noise(
            x_start=data.trajectory, t=t, noise=noise)
        x_noisy = apply_hard_conditioning(
            x_noisy, hard_conds)  # normally not needed

        # Call model forward pass (wraps DDP if applicable)
        model_out = model(x_noisy, t, data.context)

        # we don't apply hard conditioning to the output to let the model
        # learn to be as close as possible to the first/last timestep

        assert noise.shape == model_out.shape

        if unwrapped_model.model_type == 'epsilon':
            target = noise
        elif unwrapped_model.model_type == 'sample':
            target = data.trajectory
        elif unwrapped_model.model_type == 'v_prediction':
            target = noise_scheduler.get_velocity(
                data.trajectory, noise, t)
        else:
            raise NotImplementedError(
                f"Unknown model type {unwrapped_model.model_type}")

        loss = self.loss_fn(model_out, target)
        return {self.loss_fn.__class__.__name__: loss}, model_out, {}
