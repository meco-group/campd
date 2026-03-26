from abc import ABC, abstractmethod
from campd.models.diffusion import ContextTrajectoryDiffusionModel
from torch.utils.data import DataLoader
from campd.data import TrajectorySample
from typing import Dict, Any


class Summary(ABC):
    """
    Class to generate summaries of the capabilities of the model.
    """

    def __init__(self, every_n_steps: int = 1, run_first: bool = False):
        self.every_n_steps = every_n_steps
        self.run_first = run_first

    def should_run(self, step: int) -> bool:
        return step % self.every_n_steps == 0 and (step > 0 or self.run_first)

    def run(self, trainer, step: int = None) -> Any:
        if step is None:
            step = trainer.global_step

        assert self.should_run(step), "Summary should not run at this step"

        model = trainer.model
        train_dataloader = trainer.train_dataloader
        val_dataloader = trainer.val_dataloader

        is_training = model.training
        if is_training:
            model.eval()

        acc = getattr(trainer, "accelerator", None)
        unwrapped = acc.unwrap_model(model) if acc else model
        summary_results = self._run(
            unwrapped, train_dataloader, val_dataloader, step)
        if is_training:
            model.train()

        return summary_results

    @abstractmethod
    def _run(
        self,
        model: ContextTrajectoryDiffusionModel,
        train_dataloader: DataLoader[TrajectorySample],
        val_dataloader: DataLoader[TrajectorySample],
        step: int,
    ) -> Any:
        pass
