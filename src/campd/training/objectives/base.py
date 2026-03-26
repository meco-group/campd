from abc import ABC, abstractmethod
from typing import Dict, Tuple
import torch
from torch import nn


class TrainingObjective(nn.Module, ABC):
    """
    Abstract base class for training objectives.
    """
    @abstractmethod
    def step(self, model: nn.Module, batch: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computes the losses for a training step.

        Args:
            model (nn.Module): The model being trained.
            batch (torch.Tensor): The input batch.

        Returns:
            Tuple[Dict[str, torch.Tensor], torch.Tensor]: A dictionary of scalar loss terms and the features 
            used to compute the losses.
        """
        pass
