"""
Training loss functions for CAMPD.

Provides weighted variants of standard losses like L1 and L2 (MSE)
that can apply external weights to per-sample losses.
"""
import torch.nn as nn
from campd.training.registry import LOSSES

import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod


class WeightedLoss(nn.Module, ABC):
    """Base class for weighted loss functions.
    
    Subclasses must implement the :meth:`_loss` method to compute the unreduced
    per-element loss. This class handles applying the optional weights and
    computing the final mean.
    
    Args:
        weights (torch.Tensor, optional): Per-sample weights to apply. Defaults to None.
    """

    def __init__(self, weights=None):
        super().__init__()
        self.register_buffer('weights', weights)

    @abstractmethod
    def _loss(self, pred, targ):
        """Computes the unreduced per-element loss.
        
        Args:
            pred (torch.Tensor): Predictions tensor.
            targ (torch.Tensor): Target tensor.
            
        Returns:
            torch.Tensor: The unreduced loss tensor.
        """
        raise NotImplementedError

    def forward(self, pred, targ):
        """Computes the weighted loss.
        
        Args:
            pred (torch.Tensor): Predictions tensor of shape ``(batch_size, horizon, transition_dim)``.
            targ (torch.Tensor): Target tensor of shape ``(batch_size, horizon, transition_dim)``.
            
        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The computed weighted loss scalar.
                - dict: An empty dictionary for compatibility with the CAMPD trainer.
        """
        loss = self._loss(pred, targ)
        if self.weights is not None:
            weighted_loss = (loss * self.weights).mean()
        else:
            weighted_loss = loss.mean()
        return weighted_loss, {}


@LOSSES.register("WeightedL1")
class WeightedL1(WeightedLoss):
    """Weighted L1 (Mean Absolute Error) loss."""

    def _loss(self, pred, targ):
        """Computes the L1 (MAE) loss between predictions and targets without reduction."""
        return torch.abs(pred - targ)


@LOSSES.register("WeightedL2")
class WeightedL2(WeightedLoss):
    """Weighted L2 (Mean Squared Error) loss."""

    def _loss(self, pred, targ):
        """Computes the L2 (MSE) loss between predictions and targets without reduction."""
        return F.mse_loss(pred, targ, reduction='none')


# Register common losses
LOSSES.register("MSE")(nn.MSELoss)
LOSSES.register("L1")(nn.L1Loss)
