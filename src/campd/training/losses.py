import torch.nn as nn
from campd.training.registry import LOSSES

import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod


class WeightedLoss(nn.Module, ABC):

    def __init__(self, weights=None):
        super().__init__()
        self.register_buffer('weights', weights)

    @abstractmethod
    def _loss(self, pred, targ):
        raise NotImplementedError

    def forward(self, pred, targ):
        '''
            pred, targ : tensor
                [ batch_size x horizon x transition_dim ]
        '''
        loss = self._loss(pred, targ)
        if self.weights is not None:
            weighted_loss = (loss * self.weights).mean()
        else:
            weighted_loss = loss.mean()
        return weighted_loss, {}


@LOSSES.register("WeightedL1")
class WeightedL1(WeightedLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)


@LOSSES.register("WeightedL2")
class WeightedL2(WeightedLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')


# Register common losses
LOSSES.register("MSE")(nn.MSELoss)
LOSSES.register("L1")(nn.L1Loss)
