from __future__ import annotations
from typing import TYPE_CHECKING
from campd.utils.registry import Registry

if TYPE_CHECKING:
    from campd.training.callbacks import Callback
    from campd.training.summary import Summary
    from campd.training.objectives.base import TrainingObjective
    from torch import nn

LOSSES: Registry[nn.Module] = Registry("losses")
CALLBACKS: Registry[Callback] = Registry("callbacks")
SUMMARIES: Registry[Summary] = Registry("summaries")
OBJECTIVES: Registry[TrainingObjective] = Registry("objectives")
