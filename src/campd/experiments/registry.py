from typing import TYPE_CHECKING
from campd.utils.registry import Registry

if TYPE_CHECKING:
    from .base import BaseExperiment

EXPERIMENTS: Registry["BaseExperiment"] = Registry("experiments")
