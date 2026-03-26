from abc import ABC, abstractmethod
from typing import Any
from campd.data.trajectory_sample import TrajectorySample
from campd.utils.registry import Registry

VALIDATORS = Registry("validators")


class Validator(ABC):
    """
    Abstract base class for validators.
    """

    @abstractmethod
    def validate(self, batch: dict, output_dir: str) -> dict[str, Any]:
        """
        Validate the generated batch of trajectories.

        Args:
            batch: The batch of generated trajectories (e.g. TrajectorySample or dict).
            output_dir: Directory where the output is saved.

        Returns:
            A dictionary of validation metrics.
        """
        pass

    @classmethod
    def from_config(cls, cfg: Any) -> "Validator":
        return cls(**cfg)
