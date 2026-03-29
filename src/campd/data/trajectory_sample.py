from __future__ import annotations
from campd.data.normalization import DatasetNormalizer
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import torch
from campd.data.context import TrajectoryContext


@dataclass
class TrajectorySample:
    """
    Represents a single sample from the trajectory dataset.
    """
    trajectory: torch.Tensor
    context: Optional[TrajectoryContext] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    is_normalized: bool = False
    is_batched: bool = False

    shared_context: bool = False

    def __post_init__(self):
        if self.context is not None and not self.context.is_batched and self.is_batched:
            self.shared_context = True

    def to(self, device: torch.device, non_blocking: bool = False) -> 'TrajectorySample':
        """Move all tensors to device."""

        if self.trajectory is not None:
            self.trajectory = self.trajectory.to(
                device, non_blocking=non_blocking)

        if self.context is not None:
            self.context = self.context.to(device, non_blocking=non_blocking)

        return self

    @classmethod
    def collate(cls, samples: List[TrajectorySample]) -> TrajectorySample:
        """
        Collate a list of samples into a TrajectorySample (batched).
        """
        traj = torch.stack([s.trajectory for s in samples]
                           ) if samples[0].trajectory is not None else None

        ctx = TrajectoryContext.collate(
            [s.context for s in samples]) if samples[0].context is not None else None

        return cls(
            trajectory=traj,
            context=ctx,
            metadata={k: [s.metadata.get(k) for s in samples]
                      for k in samples[0].metadata},
            is_batched=True,
            is_normalized=samples[0].is_normalized,
            shared_context=False
        )

    def normalize(self, normalizer: DatasetNormalizer) -> TrajectorySample:
        """
        Normalize the trajectory and context.
        """
        if self.is_normalized:
            return self

        context_normalized = self.context.normalize(normalizer)
        trajectory_normalized = normalizer.normalize(self.trajectory, 'traj')

        return self.__class__(
            trajectory=trajectory_normalized,
            context=context_normalized,
            metadata=self.metadata,
            is_normalized=True,
            is_batched=self.is_batched,
            shared_context=self.shared_context
        )

    def denormalize(self, normalizer: DatasetNormalizer, denormalize_context: bool = True) -> TrajectorySample:
        """
        Denormalize the trajectory and context.

        Args:
            normalizer (DatasetNormalizer): Normalizer to use for denormalization.
            denormalize_context (bool, optional): Whether to denormalize the context. Defaults to True.
        """
        if not self.is_normalized:
            return self

        trajectory_denormalized = normalizer.unnormalize(
            self.trajectory, 'traj')
        if self.context is not None and denormalize_context:
            context_denormalized = self.context.denormalize(normalizer)
        else:
            context_denormalized = self.context

        return self.__class__(
            trajectory=trajectory_denormalized,
            context=context_denormalized,
            metadata=self.metadata,
            is_normalized=False,
            is_batched=self.is_batched,
            shared_context=self.shared_context
        )

    def __len__(self) -> int:
        # return self.trajectory_normalized.shape[0]
        if self.is_batched:
            return self.trajectory.shape[0]
        else:
            return 1

    def expand_shared_context(self):
        """
        If shared_context is True, duplicates the context for each batch element
        so that shared_context becomes False.
        """
        if not self.shared_context:
            return self

        assert self.is_batched, "no need to expand shared context for non-batched samples"

        self.context = self.context.expand(len(self))
        self.shared_context = False

    def get_hard_conditions(self) -> Dict[str | int, torch.Tensor]:
        """
        Get hard boundary conditions (start and goal states).

        Returns:
            Dictionary mapping timestep to state
        """
        if self.context is not None:
            return self.context.get_hard_conditions()
        else:
            start_state = self.trajectory[..., 0, :]
            goal_state = self.trajectory[..., -1, :]

            return {
                "start": start_state,
                "goal": goal_state,
            }

    def has_nan(self) -> bool:
        if self.context is None:
            return self.trajectory.isnan().any()

        return self.trajectory.isnan().any() or self.context.has_nan()

    @property
    def batch_size(self) -> int:
        if not self.is_batched:
            return 1
        return self.trajectory.shape[0]

    def slice(self, index: int | slice) -> TrajectorySample:
        """
        Slice a batched TrajectorySample.
        """
        if not self.is_batched:
            raise ValueError("Cannot slice a non-batched TrajectorySample")

        trajectory = self.trajectory[index]

        # Handle context slicing
        context = None
        if self.context is not None:
            if self.shared_context:
                context = self.context  # Shared context remains the same
            else:
                context = self.context.slice(index)

        # Determine if result is still batched
        is_batched = True
        if isinstance(index, int):
            is_batched = False

        metadata = None
        if self.metadata:
            metadata = {k: v[index] for k, v in self.metadata.items()}

        return self.__class__(
            trajectory=trajectory,
            context=context,
            metadata=metadata or {},
            is_normalized=self.is_normalized,
            is_batched=is_batched,
            shared_context=self.shared_context if is_batched else False
        )
