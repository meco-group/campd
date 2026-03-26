"""
Context data container for trajectory datasets.

This module provides a type-safe container for context data that replaces
dictionary-based context handling throughout the codebase.
"""
from __future__ import annotations

from campd.data.normalization import DatasetNormalizer
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
import torch


@dataclass
class TrajectoryContext:
    """
    Container for trajectory context data.

    Stores context data with automatically generated masks for different context types.
    Each context type has associated data and a mask indicating valid entries.
    Masks are automatically generated based on data (all zeros or NaN = invalid).

    Args:
        data: Dictionary with context data. Can be:
                - {key: tensor} - masks auto-generated
                - {key: {'data': tensor, 'mask': tensor}} - explicit masks
        start: Start state tensor.
        goal: Goal state tensor.
        components: Optional mapping for virtual keys (aliases to slices of main keys).
                    Format: {alias_key: (parent_key, start_idx, end_idx)}
                    Example: {'cuboid_centers': ('cuboids', 0, 3)}
        is_normalized: Boolean indicating whether the context is normalized

    Example:
        >>> # Auto-generate masks
        >>> context = TrajectoryContext({
        ...     'boxes': torch.randn(5, 10),
        ...     'cylinders': torch.randn(3, 9),
        ...     'camera': torch.randn(1, 255,255,3)
        ... },
        ...     start=torch.randn(10),
        ...     goal=torch.randn(10))
        >>>
        >>> # Or provide explicit masks
        >>> context = TrajectoryContext({
        ...     'boxes': {'data': torch.randn(5, 10), 'mask': torch.ones(5, dtype=torch.bool)}
        ... })
        >>>
        >>> # Access data
        >>> boxes = context['boxes']
        >>> box_mask = context.get_mask('boxes')
    """

    _items: Dict[str, Dict[str, torch.Tensor]]
    components: Dict[str, Tuple[str, int, int]]

    start: torch.Tensor
    goal: torch.Tensor

    is_normalized: bool
    is_batched: bool

    def __init__(self,
                 data: Dict[str, Any],
                 start: torch.Tensor,
                 goal: torch.Tensor,
                 is_batched: bool,
                 components: Optional[Dict[str, Tuple[str, int, int]]] = None,
                 is_normalized: bool = False):
        """
        Initialize TrajectoryContext with data.

        Args:
            data: Dictionary with context data. Can be:
                  - {key: tensor} - masks auto-generated
                  - {key: {'data': tensor, 'mask': tensor}} - explicit masks
            start: Start state tensor.
            goal: Goal state tensor.
            is_batched: Boolean indicating whether the data is batched (default: False)
            components: Optional mapping for virtual keys (aliases to slices of main keys).
                        Format: {alias_key: (parent_key, start_idx, end_idx)}
                        Example: {'cuboid_centers': ('cuboids', 0, 3)}
            is_normalized: Boolean indicating whether the context is normalized
        """
        self._items = {}
        self.components = components or {}
        self.start = start
        self.goal = goal
        self.is_normalized = is_normalized
        self.is_batched = is_batched
        for key, value in data.items():
            if isinstance(value, dict) and 'data' in value:
                # Explicit data and mask provided
                tensor = value['data']
                mask = value.get('mask', self._generate_mask(tensor))
            else:
                # Just tensor provided, generate mask
                tensor = value
                mask = self._generate_mask(tensor)

            tensor = torch.nan_to_num(tensor, nan=0.0)
            self._items[key] = {'data': tensor, 'mask': mask}

    @property
    def batch_size(self) -> int:
        """Get batch size."""
        if self.is_batched:
            first_key = next(iter(self._items))
            return self._items[first_key]['data'].shape[0]
        else:
            return 0  # Not batched

    def _generate_mask(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Generate mask from tensor data.

        Args:
            tensor: Input tensor, shape [(n_batch), n_objects, ...features]

        Returns:
            Boolean mask, shape [(n_batch), n_objects] (True = valid, False = invalid)
        """
        if self.is_batched and tensor.ndim == 2:
            raise ValueError(
                "Tensor must have at least 3 dimensions for batched data.")
        start_dim = 2 if self.is_batched else 1
        flattened_tensor = tensor.flatten(start_dim=start_dim)
        all_zeros = (flattened_tensor == 0).all(dim=-1)
        has_nan = torch.isnan(flattened_tensor).any(dim=-1)
        return ~(all_zeros | has_nan)

    def __repr__(self) -> str:
        """String representation."""
        context_info = []
        for key in self.keys():
            shape = self._items[key]['data'].shape
            n_valid = self._items[key]['mask'].sum().item()
            n_total = self._items[key]['mask'].numel()
            context_info.append(
                f"{key}: {shape} ({n_valid}/{n_total} valid)")
        return f"TrajectoryContext({', '.join(context_info)})"

    def __len__(self) -> int:
        """Return number of context types."""
        return len(self._items)

    def __getitem__(self, key: str) -> torch.Tensor:
        """Get context data for a specific type."""
        if key in self._items:
            return self._items[key]['data']
        # elif key in self.components:
        #     parent_key, start, end = self.components[key]
        #     if parent_key not in self._items:
        #         raise KeyError(f"Virtual key '{key}' depends on missing parent '{parent_key}'")
        #     return self._items[parent_key]['data'][..., start:end]
        raise KeyError(key)

    def keys(self) -> List[str]:
        """Get list of context type names (physical keys only)."""
        return list(self._items.keys())

    def all_keys(self) -> List[str]:
        """Get list of all keys including virtual components."""
        return list(self._items.keys()) + list(self.components.keys())

    def get_mask(self, key: str) -> torch.Tensor:
        """Get mask for a specific context type."""
        if key in self._items:
            return self._items[key]['mask']
        elif key in self.components:
            parent_key, _, _ = self.components[key]
            if parent_key not in self._items:
                raise KeyError(
                    f"Virtual key '{key}' depends on missing parent '{parent_key}'")
            return self._items[parent_key]['mask']
        raise KeyError(key)

    def get_item(self, key: str, sub_key: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Get data for a specific context type.

        Args:
            key: Context key (e.g. 'cuboids')
            sub_key: Optional sub-component key (e.g. 'centers').
                     Can be the exact original field name or a unique suffix.

        Returns:
            Tensor data for the specific context type.
        """
        if sub_key is None:
            # Standard access
            if key in self._items:
                return self._items[key]['data']
            elif key in self.components:
                parent_key, start, end = self.components[key]
                if parent_key not in self._items:
                    raise KeyError(
                        f"Virtual key '{key}' depends on missing parent '{parent_key}'")
                parent_item = self._items[parent_key]
                return parent_item['data'][..., start:end]
            raise KeyError(key)
        else:
            # Hierarchical access: key is parent, sub_key is component
            # First, verify parent exists
            if key not in self._items:
                raise KeyError(f"Parent key '{key}' not found in context.")

            # 1. Check exact match in components (sub_key is the alias)
            if sub_key in self.components:
                parent, start, end = self.components[sub_key]
                if parent == key:
                    parent_item = self._items[key]
                    return parent_item['data'][..., start:end]

            raise KeyError(
                f"Sub_key '{sub_key}' not found for parent '{key}'.")

    def to(self, device: torch.device, non_blocking: bool = False) -> self.__class__:
        """Move context data to specified device."""
        new_data = {}
        for key, item in self._items.items():
            new_data[key] = {
                'data': item['data'].to(device, non_blocking=non_blocking),
                'mask': item['mask'].to(device, non_blocking=non_blocking)
            }
        return self.__class__(
            data=new_data,
            components=self.components,
            start=self.start.to(device, non_blocking=non_blocking),
            goal=self.goal.to(device, non_blocking=non_blocking),
            is_normalized=self.is_normalized,
            is_batched=self.is_batched
        )

    def expand(self, num_trajectories: int) -> self.__class__:
        """
        Expand a single TrajectoryContext object into a batched object.
        """
        assert not self.is_batched, "Context is already batched"

        expanded_data = {}
        for key, item in self._items.items():
            expanded_data[key] = {
                'data': item['data'].unsqueeze(0).expand(num_trajectories, *item['data'].shape),
                'mask': item['mask'].unsqueeze(0).expand(num_trajectories, *item['mask'].shape)
            }
        return self.__class__(
            data=expanded_data,
            components=self.components,
            start=self.start.unsqueeze(0).expand(num_trajectories, -1),
            goal=self.goal.unsqueeze(0).expand(num_trajectories, -1),
            is_normalized=self.is_normalized,
            is_batched=True
        )

    def repeat_interleave(self, repeats: int, dim: int = 0) -> self.__class__:
        """
        Repeat elements of a batched context.
        """
        assert self.is_batched, "Context must be batched to repeat_interleave"
        assert dim == 0, "Only dim=0 is supported for batched contexts"

        new_data = {}
        for key, item in self._items.items():
            new_data[key] = {
                'data': item['data'].repeat_interleave(repeats, dim=dim),
                'mask': item['mask'].repeat_interleave(repeats, dim=dim)
            }
        return self.__class__(
            data=new_data,
            components=self.components,
            start=self.start.repeat_interleave(repeats, dim=dim),
            goal=self.goal.repeat_interleave(repeats, dim=dim),
            is_normalized=self.is_normalized,
            is_batched=True
        )

    @classmethod
    def collate(cls, contexts: List[TrajectoryContext]) -> TrajectoryContext:
        """
        Collate a list of TrajectoryContext objects into a single batched object.
        Unique keys across all contexts are preserved. Missing keys are not padded (assumes consistent keys).
        Variable object counts (tensor dimension 1) are padded with zeros (masked out).

        Args:
            contexts: List of TrajectoryContext objects

        Returns:
            New TrajectoryContext with batched data (shape [B, N_max, D])
        """
        if not contexts:
            raise ValueError("No contexts provided")

        # assert all contexts are batched or all are single
        assert all(ctx.is_batched for ctx in contexts) or all(
            not ctx.is_batched for ctx in contexts), "All contexts must be batched or single"
        assert all(ctx.is_normalized for ctx in contexts) or all(
            not ctx.is_normalized for ctx in contexts), "All contexts must be normalized or not normalized"

        # 1. Merge _items
        # Collect all unique keys
        all_keys = set()
        for ctx in contexts:
            all_keys.update(ctx.keys())

        data = {}
        for key in all_keys:
            # Collect tensors for this key
            tensors = []
            masks = []
            max_objects = 0
            feature_dim = 0

            # First pass: determine max dimensions
            for ctx in contexts:
                # Use ._items directly to check physical existence
                if key in ctx._items:
                    t = ctx._items[key]['data']
                    max_objects = max(max_objects, t.shape[0])
                    feature_dim = t.shape[-1]

            # Second pass: pad and collect
            for ctx in contexts:
                if key in ctx._items:
                    t = ctx._items[key]['data']
                    m = ctx._items[key]['mask']
                    # Pad if necessary
                    if t.shape[0] < max_objects:
                        padding = torch.zeros(
                            max_objects - t.shape[0], feature_dim, device=t.device, dtype=t.dtype)
                        t = torch.cat([t, padding], dim=0)
                        padding_mask = torch.zeros(
                            max_objects - m.shape[0], device=m.device, dtype=m.dtype)
                        m = torch.cat([m, padding_mask], dim=0)
                    tensors.append(t)
                    masks.append(m)
                else:
                    # Missing key for this sample
                    # Create completely empty entry
                    t = torch.zeros(max_objects, feature_dim,
                                    dtype=torch.float32)
                    m = torch.zeros(max_objects, dtype=torch.float32)
                    tensors.append(t)
                    masks.append(m)

            # Stack into batch [B, N, D]
            data[key] = {
                'data': torch.stack(tensors, dim=0),
                'mask': torch.stack(masks, dim=0)
            }

        # 2. Merge Start and Goal
        merged_start = torch.stack([ctx.start for ctx in contexts])
        merged_goal = torch.stack([ctx.goal for ctx in contexts])

        # Propagate components from the first context (assumes homogeneity)
        components = contexts[0].components
        is_normalized = contexts[0].is_normalized

        return cls(data, components=components, start=merged_start,
                   goal=merged_goal, is_batched=True, is_normalized=is_normalized)

    def normalize(self, normalizer: DatasetNormalizer) -> self.__class__:
        """
        Normalize the context.
        """
        if self.is_normalized:
            return self

        normalized_data = {}
        for key in self.keys():
            mask = self.get_mask(key)
            norm_tensor = normalizer.normalize(
                self._items[key]['data'], f'context_{key}')
            normalized_data[key] = {'data': norm_tensor, 'mask': mask}

        normalized_start = normalizer.normalize(self.start, 'traj')
        normalized_goal = normalizer.normalize(self.goal, 'traj')

        return self.__class__(data=normalized_data,
                              components=self.components,
                              start=normalized_start,
                              goal=normalized_goal,
                              is_normalized=True,
                              is_batched=self.is_batched)

    def denormalize(self, normalizer: DatasetNormalizer) -> self.__class__:
        """
        Denormalize the context.
        """
        if not self.is_normalized:
            return self

        denormalized_data = {}
        for key in self.keys():
            mask = self.get_mask(key)
            denorm_tensor = normalizer.unnormalize(
                self._items[key]['data'], f'context_{key}')
            denormalized_data[key] = {'data': denorm_tensor, 'mask': mask}

        denormalized_start = normalizer.unnormalize(self.start, 'traj')
        denormalized_goal = normalizer.unnormalize(self.goal, 'traj')

        return self.__class__(data=denormalized_data,
                              components=self.components,
                              start=denormalized_start,
                              goal=denormalized_goal,
                              is_normalized=False,
                              is_batched=self.is_batched)

    def get_start(self) -> torch.Tensor:
        """Get start state."""
        return self.start

    def get_goal(self) -> torch.Tensor:
        """Get goal state."""
        return self.goal

    def get_hard_conditions(self) -> Dict[str, torch.Tensor]:
        """Get hard conditions (e.g., {"start": tensor, "goal": tensor})."""
        return {"start": self.start, "goal": self.goal}

    def __len__(self) -> int:
        """Return number of context types."""
        return len(self._items)

    def slice(self, index: Any) -> TrajectoryContext:
        """
        Slice the context data (e.g., to get a single trajectory from a batch).

        Args:
            index: Slice or index to apply to the first dimension of data and masks

        Returns:
            New TrajectoryContext with sliced data
        """
        if not self.is_batched:
            raise ValueError("Context is not batched")
        sliced_data = {}
        for key, item in self._items.items():
            sliced_data[key] = {
                'data': item['data'][index],
                'mask': item['mask'][index]
            }
        return self.__class__(
            data=sliced_data,
            components=self.components,
            start=self.start[index],
            goal=self.goal[index],
            is_normalized=self.is_normalized,
            is_batched=False
        )

    def has_nan(self) -> bool:
        for item in self._items.values():
            if item['data'].isnan().any():
                return True
            if item['mask'].isnan().any():
                return True
        if self.start.isnan().any():
            return True
        if self.goal.isnan().any():
            return True
        return False

    def which_has_nan(self) -> List[str]:
        nan_keys = []
        for key, item in self._items.items():
            if item['data'].isnan().any():
                nan_keys.append(key)
            if item['mask'].isnan().any():
                nan_keys.append(key + "_mask")
        if self.start.isnan().any():
            nan_keys.append("start")
        if self.goal.isnan().any():
            nan_keys.append("goal")
        return nan_keys

    def validate_batch_size(self, batch_size: int) -> None:
        """
        Validate that the context batch size matches the expected batch size.

        Args:
            batch_size: Expected batch size

        Raises:
            ValueError: If context is not batched or batch size does not match
        """
        if not self.is_batched:
            raise ValueError("Context is not batched")

        # Check batch size of first item
        if self._items:
            current_batch_size = self.batch_size
            if current_batch_size != batch_size:
                raise ValueError(
                    f"Context batch size {current_batch_size} does not match expected batch size {batch_size}")
        else:
            # Fallback to start/goal if no items
            if self.start.shape[0] != batch_size:
                raise ValueError(
                    f"Context batch size {self.start.shape[0]} does not match expected batch size {batch_size}")

    def clone(self) -> self.__class__:
        """
        Returns a clone of the TrajectoryContext.
        """
        cloned_data = {}
        for key, item in self._items.items():
            cloned_data[key] = {
                'data': item['data'].clone(),
                'mask': item['mask'].clone()
            }
        return self.__class__(
            data=cloned_data,
            start=self.start.clone(),
            goal=self.goal.clone(),
            is_batched=self.is_batched,
            components=self.components,
            is_normalized=self.is_normalized
        )

    def copy_from(self, other: TrajectoryContext) -> None:
        """
        Copy data from another context into this context.
        Assumes that the other context has the same structure as this context.
        Assumes that the other context is batched if this context is batched.
        Assumes that the number of sequences in the other context is lower than or equal to the number of sequences in this context.

        Args:
            other: Context to copy from
        """
        if self.is_batched != other.is_batched:
            raise ValueError("Contexts must have the same batched status")

        if self.is_batched:
            n_seq_idx = 1
        else:
            n_seq_idx = 0

        for key in self._items.keys():
            if key not in other._items:
                raise ValueError(f"Key {key} not found in other context")

            n_seq_self = self._items[key]['data'].shape[n_seq_idx]
            n_seq_other = other._items[key]['data'].shape[n_seq_idx]

            if n_seq_self < n_seq_other:
                raise ValueError(
                    f"Context has {n_seq_self} sequences, but other context has {n_seq_other} sequences")

            data_to_copy = other._items[key]['data']
            masks_to_copy = other._items[key]['mask']

            if n_seq_self > n_seq_other:
                # add zeros to the end
                pad_shape = list(data_to_copy.shape)
                pad_shape[n_seq_idx] = n_seq_self - n_seq_other

                padding_data = torch.zeros(
                    pad_shape, device=data_to_copy.device, dtype=data_to_copy.dtype)
                data_to_copy = torch.cat(
                    [data_to_copy, padding_data], dim=n_seq_idx)

                padding_mask_shape = pad_shape[:(n_seq_idx+1)]
                padding_mask = torch.zeros(
                    padding_mask_shape, device=masks_to_copy.device, dtype=masks_to_copy.dtype)
                masks_to_copy = torch.cat(
                    [masks_to_copy, padding_mask], dim=n_seq_idx)

            assert data_to_copy.shape[n_seq_idx] == n_seq_self
            assert masks_to_copy.shape[n_seq_idx] == n_seq_self

            self._items[key]['data'].copy_(
                data_to_copy, non_blocking=True)
            self._items[key]['mask'].copy_(
                masks_to_copy, non_blocking=True)

        self.start.copy_(other.start, non_blocking=True)
        self.goal.copy_(other.goal, non_blocking=True)

        self.is_normalized = other.is_normalized
        self.components = other.components

    def copy_(self, other: TrajectoryContext) -> None:
        """Alias for copy_from to match PyTorch in-place copy convention."""
        self.copy_from(other)
