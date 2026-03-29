"""
Embedded context representations and configuration for the context encoder.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import torch


ConcatConfig = Dict[str, Optional[List[str]]]


@dataclass
class EmbeddedContext:
    """
    Container for embedded context data.

    Stores embeddings resulting from the ContextEncoder.
    Can handle both global embeddings (single vector) and per-object embeddings.
    """

    embeddings: Dict[str, torch.Tensor]
    """Dictionary mapping context keys to their embedding tensors."""
    masks: Dict[str, torch.Tensor]
    """Dictionary mapping context keys to their validity masks."""
    is_batched: bool = True
    """Whether the embeddings have a batch dimension."""

    def __getitem__(self, key: str) -> torch.Tensor:
        if key in self.embeddings:
            return self.embeddings[key]
        raise KeyError(f"Key '{key}' not found in embeddings.")

    def get_mask(self, key: str) -> Optional[torch.Tensor]:
        if key in self.masks:
            return self.masks[key]
        raise KeyError(f"Key '{key}' not found in masks.")

    def keys(self):
        return list(self.embeddings.keys())

    def concat_keys(self, concat_keys_config: ConcatConfig) -> EmbeddedContext:
        """
        Concatenate embeddings and masks based on a configuration mapping.

        The input configuration defines which existing keys should be combined
        into each virtual key.

        Args:
            concat_keys_config (ConcatConfig):
                A mapping that specifies how keys should be concatenated. The
                expected structure is::

                    {
                        "<virtual_key>": {
                            "keys": List[str] | None
                        }
                    }

                - If "keys" is a list, only those keys will be concatenated.
                - If "keys" is None, all available keys will be concatenated.

        Returns:
            EmbeddedContext:
                A new context object containing the concatenated embeddings and
                masks under the specified virtual keys.

        Notes:
            - All tensors to be concatenated must have identical shapes except
              for the feature dimension (dimension after batch).
            - The batch dimension must be consistent across all tensors.
            - Concatenation is performed along the feature dimension.
        """
        concatenated_embeddings = {}
        concatenated_masks = {}

        for name, keys in concat_keys_config.items():
            if keys is None:
                keys = list(self.embeddings.keys())
            concatenated_embeddings[name], concatenated_masks[name] = self._concat_items(
                keys)

        return EmbeddedContext(concatenated_embeddings, concatenated_masks, is_batched=self.is_batched)

    def _concat_items(self, keys: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(keys) == 0:
            raise ValueError("No keys provided for concatenation.")

        dim = 1 if self.is_batched else 0

        tensors = [self.embeddings[k] for k in keys]
        masks = [self.masks[k] for k in keys]

        return torch.cat(tensors, dim=dim), torch.cat(masks, dim=dim)

    def append(self, key: str, embedding: torch.Tensor, mask: torch.Tensor):
        """Appends an embedding and mask to an existing key's tensor.
        
        Args:
            key (str): The context key.
            embedding (torch.Tensor): The new embedding tensor to append.
            mask (torch.Tensor): The new mask tensor to append.
        """
        assert key in self.embeddings, f"Key '{key}' not found in embeddings."
        assert key in self.masks, f"Key '{key}' not found in masks."

        dim = 1 if self.is_batched else 0

        self.embeddings[key] = torch.cat(
            [self.embeddings[key], embedding], dim=dim)
        self.masks[key] = torch.cat([self.masks[key], mask], dim=dim)

    def to(self, device: torch.device) -> EmbeddedContext:
        new_embeddings = {k: v.to(device) for k, v in self.embeddings.items()}
        new_masks = {k: v.to(device) for k, v in self.masks.items()}
        # Create new instance with same config, clear cache
        return EmbeddedContext(embeddings=new_embeddings, masks=new_masks, is_batched=self.is_batched)

    def null(self) -> EmbeddedContext:
        """Creates a "null" context with zeroed-out masks.
        
        Returns a context with the same structure and shapes as `self`,
        but with all masks set to zero. Useful for the CFG unconditional pass.
        
        Returns:
            EmbeddedContext: The null context.
        """

        zero_masks: Dict[str, torch.Tensor] = {}
        for k, m in self.masks.items():
            zero_masks[k] = torch.zeros_like(m)

        return EmbeddedContext(embeddings=self.embeddings, masks=zero_masks, is_batched=self.is_batched)

    def concat_batch(self, other: EmbeddedContext) -> EmbeddedContext:
        """Concatenates two EmbeddedContexts along the batch dimension (dim=0).
        
        Args:
            other (EmbeddedContext): Another context to concatenate with.
            
        Returns:
            EmbeddedContext: The concatenated context.
            
        Raises:
            ValueError: If either context is not batched.
            KeyError: If embedding or mask keys do not match.
        """
        if self.is_batched is not True or other.is_batched is not True:
            raise ValueError(
                "concat_batch expects both EmbeddedContexts to be batched (is_batched=True).")
        if self.embeddings.keys() != other.embeddings.keys():
            raise KeyError("concat_batch requires matching embedding keys.")
        if self.masks.keys() != other.masks.keys():
            raise KeyError("concat_batch requires matching mask keys.")

        embeddings = {k: torch.cat(
            (self.embeddings[k], other.embeddings[k]), dim=0) for k in self.embeddings.keys()}

        masks: Dict[str, torch.Tensor] = {}
        for k in self.masks.keys():
            m1, m2 = self.masks[k], other.masks[k]
            assert m1 is not None and m2 is not None, f"concat_batch mask mismatch for key '{k}'"
            masks[k] = torch.cat((m1, m2), dim=0)

        return EmbeddedContext(embeddings=embeddings, masks=masks, is_batched=True)

    def has_nan(self) -> bool:
        """Checks if any embedding or mask tensor contains NaN values.
        
        Returns:
            bool: True if NaNs are present, False otherwise.
        """
        return any(v.isnan().any() for v in self.embeddings.values()) or any(v.isnan().any() for v in self.masks.values())

    def clone(self) -> EmbeddedContext:
        """Creates a deep copy of the EmbeddedContext.
        
        Returns:
            EmbeddedContext: The cloned context.
        """
        return EmbeddedContext(
            embeddings={k: v.clone() for k, v in self.embeddings.items()},
            masks={k: v.clone() for k, v in self.masks.items()},
            is_batched=self.is_batched
        )

    def copy_(self, other: EmbeddedContext) -> None:
        """Copies the embeddings and masks from another context in-place.
        
        Args:
            other (EmbeddedContext): The source context to copy data from.
        """
        assert self.embeddings.keys() == other.embeddings.keys()
        assert self.masks.keys() == other.masks.keys()
        assert self.is_batched == other.is_batched
        for k in self.embeddings.keys():
            self.embeddings[k].copy_(other.embeddings[k])
            self.masks[k].copy_(other.masks[k])
