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
    masks: Dict[str, torch.Tensor]
    is_batched: bool = True

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
        Concatenates the embeddings and masks according to the concat_config.
        concat_config must be a dictionary with the following structure:
        {
            "<virtual_key>": {
                "keys": List[str] | None (== all keys)
            }
        }
        It is assumed that the values of the keys that are concatenated have the same shape
        except for the dimension after the batch dimension.
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
        """
        Returns a "null" context with the same structure/shapes as self,
        but with masks zeroed out. Useful for CFG unconditional pass.
        """

        zero_masks: Dict[str, torch.Tensor] = {}
        for k, m in self.masks.items():
            zero_masks[k] = torch.zeros_like(m)

        return EmbeddedContext(embeddings=self.embeddings, masks=zero_masks, is_batched=self.is_batched)

    def concat_batch(self, other: EmbeddedContext) -> EmbeddedContext:
        """
        Concatenate two EmbeddedContexts along the batch dimension (dim=0).
        Keys must match. Masks are concatenated when present.
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
        return any(v.isnan().any() for v in self.embeddings.values()) or any(v.isnan().any() for v in self.masks.values())

    def clone(self) -> EmbeddedContext:
        """
        Returns a clone of the EmbeddedContext.
        """
        return EmbeddedContext(
            embeddings={k: v.clone() for k, v in self.embeddings.items()},
            masks={k: v.clone() for k, v in self.masks.items()},
            is_batched=self.is_batched
        )

    def copy_(self, other: EmbeddedContext) -> None:
        """
        Copies the embeddings and masks from other to self.
        """
        assert self.embeddings.keys() == other.embeddings.keys()
        assert self.masks.keys() == other.masks.keys()
        assert self.is_batched == other.is_batched
        for k in self.embeddings.keys():
            self.embeddings[k].copy_(other.embeddings[k])
            self.masks[k].copy_(other.masks[k])
