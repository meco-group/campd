from __future__ import annotations

from typing import Mapping, Optional, Protocol
import torch.nn as nn
from campd.data.context import TrajectoryContext
from campd.data.embedded_context import EmbeddedContext, ConcatConfig
from campd.utils.registry import Spec
from campd.architectures.registry import CONTEXT_NETS, MODULES
from pydantic import BaseModel
import torch


class KeyNetModule(Protocol):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        ...


class ContextEncoderCfg(BaseModel):
    key_networks: Mapping[str, Spec[KeyNetModule]]
    concat_config: Optional[ConcatConfig] = None
    include_start_goal: bool = False


@CONTEXT_NETS.register("ContextEncoder")
class ContextEncoder(nn.Module):
    """
    Encodes TrajectoryContext into EmbeddedContext using a dedicated network for each context key.
    """

    def __init__(self, config: ContextEncoderCfg):
        """
        Args:
            config: ContextEncoderCfg object or dictionary.
        """
        super().__init__()

        self.config = config

        key_networks = {}
        for key, spec in self.config.key_networks.items():
            key_networks[key] = spec.build_from(MODULES)

        if self.config.include_start_goal:
            assert "start" in key_networks and "goal" in key_networks

        self.key_networks: nn.ModuleDict[str,
                                         KeyNetModule] = nn.ModuleDict(key_networks)

    @property
    def context_keys(self) -> list[str]:
        return list(self.key_networks.keys())

    @property
    def context_dims(self) -> dict[str, int]:
        return {key: net.in_features for key, net in self.key_networks.items()}

    @classmethod
    def from_config(cls, config: ContextEncoderCfg | dict) -> ContextEncoder:
        """
        Factory method to create ContextEncoder from config.
        """
        config = ContextEncoderCfg.model_validate(config)
        return cls(config)

    def forward(self, context: TrajectoryContext) -> EmbeddedContext:
        """
        Args:
            context: TrajectoryContext to encode.
        Returns:
            EmbeddedContext containing the encoded context.
        """
        assert context.is_normalized

        embeddings: dict[str, torch.Tensor] = {}
        masks: dict[str, torch.Tensor] = {}

        for key in context.keys():
            if key in self.key_networks:
                data = context[key]  # (b, n_seq, context_dim)
                b, n_seq, context_dim = data.shape

                out = self.key_networks[key](data)
                embeddings[key] = out

                mask = context.get_mask(key)
                masks[key] = mask

        if self.config.include_start_goal:
            embeddings["start"] = self.key_networks["start"](
                context.start.unsqueeze(1))
            embeddings["goal"] = self.key_networks["goal"](
                context.goal.unsqueeze(1))
            masks["start"] = torch.ones(
                (context.batch_size, 1), device=context.start.device, dtype=torch.bool)
            masks["goal"] = torch.ones(
                (context.batch_size, 1), device=context.start.device, dtype=torch.bool)

        embedded_context = EmbeddedContext(
            embeddings=embeddings,
            masks=masks,
            is_batched=context.is_batched
        )

        if self.config.concat_config:
            return embedded_context.concat_keys(self.config.concat_config)

        return embedded_context
