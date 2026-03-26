from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from campd.data.embedded_context import EmbeddedContext


class ReverseDiffusionNetwork(nn.Module, ABC):
    """
    Abstract base class for neural networks that learn the reverse diffusion process.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor, embedded_context_batch: EmbeddedContext) -> torch.Tensor:
        """
        Forward pass of the reserve diffusion network.

        Args:
            x (torch.Tensor): The batched noisy input data.
            t (torch.Tensor): The batched diffusion timestep(s).
            embedded_context_batch (EmbeddedContext): The embedded context for the batch.

        Returns:
            torch.Tensor: The batched predicted noise or denoised data.
        """
        pass
