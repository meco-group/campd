
"""
Core neural network layers and building blocks for CAMPD architectures.

Includes standard MLP implementations, Temporal U-Net residual blocks,
attention mechanisms, and various normalizations/activations.
"""

import math
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from campd.architectures.registry import MODULES

from pydantic import BaseModel
from typing import Literal


#: Dictionary mapping activation function names to their PyTorch module classes.
ACTIVATIONS = {
    'relu': nn.ReLU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    'leaky_relu': nn.LeakyReLU,
    'elu': nn.ELU,
    'prelu': nn.PReLU,
    'softplus': nn.Softplus,
    'mish': nn.Mish,
    'identity': nn.Identity
}


class MLP1DCfg(BaseModel):
    """Configuration for a 1D Multi-Layer Perceptron."""
    in_dim: int
    """Input feature dimension."""
    out_dim: int
    """Output feature dimension."""
    hidden_dim: int = 16
    """Dimension of hidden layers."""
    n_layers: int = 1
    """Number of hidden layers."""
    act: Literal[tuple(ACTIVATIONS.keys())] = 'mish'
    """Activation function name (e.g., 'relu', 'mish', 'elu')."""
    layer_norm: bool = True
    """Whether to apply layer normalization after linear layers."""


@MODULES.register('MLP1D')
class MLP1D(nn.Module):
    """A standard 1D Multi-Layer Perceptron (MLP) module.
    
    Constructs a sequence of Linear -> [LayerNorm] -> Activation layers.
    """
    def __init__(self, config: MLP1DCfg):
        super(MLP1D, self).__init__()

        act_func = ACTIVATIONS[config.act]
        layers = [
            nn.Linear(config.in_dim, config.hidden_dim),
            nn.LayerNorm(
                config.hidden_dim) if config.layer_norm else nn.Identity(),
            act_func(),
        ]

        for _ in range(config.n_layers - 1):
            layers += [
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.LayerNorm(
                    config.hidden_dim) if config.layer_norm else nn.Identity(),
                act_func(),
            ]

        layers.append(nn.Linear(config.hidden_dim, config.out_dim))

        self._network = nn.Sequential(
            *layers
        )
        self.in_features = config.in_dim

    @classmethod
    def from_config(cls, config: MLP1DCfg | dict):
        """Instantiates an MLP1D from a configuration object or dictionary.
        
        Args:
            config (MLP1DCfg | dict): Configuration for the MLP1D.
            
        Returns:
            MLP1D: An instantiated MLP1D module.
        """
        config = MLP1DCfg.model_validate(config)
        return cls(config)

    def forward(self, x):
        """Forward pass through the MLP.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor representing the processed features.
        """
        return self._network(x)


########################################################################################################################
# Modules Temporal Unet
########################################################################################################################

class Residual(nn.Module):
    """Applies a residual connection around a given function/module.
    
    Args:
        fn (nn.Module): The module to wrap.
    """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class PreNorm(nn.Module):
    """Applies LayerNorm before a given function/module.
    
    Args:
        dim (int): Feature dimension for normalization.
        fn (nn.Module): The module to wrap.
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class LayerNorm(nn.Module):
    """Custom LayerNorm implementation avoiding standard PyTorch constraints.
    
    Args:
        dim (int): Feature dimension.
        eps (float): Small value to avoid division by zero.
    """
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class TimeEncoder(nn.Module):
    """Encodes time steps using sinusoidal embeddings followed by an MLP.
    
    Args:
        dim (int): Base embedding dimension.
        dim_out (int): Output embedding dimension.
    """
    def __init__(self, dim, dim_out):
        super().__init__()
        self.encoder = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim_out)
        )

    def forward(self, x):
        return self.encoder(x)


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings for time/position encoding.
    
    Args:
        dim (int): Embedding dimension. Must be an even number.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    """Downsamples a 1D sequence using a strided convolution.
    
    Args:
        dim (int): Number of channels.
    """
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    """Upsamples a 1D sequence using a transposed convolution.
    
    Args:
        dim (int): Number of channels.
    """
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            dim, dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """A convolutional block applying Conv1d -> GroupNorm -> Mish.
    
    Args:
        inp_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolving kernel.
        padding (int, optional): Zero-padding added to both sides of the input.
        n_groups (int): Number of groups for GroupNorm. Defaults to 8.
    """

    def __init__(self, inp_channels, out_channels, kernel_size, padding=None, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, stride=1,
                      padding=padding if padding is not None else kernel_size // 2),
            Rearrange(
                'batch channels n_support_points -> batch channels 1 n_support_points'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange(
                'batch channels 1 n_support_points -> batch channels n_support_points'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ResidualTemporalBlock(nn.Module):
    """A residual temporal block with conditioning for diffusion models.
    
    Args:
        inp_channels (int): Input channel dimension.
        out_channels (int): Output channel dimension.
        cond_embed_dim (int): Conditioning embedding dimension.
        n_support_points (int): Number of support points (sequence length).
        kernel_size (int): Size of the convolving kernel. Defaults to 5.
    """
    def __init__(self, inp_channels, out_channels, cond_embed_dim, n_support_points, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size,
                        n_groups=group_norm_n_groups(out_channels)),
            Conv1dBlock(out_channels, out_channels, kernel_size,
                        n_groups=group_norm_n_groups(out_channels)),
        ])

        # Without context conditioning, cond_mlp handles only time embeddings
        self.cond_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, kernel_size=1, stride=1, padding=0) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, c):
        '''
            x : [ batch_size x inp_channels x n_support_points ]
            c : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x n_support_points ]
        '''
        h = self.blocks[0](x) + self.cond_mlp(c)
        h = self.blocks[1](h)
        res = self.residual_conv(x)
        out = h + res

        return out


def group_norm_n_groups(n_channels, target_n_groups=8):
    """Safely computes the number of groups for GroupNorm based on channels.
    
    Finds a valid number of groups (divisible by n_channels) close to target.
    
    Args:
        n_channels (int): Number of channels.
        target_n_groups (int): Target number of groups. Defaults to 8.
        
    Returns:
        int: Realized number of groups for GroupNorm.
    """
    if n_channels < target_n_groups:
        return 1
    for n_groups in range(target_n_groups, target_n_groups + 10):
        if n_channels % n_groups == 0:
            return n_groups
    return 1
