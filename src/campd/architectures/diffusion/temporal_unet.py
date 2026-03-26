from typing import Tuple
import einops
import torch
import torch.nn as nn
from campd.architectures.diffusion.base import ReverseDiffusionNetwork
from campd.data.embedded_context import EmbeddedContext

from campd.architectures.layers.layers import Downsample1d, Conv1dBlock, Upsample1d, \
    ResidualTemporalBlock, TimeEncoder, group_norm_n_groups
from campd.architectures.layers.attention import SpatialTransformer
from ..registry import REVERSE_NETS

from pydantic import BaseModel


class TemporalUnetCfg(BaseModel):
    n_support_points: int
    state_dim: int
    unet_input_dim: int = 32
    dim_mults: Tuple[int, ...] = (1, 2, 4, 8)
    time_emb_dim: int = 32
    enable_conditioning: bool = False
    conditioning_embed_dim: int = 0
    attention_num_heads: int = 0
    attention_dim_head: int = 0
    add_time_emb_to_conditioning: bool = False


@REVERSE_NETS.register('TemporalUnet')
class TemporalUnet(ReverseDiffusionNetwork):
    conditioning_key = 'all'

    def __init__(self, config: TemporalUnetCfg):
        super().__init__()
        self.config = config

        dims = [self.config.state_dim, *
                map(lambda m: self.config.unet_input_dim * m, self.config.dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        # print(f'[ models/temporal ] Channel dimensions: {in_out}')

        # Networks
        self.time_mlp = TimeEncoder(32, self.config.time_emb_dim)

        # Unet
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        n_support_points = self.config.n_support_points

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(
                    dim_in, dim_out, self.config.time_emb_dim, n_support_points=n_support_points),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                n_support_points = n_support_points // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(
            mid_dim, mid_dim, self.config.time_emb_dim, n_support_points=n_support_points)
        self.mid_attention = SpatialTransformer(mid_dim, self.config.attention_num_heads, self.config.attention_dim_head, depth=1,
                                                context_dim=self.config.conditioning_embed_dim) if self.config.enable_conditioning else nn.Identity()
        self.mid_block2 = ResidualTemporalBlock(
            mid_dim, mid_dim, self.config.time_emb_dim, n_support_points=n_support_points)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(
                    dim_out * 2, dim_in, self.config.time_emb_dim, n_support_points=n_support_points),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

            if not is_last:
                n_support_points = n_support_points * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(self.config.unet_input_dim, self.config.unet_input_dim,
                        kernel_size=5, n_groups=group_norm_n_groups(self.config.unet_input_dim)),
            nn.Conv1d(self.config.unet_input_dim, self.config.state_dim, 1),
        )

        if self.config.add_time_emb_to_conditioning:
            assert self.config.conditioning_embed_dim == self.config.time_emb_dim, "Conditioning embed dim must be equal to time emb dim if add_time_emb_to_conditioning is True"
        print(
            f'[ TemporalUnet ] Number of parameters: {sum(p.numel() for p in self.parameters())}')

    @classmethod
    def from_config(cls, config: TemporalUnetCfg | dict):
        config = TemporalUnetCfg.model_validate(config)
        return cls(config)

    def forward(self, x: torch.Tensor, t: torch.Tensor, embedded_context_batch: EmbeddedContext) -> torch.Tensor:
        """
        x : [ batch x horizon x state_dim ]
        t : [ batch ] (int or float usually, but here Tensor)
        embedded_context_batch: EmbeddedContext, assumed to have a key called "all" that contains all the
        embeddings stacked on the second dimension
        """
        b, h, d = x.shape

        t_emb = self.time_mlp(t)

        if self.config.enable_conditioning:
            if self.config.add_time_emb_to_conditioning:
                embedded_context_batch.append(
                    self.conditioning_key,
                    t_emb.unsqueeze(1),
                    torch.ones((b, 1), dtype=torch.bool, device=t.device)
                )

            ctx_all = embedded_context_batch["all"]
            assert ctx_all.shape[2] == self.config.conditioning_embed_dim, "Shape mismatch between embedded context and config"
            assert ctx_all.shape[0] == b, "Shape mismatch between embedded context and config"

        x = einops.rearrange(x, 'b h c -> b c h')

        history = []
        for resnet, downsample in self.downs:
            x = resnet(x, t_emb)
            history.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t_emb)
        if self.config.enable_conditioning:
            x = self.mid_attention(
                x, context=ctx_all, mask=embedded_context_batch.get_mask("all"))
        x = self.mid_block2(x, t_emb)

        for resnet, upsample in self.ups:
            x = torch.cat((x, history.pop()), dim=1)
            x = resnet(x, t_emb)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b c h -> b h c')

        return x
